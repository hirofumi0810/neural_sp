#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Train the RNNLM."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cProfile
import math
import numpy as np
import os
from setproctitle import setproctitle
import shutil
import time
import torch
from tqdm import tqdm

from neural_sp.bin.args_lm import parse
from neural_sp.bin.lr_controller import Controller
from neural_sp.bin.train_utils import load_config
from neural_sp.bin.train_utils import save_config
from neural_sp.bin.train_utils import set_logger
from neural_sp.bin.reporter import Reporter
from neural_sp.datasets.loader_lm import Dataset
from neural_sp.evaluators.ppl import eval_ppl
from neural_sp.models.data_parallel import CustomDataParallel
from neural_sp.models.rnnlm.rnnlm_seq import SeqRNNLM
from neural_sp.utils.general import mkdir_join


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


def main():

    args = parse()

    # Load a conf file
    if args.resume:
        conf = load_config(os.path.join(args.resume, 'conf.yml'))
        for k, v in conf.items():
            setattr(args, k, v)

    # Load dataset
    train_set = Dataset(tsv_path=args.train_set,
                        dict_path=args.dict,
                        unit=args.unit,
                        wp_model=args.wp_model,
                        batch_size=args.batch_size * args.n_gpus,
                        n_epochs=args.n_epochs,
                        bptt=args.bptt,
                        shuffle=False)
    dev_set = Dataset(tsv_path=args.dev_set,
                      dict_path=args.dict,
                      unit=args.unit,
                      wp_model=args.wp_model,
                      batch_size=args.batch_size * args.n_gpus,
                      bptt=args.bptt,
                      shuffle=True)
    eval_sets = []
    for s in args.eval_sets:
        eval_sets += [Dataset(tsv_path=s,
                              dict_path=args.dict,
                              unit=args.unit,
                              wp_model=args.wp_model,
                              batch_size=1,
                              bptt=args.bptt)]

    args.vocab = train_set.vocab

    # Model setting
    model = SeqRNNLM(args)
    dir_name = args.rnn_type
    dir_name += str(args.n_units) + 'H'
    dir_name += str(args.n_projs) + 'P'
    dir_name += str(args.n_layers) + 'L'
    dir_name += '_emb' + str(args.emb_dim)
    dir_name += '_' + args.optimizer
    dir_name += '_lr' + str(args.learning_rate)
    dir_name += '_bs' + str(args.batch_size)
    dir_name += '_bptt' + str(args.bptt)
    if args.tie_embedding:
        dir_name += '_tie'
    if args.residual:
        dir_name += '_residual'
    if args.use_glu:
        dir_name += '_glu'
    if args.backward:
        dir_name += '_bwd'

    if args.resume:
        raise NotImplementedError
    else:
        # Set save path
        save_path = mkdir_join(args.model, '_'.join(os.path.basename(args.train_set).split('.')[:-1]), dir_name)
        model.set_save_path(save_path)  # avoid overwriting

        # Save the conf file as a yaml file
        save_config(vars(args), os.path.join(model.save_path, 'conf.yml'))

        # Save the dictionary & wp_model
        shutil.copy(args.dict, os.path.join(model.save_path, 'dict.txt'))
        if args.unit == 'wp':
            shutil.copy(args.wp_model, os.path.join(model.save_path, 'wp.model'))

        # Setting for logging
        logger = set_logger(os.path.join(model.save_path, 'train.log'), key='training')

        for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
            logger.info('%s: %s' % (k, str(v)))

        # Count total parameters
        for n in sorted(list(model.num_params_dict.keys())):
            nparams = model.num_params_dict[n]
            logger.info("%s %d" % (n, nparams))
        logger.info("Total %.2f M parameters" % (model.total_parameters / 1000000))
        logger.info(model)

        # Set optimizer
        model.set_optimizer(optimizer=args.optimizer,
                            learning_rate=float(args.learning_rate),
                            weight_decay=float(args.weight_decay))

        epoch, step = 1, 1
        ppl_dev_best = 10000

        # Set learning rate controller
        lr_controller = Controller(learning_rate=float(args.learning_rate),
                                   decay_type=args.decay_type,
                                   decay_start_epoch=args.decay_start_epoch,
                                   decay_rate=args.decay_rate,
                                   decay_patient_n_epochs=args.decay_patient_n_epochs,
                                   lower_better=True,
                                   best_value=ppl_dev_best)

    train_set.epoch = epoch - 1  # start from index:0

    # GPU setting
    if args.n_gpus >= 1:
        model = CustomDataParallel(model,
                                   device_ids=list(range(0, args.n_gpus, 1)),
                                   deterministic=False,
                                   benchmark=True)
        model.cuda()

    logger.info('PID: %s' % os.getpid())
    logger.info('USERNAME: %s' % os.uname()[1])

    # Set process name
    if args.job_name:
        setproctitle(args.job_name)
    else:
        setproctitle(dir_name)

    # Set reporter
    reporter = Reporter(model.module.save_path, tensorboard=True)

    hidden = None
    start_time_train = time.time()
    start_time_epoch = time.time()
    start_time_step = time.time()
    not_improved_epoch = 0
    pbar_epoch = tqdm(total=len(train_set))
    while True:
        # Compute loss in the training set
        ys_train, is_new_epoch = train_set.next()

        model.module.optimizer.zero_grad()
        loss, hidden, reporter = model(ys_train, hidden, reporter)
        if len(model.device_ids) > 1:
            loss.backward(torch.ones(len(model.device_ids)))
        else:
            loss.backward()
        loss.detach()  # Trancate the graph
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.clip_grad_norm)
        model.module.optimizer.step()
        loss_train = loss.item()
        del loss
        hidden = model.module.repackage_hidden(hidden)
        reporter.step(is_eval=False)

        if step % args.print_step == 0:
            # Compute loss in the dev set
            ys_dev = dev_set.next()[0]
            loss, _, reporter = model(ys_dev, None, reporter, is_eval=True)
            loss_dev = loss.item()
            del loss
            reporter.step(is_eval=True)

            duration_step = time.time() - start_time_step
            logger.info("step:%d(ep:%.2f) loss:%.3f(%.3f)/ppl:%.3f(%.3f)/lr:%.5f/bs:%d (%.2f min)" %
                        (step, train_set.epoch_detail, loss_train, loss_dev,
                         math.exp(loss_train), math.exp(loss_dev),
                         lr_controller.lr, len(ys_train), duration_step / 60))
            start_time_step = time.time()
        step += args.n_gpus
        pbar_epoch.update(np.prod(ys_train.shape))

        # Save fugures of loss and accuracy
        if step % (args.print_step * 10) == 0:
            reporter.snapshot()

        # Save checkpoint and evaluate model per epoch
        if is_new_epoch:
            duration_epoch = time.time() - start_time_epoch
            logger.info('========== EPOCH:%d (%.2f min) ==========' % (epoch, duration_epoch / 60))

            if epoch < args.eval_start_epoch:
                # Save the model
                model.module.save_checkpoint(
                    model.module.save_path, lr_controller,
                    epoch, step - 1, ppl_dev_best)
            else:
                start_time_eval = time.time()
                # dev
                ppl_dev = eval_ppl([model.module], dev_set,
                                   batch_size=1, bptt=args.bptt)
                logger.info('PPL (%s): %.3f' % (dev_set.set, ppl_dev))

                # Update learning rate
                model.module.optimizer = lr_controller.decay(
                    model.module.optimizer,
                    epoch=epoch, value=ppl_dev)

                if ppl_dev < ppl_dev_best:
                    ppl_dev_best = ppl_dev
                    not_improved_epoch = 0
                    logger.info('||||| Best Score |||||')

                    # Save the model
                    model.module.save_checkpoint(
                        model.module.save_path, lr_controller,
                        epoch, step - 1, ppl_dev_best)

                    # test
                    ppl_test_avg = 0.
                    for eval_set in eval_sets:
                        ppl_test = eval_ppl([model.module], eval_set,
                                            batch_size=1, bptt=args.bptt)
                        logger.info('PPL (%s): %.3f' % (eval_set.set, ppl_test))
                        ppl_test_avg += ppl_test
                    if len(eval_sets) > 0:
                        logger.info('PPL (avg.): %.3f' % (ppl_test_avg / len(eval_sets)))
                else:
                    not_improved_epoch += 1

                duration_eval = time.time() - start_time_eval
                logger.info('Evaluation time: %.2f min' % (duration_eval / 60))

                # Early stopping
                if not_improved_epoch == args.not_improved_patient_n_epochs:
                    break

                # Convert to fine-tuning stage
                if epoch == args.convert_to_sgd_epoch:
                    model.module.set_optimizer('sgd',
                                               learning_rate=args.learning_rate,
                                               weight_decay=float(args.weight_decay))
                    lr_controller = Controller(learning_rate=args.learning_rate,
                                               decay_type='epoch',
                                               decay_start_epoch=epoch,
                                               decay_rate=0.5,
                                               lower_better=True)
                    logger.info('========== Convert to SGD ==========')

            pbar_epoch = tqdm(total=len(train_set))

            if epoch == args.n_epochs:
                break

            start_time_step = time.time()
            start_time_epoch = time.time()
            epoch += 1

    duration_train = time.time() - start_time_train
    logger.info('Total time: %.2f hour' % (duration_train / 3600))

    if reporter.tensorboard:
        reporter.tf_writer.close()
    pbar_epoch.close()

    return model.module.save_path


if __name__ == '__main__':
    # Setting for profiling
    pr = cProfile.Profile()
    save_path = pr.runcall(main)
    pr.dump_stats(os.path.join(save_path, 'train.profile'))
