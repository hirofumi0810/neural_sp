#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Train the LM."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cProfile
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
from neural_sp.bin.train_utils import set_save_path
from neural_sp.bin.train_utils import load_checkpoint
from neural_sp.bin.train_utils import save_checkpoint
from neural_sp.bin.reporter import Reporter
from neural_sp.datasets.loader_lm import Dataset
from neural_sp.evaluators.ppl import eval_ppl
from neural_sp.models.data_parallel import CustomDataParallel
from neural_sp.models.lm.lm import select_lm
from neural_sp.utils import mkdir_join


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


def main():

    args = parse()

    # Load a conf file
    if args.resume:
        conf = load_config(os.path.join(os.path.dirname(args.resume), 'conf.yml'))
        for k, v in conf.items():
            if k != 'resume':
                setattr(args, k, v)

    # Load dataset
    train_set = Dataset(corpus=args.corpus,
                        tsv_path=args.train_set,
                        dict_path=args.dict,
                        nlsyms=args.nlsyms,
                        unit=args.unit,
                        wp_model=args.wp_model,
                        batch_size=args.batch_size * args.n_gpus,
                        n_epochs=args.n_epochs,
                        min_n_tokens=args.min_n_tokens,
                        bptt=args.bptt,
                        backward=args.backward,
                        serialize=args.serialize)
    dev_set = Dataset(corpus=args.corpus,
                      tsv_path=args.dev_set,
                      dict_path=args.dict,
                      nlsyms=args.nlsyms,
                      unit=args.unit,
                      wp_model=args.wp_model,
                      batch_size=args.batch_size * args.n_gpus,
                      bptt=args.bptt,
                      backward=args.backward,
                      serialize=args.serialize)
    eval_sets = []
    for s in args.eval_sets:
        eval_sets += [Dataset(corpus=args.corpus,
                              tsv_path=s,
                              dict_path=args.dict,
                              nlsyms=args.nlsyms,
                              unit=args.unit,
                              wp_model=args.wp_model,
                              batch_size=1,
                              bptt=args.bptt,
                              backward=args.backward,
                              serialize=args.serialize)]

    args.vocab = train_set.vocab

    # Set save path
    if args.resume:
        save_path = os.path.dirname(args.resume)
        dir_name = os.path.basename(save_path)
    else:
        dir_name = make_model_name(args)
        save_path = mkdir_join(args.model, '_'.join(os.path.basename(args.train_set).split('.')[:-1]), dir_name)
        save_path = set_save_path(save_path)  # avoid overwriting

    # Set logger
    logger = set_logger(os.path.join(save_path, 'train.log'), key='training')

    # Model setting
    model = select_lm(args)
    model.save_path = save_path

    if args.resume:
        # Set optimizer
        epoch = int(args.resume.split('-')[-1])
        model.set_optimizer(optimizer='sgd' if epoch > conf['convert_to_sgd_epoch'] + 1 else conf['optimizer'],
                            learning_rate=float(conf['learning_rate']),  # on-the-fly
                            weight_decay=float(conf['weight_decay']))

        # Restore the last saved model
        model, checkpoint = load_checkpoint(model, args.resume, resume=True)
        lr_controller = checkpoint['lr_controller']
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        ppl_dev_best = checkpoint['metric_dev_best']

        # Resume between convert_to_sgd_epoch and convert_to_sgd_epoch + 1
        if epoch == conf['convert_to_sgd_epoch'] + 1:
            model.set_optimizer(optimizer='sgd',
                                learning_rate=args.learning_rate,
                                weight_decay=float(conf['weight_decay']))
            logger.info('========== Convert to SGD ==========')
    else:
        # Save the conf file as a yaml file
        save_config(vars(args), os.path.join(model.save_path, 'conf.yml'))

        # Save the nlsyms, dictionar, and wp_model
        if args.nlsyms:
            shutil.copy(args.nlsyms, os.path.join(model.save_path, 'nlsyms.txt'))
        shutil.copy(args.dict, os.path.join(model.save_path, 'dict.txt'))
        if args.unit == 'wp':
            shutil.copy(args.wp_model, os.path.join(model.save_path, 'wp.model'))

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
                            weight_decay=float(args.weight_decay),
                            transformer=args.lm_type == 'transformer')

        epoch, step = 1, 1
        ppl_dev_best = 10000

        # Set learning rate controller
        lr_controller = Controller(learning_rate=float(args.learning_rate),
                                   decay_type=args.decay_type,
                                   decay_start_epoch=args.decay_start_epoch,
                                   decay_rate=args.decay_rate,
                                   decay_patient_n_epochs=args.decay_patient_n_epochs,
                                   lower_better=True,
                                   best_value=ppl_dev_best,
                                   model_size=args.d_model,
                                   warmup_start_learning_rate=args.warmup_start_learning_rate,
                                   warmup_n_steps=args.warmup_n_steps,
                                   lr_init_factor=10,
                                   transformer=args.lm_type == 'transformer')

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
        if 'gated_conv' not in args.lm_type and args.lm_type != 'transformer':
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
                         np.exp(loss_train), np.exp(loss_dev),
                         lr_controller.lr, ys_train.shape[0], duration_step / 60))
            start_time_step = time.time()
        step += args.n_gpus
        pbar_epoch.update(ys_train.shape[0] * (ys_train.shape[1] - 1))

        # Save fugures of loss and accuracy
        if step % (args.print_step * 10) == 0:
            reporter.snapshot()
            if args.lm_type == 'transformer':
                model.module.plot_attention()

        # Save checkpoint and evaluate model per epoch
        if is_new_epoch:
            duration_epoch = time.time() - start_time_epoch
            logger.info('========== EPOCH:%d (%.2f min) ==========' % (epoch, duration_epoch / 60))

            if epoch < args.eval_start_epoch:
                # Save the model
                save_checkpoint(model.module, model.module.save_path, lr_controller,
                                epoch, step - 1, ppl_dev_best,
                                remove_old_checkpoints=True)
            else:
                start_time_eval = time.time()
                # dev
                ppl_dev, _ = eval_ppl([model.module], dev_set,
                                      batch_size=1, bptt=args.bptt)
                logger.info('PPL (%s): %.2f' % (dev_set.set, ppl_dev))

                # Update learning rate
                model.module.optimizer = lr_controller.decay(
                    model.module.optimizer,
                    epoch=epoch, value=ppl_dev)

                if ppl_dev < ppl_dev_best:
                    ppl_dev_best = ppl_dev
                    not_improved_epoch = 0
                    logger.info('||||| Best Score |||||')

                    # Save the model
                    save_checkpoint(model.module, model.module.save_path, lr_controller,
                                    epoch, step - 1, ppl_dev_best,
                                    remove_old_checkpoints=True)

                    # test
                    ppl_test_avg = 0.
                    for eval_set in eval_sets:
                        ppl_test, _ = eval_ppl([model.module], eval_set,
                                               batch_size=1, bptt=args.bptt)
                        logger.info('PPL (%s): %.2f' % (eval_set.set, ppl_test))
                        ppl_test_avg += ppl_test
                    if len(eval_sets) > 0:
                        logger.info('PPL (avg.): %.2f' % (ppl_test_avg / len(eval_sets)))
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


def make_model_name(args):
    dir_name = args.lm_type
    if args.lm_type == 'transformer':
        dir_name += str(args.d_model) + 'dmodel'
        dir_name += str(args.d_ff) + 'dff'
        dir_name += str(args.transformer_n_layers) + 'L'
    elif 'gated_conv' not in args.lm_type or args.lm_type == 'gated_conv_custom':
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
    if args.serialize:
        dir_name += '_serialize'
    if args.min_n_tokens > 0:
        dir_name += '_' + str(args.min_n_tokens) + 'tokens'
    if args.adaptive_softmax:
        dir_name += '_adaptiveSM'
    return dir_name


if __name__ == '__main__':
    # Setting for profiling
    pr = cProfile.Profile()
    save_path = pr.runcall(main)
    pr.dump_stats(os.path.join(save_path, 'train.profile'))
