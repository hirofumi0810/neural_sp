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
from neural_sp.bin.train_utils import load_config
from neural_sp.bin.train_utils import save_config
from neural_sp.bin.train_utils import set_logger
from neural_sp.bin.train_utils import set_save_path
from neural_sp.bin.train_utils import load_checkpoint
from neural_sp.bin.train_utils import save_checkpoint
from neural_sp.datasets.loader_lm import Dataset
from neural_sp.evaluators.ppl import eval_ppl
from neural_sp.models.data_parallel import CustomDataParallel
from neural_sp.models.lm.select import select_lm
from neural_sp.trainers.optimizer import set_optimizer
from neural_sp.trainers.reporter import Reporter
from neural_sp.trainers.lr_scheduler import LRScheduler
from neural_sp.trainers.model_name import set_lm_name
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
        dir_name = set_lm_name(args)
        save_path = mkdir_join(args.model_save_dir, '_'.join(
            os.path.basename(args.train_set).split('.')[:-1]), dir_name)
        save_path = set_save_path(save_path)  # avoid overwriting

    # Set logger
    logger = set_logger(os.path.join(save_path, 'train.log'), key='training', stdout=args.stdout)

    # Model setting
    model = select_lm(args, save_path)

    if args.resume:
        # Set optimizer
        epoch = int(args.resume.split('-')[-1])
        optimizer = set_optimizer(model, 'sgd' if epoch > conf['convert_to_sgd_epoch'] else conf['optimizer'],
                                  conf['lr'], conf['weight_decay'])

        # Restore the last saved model
        model, optimizer = load_checkpoint(model, args.resume, optimizer, resume=True)

        # Resume between convert_to_sgd_epoch -1 and convert_to_sgd_epoch
        if epoch == conf['convert_to_sgd_epoch']:
            optimizer = set_optimizer(model, 'sgd', args.lr, conf['weight_decay'])
            optimizer = LRScheduler(optimizer, args.lr,
                                    decay_type='always',
                                    decay_start_epoch=0,
                                    decay_rate=0.5)
            logger.info('========== Convert to SGD ==========')
    else:
        # Save the conf file as a yaml file
        save_config(vars(args), os.path.join(save_path, 'conf.yml'))

        # Save the nlsyms, dictionar, and wp_model
        if args.nlsyms:
            shutil.copy(args.nlsyms, os.path.join(save_path, 'nlsyms.txt'))
        shutil.copy(args.dict, os.path.join(save_path, 'dict.txt'))
        if args.unit == 'wp':
            shutil.copy(args.wp_model, os.path.join(save_path, 'wp.model'))

        for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
            logger.info('%s: %s' % (k, str(v)))

        # Count total parameters
        for n in sorted(list(model.num_params_dict.keys())):
            n_params = model.num_params_dict[n]
            logger.info("%s %d" % (n, n_params))
        logger.info("Total %.2f M parameters" % (model.total_parameters / 1000000))
        logger.info(model)

        # Set optimizer
        optimizer = set_optimizer(model, args.optimizer, args.lr, args.weight_decay)

        # Wrap optimizer by learning rate scheduler
        optimizer = LRScheduler(optimizer, args.lr,
                                decay_type=args.lr_decay_type,
                                decay_start_epoch=args.lr_decay_start_epoch,
                                decay_rate=args.lr_decay_rate,
                                decay_patient_n_epochs=args.lr_decay_patient_n_epochs,
                                warmup_start_lr=args.warmup_start_lr,
                                warmup_n_steps=args.warmup_n_steps,
                                model_size=args.d_model,
                                factor=args.lr_factor,
                                noam=args.lm_type == 'transformer')

    # GPU setting
    if args.n_gpus >= 1:
        model = CustomDataParallel(model,
                                   device_ids=list(range(0, args.n_gpus, 1)),
                                   deterministic=False,
                                   benchmark=True)
        model.cuda()

    # Set process name
    logger.info('PID: %s' % os.getpid())
    logger.info('USERNAME: %s' % os.uname()[1])
    setproctitle(args.job_name if args.job_name else dir_name)

    # Set reporter
    reporter = Reporter(save_path, tensorboard=True)

    hidden = None
    start_time_train = time.time()
    start_time_epoch = time.time()
    start_time_step = time.time()
    not_improved_n_epochs = 0
    pbar_epoch = tqdm(total=len(train_set))
    accum_n_tokens = 0
    while True:
        # Compute loss in the training set
        ys_train, is_new_epoch = train_set.next()
        accum_n_tokens += sum([len(y) for y in ys_train])
        optimizer.zero_grad()
        loss, hidden, reporter = model(ys_train, hidden, reporter)
        # loss /= args.accum_grad_n_steps
        if len(model.device_ids) > 1:
            loss.backward(torch.ones(len(model.device_ids)))
        else:
            loss.backward()
        loss.detach()  # Trancate the graph
        if args.accum_grad_n_tokens == 0 or accum_n_tokens >= args.accum_grad_n_tokens:
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            accum_n_tokens = 0
        loss_train = loss.item()
        del loss
        hidden = model.module.repackage_state(hidden)
        reporter.step()

        if optimizer._step % args.print_step == 0:
            # Compute loss in the dev set
            ys_dev = dev_set.next()[0]
            loss, _, reporter = model(ys_dev, None, reporter, is_eval=True)
            loss_dev = loss.item()
            del loss
            reporter.step(is_eval=True)

            duration_step = time.time() - start_time_step
            logger.info("step:%d(ep:%.2f) loss:%.3f(%.3f)/ppl:%.3f(%.3f)/lr:%.5f/bs:%d (%.2f min)" %
                        (optimizer._step, optimizer._epoch + train_set.epoch_detail,
                         loss_train, loss_dev,
                         np.exp(loss_train), np.exp(loss_dev),
                         optimizer.lr, ys_train.shape[0], duration_step / 60))
            start_time_step = time.time()
        pbar_epoch.update(ys_train.shape[0] * (ys_train.shape[1] - 1))

        # Save fugures of loss and accuracy
        if optimizer._step % (args.print_step * 10) == 0:
            reporter.snapshot()
            if args.lm_type == 'transformer':
                model.module.plot_attention()

        # Save checkpoint and evaluate model per epoch
        if is_new_epoch:
            duration_epoch = time.time() - start_time_epoch
            logger.info('========== EPOCH:%d (%.2f min) ==========' %
                        (optimizer._epoch + 1, duration_epoch / 60))

            if optimizer._epoch + 1 < args.eval_start_epoch:
                optimizer.epoch()
                reporter.epoch()

                # Save the model
                save_checkpoint(model, save_path, optimizer, optimizer._epoch,
                                remove_old_checkpoints=args.lm_type != 'transformer')
            else:
                start_time_eval = time.time()
                # dev
                ppl_dev, _ = eval_ppl([model.module], dev_set,
                                      batch_size=1, bptt=args.bptt)
                logger.info('PPL (%s): %.2f' % (dev_set.set, ppl_dev))
                optimizer.epoch(ppl_dev)
                reporter.epoch(ppl_dev, name='perplexity')

                if ppl_dev < optimizer.metric_best:
                    not_improved_n_epochs = 0
                    logger.info('||||| Best Score |||||')

                    # Save the model
                    save_checkpoint(model, save_path, optimizer, optimizer._epoch,
                                    remove_old_checkpoints=args.lm_type != 'transformer')

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
                    not_improved_n_epochs += 1

                duration_eval = time.time() - start_time_eval
                logger.info('Evaluation time: %.2f min' % (duration_eval / 60))

                # Early stopping
                if not_improved_n_epochs == args.stop_patient_n_epochs:
                    break

                # Convert to fine-tuning stage
                if optimizer._epoch == args.convert_to_sgd_epoch:
                    optimizer = set_optimizer(model, 'sgd', args.lr, args.weight_decay)
                    optimizer = LRScheduler(optimizer, args.lr,
                                            decay_type='always',
                                            decay_start_epoch=0,
                                            decay_rate=0.5)
                    logger.info('========== Convert to SGD ==========')

            pbar_epoch = tqdm(total=len(train_set))

            if optimizer._epoch == args.n_epochs:
                break

            start_time_step = time.time()
            start_time_epoch = time.time()

    duration_train = time.time() - start_time_train
    logger.info('Total time: %.2f hour' % (duration_train / 3600))

    if reporter.tensorboard:
        reporter.tf_writer.close()
    pbar_epoch.close()

    return save_path


if __name__ == '__main__':
    # Setting for profiling
    pr = cProfile.Profile()
    save_path = pr.runcall(main)
    pr.dump_stats(os.path.join(save_path, 'train.profile'))
