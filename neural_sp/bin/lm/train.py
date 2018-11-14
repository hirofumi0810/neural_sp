#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Train the RNNLM."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cProfile
import math
import numpy as np
import os
# from setproctitle import setproctitle
import shutil
from tensorboardX import SummaryWriter
import time
import torch
from tqdm import tqdm

from neural_sp.bin.asr.train_utils import Controller
from neural_sp.bin.asr.train_utils import Reporter
from neural_sp.bin.lm.train_utils import Updater
from neural_sp.datasets.loader_lm import Dataset
from neural_sp.evaluators.ppl import eval_ppl
from neural_sp.models.data_parallel import CustomDataParallel
from neural_sp.models.rnnlm.rnnlm_seq import SeqRNNLM
from neural_sp.utils.config import load_config
from neural_sp.utils.config import save_config
from neural_sp.utils.general import mkdir_join
from neural_sp.utils.general import set_logger

parser = argparse.ArgumentParser()
# general
parser.add_argument('--ngpus', type=int, default=0,
                    help='number of GPUs (0 indicates CPU)')
parser.add_argument('--config', type=str, default=None,
                    help='path to a yaml file for configuration')
parser.add_argument('--model', type=str, default=None,
                    help='directory to save a model')
parser.add_argument('--resume_model', type=str, default=None,
                    help='path to the model to resume training')
parser.add_argument('--job_name', type=str, default='',
                    help='name of job')
# dataset
parser.add_argument('--train_set', type=str,
                    help='path to a csv file for the training set')
parser.add_argument('--dev_set', type=str,
                    help='path to a csv file for the development set')
parser.add_argument('--eval_sets', type=str, default=[], nargs='+',
                    help='path to csv files for the evaluation sets')
parser.add_argument('--dict', type=str,
                    help='path to a dictionary file')
parser.add_argument('--label_type', type=str, default='word',
                    choices=['word', 'wp', 'char'],
                    help='')
parser.add_argument('--wp_model', type=str, default=False, nargs='?',
                    help='path to of the wordpiece model')
parser.add_argument('--eos', type=int, default=2,
                    help='index of <eos>')
# features
parser.add_argument('--dynamic_batching', type=bool, default=False,
                    help='')
# topology (encoder)
parser.add_argument('--rnn_type', type=str, default='lstm',
                    choices=['blstm', 'bgru'],
                    help='')
parser.add_argument('--num_units', type=int, default=320,
                    help='')
parser.add_argument('--num_projs', type=int, default=0,
                    help='')
parser.add_argument('--num_layers', type=int, default=5,
                    help='')
parser.add_argument('--emb_dim', type=int, default=5,
                    help='')
parser.add_argument('--tie_weights', type=bool, default=False,
                    help='')
parser.add_argument('--residual', type=bool, default=False,
                    help='')
parser.add_argument('--backward', type=bool, default=False,
                    help='')
# optimization
parser.add_argument('--batch_size', type=int, default=256,
                    help='')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='')
parser.add_argument('--num_epochs', type=int, default=50,
                    help='')
parser.add_argument('--convert_to_sgd_epoch', type=int, default=20,
                    help='')
parser.add_argument('--print_step', type=int, default=100,
                    help='')
# annealing
parser.add_argument('--decay_type', type=str, default='per_epoch',
                    choices=['per_epoch'],
                    help='')
parser.add_argument('--decay_start_epoch', type=int, default=10,
                    help='')
parser.add_argument('--decay_rate', type=float, default=0.9,
                    help='')
parser.add_argument('--decay_patient_epoch', type=int, default=0,
                    help='')
parser.add_argument('--sort_stop_epoch', type=int, default=100,
                    help='')
parser.add_argument('--not_improved_patient_epoch', type=int, default=5,
                    help='')
parser.add_argument('--eval_start_epoch', type=int, default=1,
                    help='')
# initialization
parser.add_argument('--param_init', type=float, default=0.1,
                    help='')
parser.add_argument('--param_init_dist', type=str, default='uniform',
                    help='')
parser.add_argument('--rec_weight_orthogonal', type=bool, default=False,
                    help='')
# regularization
parser.add_argument('--clip_grad_norm', type=float, default=5.0,
                    help='')
parser.add_argument('--dropout_hidden', type=float, default=0.0,
                    help='')
parser.add_argument('--dropout_out', type=float, default=0.0,
                    help='')
parser.add_argument('--dropout_emb', type=float, default=0.0,
                    help='')
parser.add_argument('--weight_decay', type=float, default=1e-6,
                    help='')
parser.add_argument('--logits_temp', type=float, default=1.0,
                    help='')
args = parser.parse_args()


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


def main():

    # Load a config file
    if args.resume_model is None:
        config = load_config(args.config)
    else:
        # Restart from the last checkpoint
        config = load_config(os.path.join(args.resume_model, 'config.yml'))

    # Merge config with args
    for k, v in config.items():
        setattr(args, k, v)

    # Load dataset
    train_set = Dataset(csv_path=args.train_set,
                        dict_path=args.dict,
                        label_type=args.label_type,
                        batch_size=args.batch_size * args.ngpus,
                        bptt=args.bptt,
                        eos=args.eos,
                        max_epoch=args.num_epochs,
                        shuffle=True)
    dev_set = Dataset(csv_path=args.dev_set,
                      dict_path=args.dict,
                      label_type=args.label_type,
                      batch_size=args.batch_size * args.ngpus,
                      bptt=args.bptt,
                      eos=args.eos,
                      shuffle=True)
    eval_sets = []
    for set in args.eval_sets:
        eval_sets += [Dataset(csv_path=set,
                              dict_path=args.dict,
                              label_type=args.label_type,
                              batch_size=1,
                              bptt=args.bptt,
                              eos=args.eos,
                              is_test=True)]

    args.num_classes = train_set.num_classes

    # Model setting
    model = SeqRNNLM(args)
    model.name = args.rnn_type
    model.name += str(args.num_units) + 'H'
    model.name += str(args.num_projs) + 'P'
    model.name += str(args.num_layers) + 'L'
    model.name += '_emb' + str(args.emb_dim)
    model.name += '_' + args.optimizer
    model.name += '_lr' + str(args.learning_rate)
    model.name += '_bs' + str(args.batch_size)
    if args.tie_weights:
        model.name += '_tie'
    if args.residual:
        model.name += '_residual'
    if args.backward:
        model.name += '_bwd'

    if args.resume_model is None:
        # Set save path
        save_path = mkdir_join(args.model, '_'.join(os.path.basename(args.train_set).split('.')[:-1]), model.name)
        model.set_save_path(save_path)  # avoid overwriting

        # Save the config file as a yaml file
        save_config(vars(args), model.save_path)

        # Save the dictionary & wp_model
        shutil.copy(args.dict, os.path.join(model.save_path, 'dict.txt'))
        if args.label_type == 'wp':
            shutil.copy(args.wp_model, os.path.join(model.save_path, 'wp.model'))

        # Setting for logging
        logger = set_logger(os.path.join(model.save_path, 'train.log'), key='training')

        for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
            logger.info('%s: %s' % (k, str(v)))

        # Count total parameters
        for name in sorted(list(model.num_params_dict.keys())):
            num_params = model.num_params_dict[name]
            logger.info("%s %d" % (name, num_params))
        logger.info("Total %.2f M parameters" % (model.total_parameters / 1000000))
        logger.info(model)

        # Set optimizer
        model.set_optimizer(optimizer=args.optimizer,
                            learning_rate_init=float(args.learning_rate),
                            weight_decay=float(args.weight_decay),
                            clip_grad_norm=args.clip_grad_norm,
                            lr_schedule=False,
                            factor=args.decay_rate,
                            patience_epoch=args.decay_patient_epoch)

        epoch, step = 1, 0
        learning_rate = float(args.learning_rate)
        metric_dev_best = 10000

    else:
        raise NotImplementedError()

    train_set.epoch = epoch - 1

    # GPU setting
    if args.ngpus >= 1:
        model = CustomDataParallel(model,
                                   device_ids=list(range(0, args.ngpus, 1)),
                                   deterministic=False,
                                   benchmark=True)
        model.cuda()

    logger.info('PID: %s' % os.getpid())
    logger.info('USERNAME: %s' % os.uname()[1])

    # Set process name
    # setproctitle(args.job_name)

    # Set learning rate controller
    lr_controller = Controller(learning_rate_init=learning_rate,
                               decay_type=args.decay_type,
                               decay_start_epoch=args.decay_start_epoch,
                               decay_rate=args.decay_rate,
                               decay_patient_epoch=args.decay_patient_epoch,
                               lower_better=True,
                               best_value=metric_dev_best)

    # Set reporter
    reporter = Reporter(model.module.save_path, max_loss=10)

    # Set the updater
    updater = Updater(args.clip_grad_norm)

    # Setting for tensorboard
    tf_writer = SummaryWriter(model.module.save_path)

    start_time_train = time.time()
    start_time_epoch = time.time()
    start_time_step = time.time()
    not_improved_epoch = 0
    loss_train_mean, acc_train_mean = 0, 0
    pbar_epoch = tqdm(total=len(train_set))
    pbar_all = tqdm(total=len(train_set) * args.num_epochs)
    while True:
        # Compute loss in the training set (including parameter update)
        ys_train, is_new_epoch = train_set.next()
        model, loss_train, acc_train = updater(model, ys_train)
        loss_train_mean += loss_train
        acc_train_mean += acc_train
        pbar_epoch.update(np.sum([len(y) for y in ys_train]))

        if (step + 1) % args.print_step == 0:
            # Compute loss in the dev set
            ys_dev = dev_set.next()[0]
            model, loss_dev, acc_dev = updater(model, ys_dev, is_eval=True)

            loss_train_mean /= args.print_step
            acc_train_mean /= args.print_step
            reporter.step(step, loss_train_mean, loss_dev, acc_train_mean, acc_dev)

            # Logging by tensorboard
            tf_writer.add_scalar('train/loss', loss_train_mean, step + 1)
            tf_writer.add_scalar('dev/loss', loss_dev, step + 1)
            for n, p in model.module.named_parameters():
                n = n.replace('.', '/')
                if p.grad is not None:
                    tf_writer.add_histogram(n, p.data.cpu().numpy(), step + 1)
                    tf_writer.add_histogram(n + '/grad', p.grad.data.cpu().numpy(), step + 1)

            duration_step = time.time() - start_time_step
            logger.info("...Step:%d(ep:%.2f) loss:%.2f(%.2f)/acc:%.2f(%.2f)/ppl:%.2f(%.2f)/lr:%.5f/bs:%d (%.2f min)" %
                        (step + 1, train_set.epoch_detail,
                         loss_train_mean, loss_dev, acc_train_mean, acc_dev,
                         math.exp(loss_train_mean), math.exp(loss_dev),
                         learning_rate, len(ys_train), duration_step / 60))
            start_time_step = time.time()
            loss_train_mean, acc_train_mean = 0, 0
        step += args.ngpus

        # Save checkpoint and evaluate model per epoch
        if is_new_epoch:
            duration_epoch = time.time() - start_time_epoch
            logger.info('===== EPOCH:%d (%.2f min) =====' % (epoch, duration_epoch / 60))

            # Save fugures of loss and accuracy
            reporter.epoch()

            if epoch < args.eval_start_epoch:
                # Save the model
                model.module.save_checkpoint(model.module.save_path, epoch, step,
                                             learning_rate, metric_dev_best)
            else:
                start_time_eval = time.time()
                # dev
                ppl_dev = eval_ppl([model.module], dev_set, args.bptt)
                logger.info(' PPL (%s): %.3f' % (dev_set.set, ppl_dev))

                # Update learning rate
                model.module.optimizer, learning_rate = lr_controller.decay_lr(
                    optimizer=model.module.optimizer,
                    learning_rate=learning_rate,
                    epoch=epoch,
                    value=ppl_dev)

                if ppl_dev < metric_dev_best:
                    metric_dev_best = ppl_dev
                    not_improved_epoch = 0
                    logger.info('||||| Best Score |||||')

                    # Save the model
                    model.module.save_checkpoint(model.module.save_path, epoch, step,
                                                 learning_rate, metric_dev_best)

                    # test
                    ppl_test_mean = 0.
                    for eval_set in eval_sets:
                        ppl_test = eval_ppl([model.module], eval_set, args.bptt)
                        logger.info(' PPL (%s): %.3f' % (eval_set.set, ppl_test))
                        ppl_test_mean += ppl_test
                    if len(eval_sets) > 0:
                        logger.info(' PPL (mean): %.3f' % (ppl_test_mean / len(eval_sets)))
                else:
                    not_improved_epoch += 1

                duration_eval = time.time() - start_time_eval
                logger.info('Evaluation time: %.2f min' % (duration_eval / 60))

                # Early stopping
                if not_improved_epoch == args.not_improved_patient_epoch:
                    break

                if epoch == args.convert_to_sgd_epoch:
                    # Convert to fine-tuning stage
                    model.module.set_optimizer(
                        'sgd',
                        learning_rate_init=float(args.learning_rate),  # TODO: ?
                        weight_decay=float(args.weight_decay),
                        clip_grad_norm=args.clip_grad_norm,
                        lr_schedule=False,
                        factor=args.decay_rate,
                        patience_epoch=args.decay_patient_epoch)
                    logger.info('========== Convert to SGD ==========')

            pbar_epoch = tqdm(total=len(train_set))
            pbar_all.update(len(train_set))

            if epoch == args.num_epochs:
                break

            start_time_step = time.time()
            start_time_epoch = time.time()
            epoch += 1

    duration_train = time.time() - start_time_train
    logger.info('Total time: %.2f hour' % (duration_train / 3600))

    tf_writer.close()
    pbar_epoch.close()
    pbar_all.close()

    return model.module.save_path


if __name__ == '__main__':
    # Setting for profiling
    pr = cProfile.Profile()
    save_path = pr.runcall(main)
    pr.dump_stats(os.path.join(save_path, 'train.profile'))
