#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join, abspath
import sys
import time
from setproctitle import setproctitle
import argparse
from tensorboardX import SummaryWriter
import logging

import torch
torch.manual_seed(1623)
torch.cuda.manual_seed_all(1623)

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.timit.data.load_dataset import Dataset
from examples.timit.metrics.per import do_eval_per
from utils.training.learning_rate_controller import Controller
from utils.training.plot import plot_loss
from utils.training.training_loop import train_step
from utils.directory import mkdir_join, mkdir
from utils.io.variable import var2np
from utils.config import load_config, save_config

MAX_DECODE_LEN_PHONE = 40

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1,
                    help='the index of GPU (negative value indicates CPU)')
parser.add_argument('--config_path', type=str,
                    help='path to the configuration file')
parser.add_argument('--model_save_path', type=str,
                    help='path to save the model')


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(args.config_path)

    # Load model
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # Set process name
    setproctitle(
        'timit_' + params['model_type'] + '_' + params['label_type'])

    # Set save path
    save_path = mkdir_join(
        args.model_save_path, params['backend'], 'timit',
        params['model_type'], params['label_type'], model.name)
    model.set_save_path(save_path)

    # Save config file
    save_config(config_path=args.config_path, save_path=model.save_path)

    # Settig for logging
    logger = logging.getLogger('training')
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    fh = logging.FileHandler(join(model.save_path, 'train.log'))
    fh.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s line:%(lineno)d %(levelname)s:   %(message)s')
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)

    logger.info(params)

    logger.info('PID: %s' % os.getpid())
    logger.info('USERNAME: %s' % os.uname()[1])

    # Load dataset
    vocab_file_path_train = '../metrics/vocab_files/' + \
        params['label_type'] + '.txt'
    vocab_file_path_eval = '../metrics/vocab_files/phone39.txt'
    train_data = Dataset(
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='train', label_type=params['label_type'],
        vocab_file_path=vocab_file_path_train,
        batch_size=params['batch_size'],
        max_epoch=params['num_epoch'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, sort_stop_epoch=params['sort_stop_epoch'],
        save_format=params['save_format'], num_enque=None)
    dev_data = Dataset(
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='dev', label_type=params['label_type'],
        vocab_file_path=vocab_file_path_train,
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=True, save_format=params['save_format'])
    test_data = Dataset(
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='test', label_type='phone39',
        vocab_file_path=vocab_file_path_eval,
        batch_size=1, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        save_format=params['save_format'])

    # Count total parameters
    for name in sorted(list(model.num_params_dict.keys())):
        num_params = model.num_params_dict[name]
        logger.info("%s %d" % (name, num_params))
    logger.info("Total %.3f M parameters" % (model.total_parameters / 1000000))

    # Define optimizer
    model.set_optimizer(
        optimizer=params['optimizer'],
        learning_rate_init=float(params['learning_rate']),
        weight_decay=float(params['weight_decay']),
        clip_grad_norm=params['clip_grad_norm'],
        lr_schedule=False,
        factor=params['decay_rate'],
        patience_epoch=params['decay_patient_epoch'])

    # Define learning rate controller
    lr_controller = Controller(
        learning_rate_init=params['learning_rate'],
        backend=params['backend'],
        decay_start_epoch=params['decay_start_epoch'],
        decay_rate=params['decay_rate'],
        decay_patient_epoch=params['decay_patient_epoch'],
        lower_better=True)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    # Setting for tensorboard
    if params['backend'] == 'pytorch':
        tf_writer = SummaryWriter(model.save_path)

    # Train model
    csv_steps, csv_loss_train, csv_loss_dev = [], [], []
    start_time_train = time.time()
    start_time_epoch = time.time()
    start_time_step = time.time()
    per_dev_best = 1
    not_improved_epoch = 0
    learning_rate = float(params['learning_rate'])
    loss_train_mean = 0.
    for step, (batch_train, is_new_epoch) in enumerate(train_data):

        # Compute loss in the training set (including parameter update)
        model, loss_train_val = train_step(
            model, batch_train, params['clip_grad_norm'], backend=params['backend'])
        loss_train_mean += loss_train_val

        # Inject Gaussian noise to all parameters
        if float(params['weight_noise_std']) > 0 and learning_rate < float(params['learning_rate']):
            model.weight_noise_injection = True

        if (step + 1) % params['print_step'] == 0:

            # Compute loss in the dev set
            batch_dev = dev_data.next()[0]
            loss_dev = model(
                batch_dev['xs'], batch_dev['ys'], batch_dev['x_lens'], batch_dev['y_lens'], is_eval=True)

            loss_train_mean /= params['print_step']
            csv_steps.append(step)
            csv_loss_train.append(loss_train_mean)
            csv_loss_dev.append(loss_dev)

            # Logging by tensorboard
            if params['backend'] == 'pytorch':
                tf_writer.add_scalar('train/loss', loss_train_mean, step + 1)
                tf_writer.add_scalar('dev/loss', loss_dev, step + 1)
                for name, param in model.named_parameters():
                    name = name.replace('.', '/')
                    tf_writer.add_histogram(name, var2np(param), step + 1)
                    tf_writer.add_histogram(
                        name + '/grad', var2np(param.grad), step + 1)

            duration_step = time.time() - start_time_step
            logger.info("...Step:%d (epoch:%.3f): loss:%.3f (%.3f) / lr:%.5f / batch:%d / x_lens:%d (%.3f min)" %
                        (step + 1, train_data.epoch_detail,
                         loss_train_mean, loss_dev,
                         learning_rate, train_data.current_batch_size,
                         max(batch_train['x_lens']), duration_step / 60))
            start_time_step = time.time()
            loss_train_mean = 0.

        # Save checkpoint and evaluate model per epoch
        if is_new_epoch:
            duration_epoch = time.time() - start_time_epoch
            logger.info('===== EPOCH:%d (%.3f min) =====' %
                        (train_data.epoch, duration_epoch / 60))

            # Save fugure of loss
            plot_loss(csv_loss_train, csv_loss_dev, csv_steps,
                      save_path=model.save_path)

            if train_data.epoch < params['eval_start_epoch']:
                # Save the model
                model.save_checkpoint(model.save_path, epoch=train_data.epoch)
            else:
                start_time_eval = time.time()
                # dev
                per_dev_epoch = do_eval_per(
                    model=model,
                    model_type=params['model_type'],
                    dataset=dev_data,
                    label_type=params['label_type'],
                    beam_width=1,
                    max_decode_len=MAX_DECODE_LEN_PHONE,
                    eval_batch_size=1)
                logger.info('  PER (dev): %f %%' % (per_dev_epoch * 100))

                if per_dev_epoch < per_dev_best:
                    per_dev_best = per_dev_epoch
                    not_improved_epoch = 0
                    logger.info('■■■ ↑Best Score (PER)↑ ■■■')

                    # Save the model
                    model.save_checkpoint(
                        model.save_path, epoch=train_data.epoch)

                    # test
                    per_test = do_eval_per(
                        model=model,
                        model_type=params['model_type'],
                        dataset=test_data,
                        label_type=params['label_type'],
                        beam_width=1,
                        max_decode_len=MAX_DECODE_LEN_PHONE,
                        eval_batch_size=1)
                    logger.info('  PER (test): %f %%' % (per_test * 100))
                else:
                    not_improved_epoch += 1

                duration_eval = time.time() - start_time_eval
                logger.info('Evaluation time: %.3f min' % (duration_eval / 60))

                # Early stopping
                if not_improved_epoch == params['not_improved_patient_epoch']:
                    break

                # Update learning rate
                model.optimizer, learning_rate = lr_controller.decay_lr(
                    optimizer=model.optimizer,
                    learning_rate=learning_rate,
                    epoch=train_data.epoch,
                    value=per_dev_epoch)

                if train_data.epoch == params['convert_to_sgd_epoch']:
                    # Convert to fine-tuning stage
                    model.set_optimizer(
                        'momentum',
                        learning_rate_init=learning_rate,
                        weight_decay=float(params['weight_decay']),
                        clip_grad_norm=params['clip_grad_norm'],
                        lr_schedule=False,
                        factor=params['decay_rate'],
                        patience_epoch=params['decay_patient_epoch'])

            start_time_step = time.time()
            start_time_epoch = time.time()

    duration_train = time.time() - start_time_train
    logger.info('Total time: %.3f hour' % (duration_train / 3600))

    # Training was finished correctly
    with open(join(model.save_path, 'complete.txt'), 'w') as f:
        f.write('')

    if params['backend'] == 'pytorch':
        tf_writer.close()


if __name__ == '__main__':
    main()
