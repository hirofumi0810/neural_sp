#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the model (Switchboard corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import time
from setproctitle import setproctitle
import yaml
import shutil
import argparse
from tensorboardX import SummaryWriter
import logging

import torch
torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.swbd.data.load_dataset_hierarchical import Dataset
from examples.swbd.metrics.cer import do_eval_cer
from examples.swbd.metrics.wer import do_eval_wer
from utils.training.learning_rate_controller import Controller
from utils.training.plot import plot_loss
from utils.training.training_loop import train_hierarchical_step
from utils.directory import mkdir_join, mkdir
from utils.io.variable import var2np

MAX_DECODE_LEN_WORD = 100
MAX_DECODE_LEN_CHAR = 300

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
    with open(args.config_path, "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Get voabulary number (excluding blank, <SOS>, <EOS> classes)
    with open('../metrics/vocab_num.yml', "r") as f:
        vocab_num = yaml.load(f)
        params['num_classes'] = vocab_num[params['data_size']
                                          ][params['label_type']]
        params['num_classes_sub'] = vocab_num[params['data_size']
                                              ][params['label_type_sub']]

    # Model setting
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # Set process name
    setproctitle('swbd_' + params['model_type'] + '_' +
                 params['label_type'] + '_' + params['label_type_sub'] + '_' + params['data_size'])

    # Set save path
    save_path = mkdir_join(
        args.model_save_path,  params['backend'], 'swbd',
        params['model_type'],
        params['label_type'] + '_' + params['label_type_sub'],
        params['data_size'], model.name)
    model.set_save_path(save_path)

    # Save config file
    shutil.copyfile(args.config_path, join(model.save_path, 'config.yml'))

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

    # Load dataset
    vocab_file_path = '../metrics/vocab_files/' + \
        params['label_type'] + '_' + params['data_size'] + '.txt'
    vocab_file_path_sub = '../metrics/vocab_files/' + \
        params['label_type_sub'] + '_' + params['data_size'] + '.txt'
    train_data = Dataset(
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='train', data_size=params['data_size'],
        label_type=params['label_type'],
        label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
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
        data_type='dev', data_size=params['data_size'],
        label_type=params['label_type'],
        label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=True, save_format=params['save_format'])
    eval2000_swbd_data = Dataset(
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='eval2000_swbd', data_size=params['data_size'],
        label_type=params['label_type'],
        label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        save_format=params['save_format'])
    eval2000_ch_data = Dataset(
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='eval2000_ch', data_size=params['data_size'],
        label_type=params['label_type'],
        label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        save_format=params['save_format'])

    # Count total parameters
    for name in sorted(list(model.num_params_dict.keys())):
        num_params = model.num_params_dict[name]
        logger.info("%s %d" % (name, num_params))
    logger.info("Total %.3f M parameters" % (model.total_parameters / 1000000))

    # Define optimizer
    optimizer, _ = model.set_optimizer(
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
        decay_start_epoch=params['decay_start_epoch'],
        decay_rate=params['decay_rate'],
        decay_patient_epoch=params['decay_patient_epoch'],
        lower_better=True)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    # Setting for tensorboard
    tf_writer = SummaryWriter(model.save_path)

    # Train model
    csv_steps, csv_loss_train, csv_loss_dev = [], [], []
    start_time_train = time.time()
    start_time_epoch = time.time()
    start_time_step = time.time()
    ler_dev_best = 1
    not_improved_epoch = 0
    learning_rate = float(params['learning_rate'])
    loss_train_vaL_mean, loss_main_train_val_mean, loss_sub_train_val_mean = 0., 0., 0.
    for step, (batch, is_new_epoch) in enumerate(train_data):

        # Compute loss in the training set (including parameter update)
        batch_size_step = train_data._batch_size
        model, loss_train_val, loss_main_train_val, loss_sub_train_val, div_num = train_hierarchical_step(
            model, batch, params['clip_grad_norm'], backend=params['backend'])
        loss_train_vaL_mean += loss_train_val
        loss_main_train_val_mean += loss_main_train_val
        loss_sub_train_val_mean += loss_sub_train_val

        # on-the-fly setting
        train_data._batch_size = train_data.batch_size // div_num

        # Inject Gaussian noise to all parameters
        if float(params['weight_noise_std']) > 0 and learning_rate < float(params['learning_rate']):
            model.weight_noise_injection = True

        if (step + 1) % params['print_step'] == 0:

            # Compute loss in the dev set
            inputs, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub, _ = dev_data.next()[
                0]
            loss_dev_val, loss_main_dev_val, loss_sub_dev_val = model(
                inputs, labels, labels_sub, inputs_seq_len,
                labels_seq_len, labels_seq_len_sub, is_eval=True)

            loss_train_vaL_mean /= params['print_step']
            loss_main_train_val_mean /= params['print_step']
            loss_sub_train_val_mean /= params['print_step']
            csv_steps.append(step)
            csv_loss_train.append(loss_train_vaL_mean)
            csv_loss_dev.append(loss_dev_val)

            # Logging by tensorboard
            tf_writer.add_scalar('train/loss', loss_train_vaL_mean, step + 1)
            tf_writer.add_scalar(
                'train/loss_main', loss_main_train_val_mean, step + 1)
            tf_writer.add_scalar(
                'train/loss_sub', loss_sub_train_val_mean, step + 1)
            tf_writer.add_scalar('dev/loss', loss_dev_val, step + 1)
            tf_writer.add_scalar('dev/loss_main', loss_main_dev_val, step + 1)
            tf_writer.add_scalar('dev/loss_sub', loss_sub_dev_val, step + 1)
            if params['backend'] == 'pytorch':
                for name, param in model.named_parameters():
                    name = name.replace('.', '/')
                    tf_writer.add_histogram(
                        name, var2np(param.clone()), step + 1)
                    tf_writer.add_histogram(
                        name + '/grad', var2np(param.grad.clone()), step + 1)
            elif params['backend'] == 'chainer':
                for name, param in model.named_parameters():
                    name = name[1:]
                    # tf_writer.add_histogram(
                    #     name, var2np(param.clone()), step + 1)
                    # tf_writer.add_histogram(
                    #     name + '/grad', var2np(param.grad.clone()), step + 1)

            duration_step = time.time() - start_time_step
            logger.info("...Step:%d (epoch:%.3f): loss:%.3f/%.3f (%.3f/%.3f) / lr:%.5f / batch:%d (%.3f min)" %
                        (step + 1, train_data.epoch_detail,
                         loss_main_train_val_mean, loss_sub_train_val_mean,
                         loss_main_dev_val, loss_sub_dev_val,
                         learning_rate, batch_size_step, duration_step / 60))
            start_time_step = time.time()
            loss_train_vaL_mean, loss_main_train_val_mean, loss_sub_train_val_mean = 0., 0., 0.

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
                wer_dev_epoch = do_eval_wer(
                    model=model,
                    model_type=params['model_type'],
                    dataset=dev_data,
                    label_type=params['label_type'],
                    beam_width=1,
                    max_decode_len=MAX_DECODE_LEN_WORD,
                    eval_batch_size=1)
                logger.info('  WER (dev): %f %%' % (wer_dev_epoch * 100))

                if wer_dev_epoch < ler_dev_best:
                    ler_dev_best = wer_dev_epoch
                    not_improved_epoch = 0
                    logger.info('■■■ ↑Best Score (WER)↑ ■■■')

                    # Save the model
                    model.save_checkpoint(
                        model.save_path, epoch=train_data.epoch)

                    # test
                    wer_test_swbd = do_eval_wer(
                        model=model,
                        model_type=params['model_type'],
                        dataset=eval2000_swbd_data,
                        label_type=params['label_type'],
                        beam_width=1,
                        max_decode_len=MAX_DECODE_LEN_WORD,
                        eval_batch_size=1)
                    logger.info('  WER (SWB, main): %f %%' %
                                (wer_test_swbd * 100))
                    cer_eval2000_swbd, _ = do_eval_cer(
                        model=model,
                        model_type=params['model_type'],
                        dataset=eval2000_swbd_data,
                        label_type=params['label_type_sub'],
                        beam_width=1,
                        max_decode_len=MAX_DECODE_LEN_CHAR,
                        eval_batch_size=1)
                    logger.info('  CER (SWB, sub): %f %%' %
                                (cer_eval2000_swbd * 100))
                    wer_eval2000_ch = do_eval_wer(
                        model=model,
                        model_type=params['model_type'],
                        dataset=eval2000_ch_data,
                        label_type=params['label_type'],
                        beam_width=1,
                        max_decode_len=MAX_DECODE_LEN_WORD,
                        eval_batch_size=1)
                    logger.info('  WER (CHE, main): %f %%' %
                                (wer_eval2000_ch * 100))
                    cer_eval2000_ch, _ = do_eval_cer(
                        model=model,
                        model_type=params['model_type'],
                        dataset=eval2000_ch_data,
                        label_type=params['label_type_sub'],
                        beam_width=1,
                        max_decode_len=MAX_DECODE_LEN_CHAR,
                        eval_batch_size=1)
                    logger.info('  CER (CHE, sub): %f %%' %
                                (cer_eval2000_ch * 100))
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
                    value=wer_dev_epoch)

            start_time_step = time.time()
            start_time_epoch = time.time()

    duration_train = time.time() - start_time_train
    logger.info('Total time: %.3f hour' % (duration_train / 3600))

    # Training was finished correctly
    with open(join(model.save_path, 'complete.txt'), 'w') as f:
        f.write('')

    tf_writer.close()


if __name__ == '__main__':
    main()
