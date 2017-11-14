#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the CTC model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile, abspath
import sys
import time
from setproctitle import setproctitle
import yaml
import shutil

import torch.nn as nn

sys.path.append(abspath('../../../'))
from examples.librispeech.data.load_dataset_ctc import Dataset
from examples.librispeech.metrics.cer import do_eval_cer
# from examples.librispeech.metrics.wer import do_eval_wer
from utils.training.learning_rate_controller import Controller
from utils.training.plot import plot_loss
from utils.directory import mkdir_join, mkdir
from utils.io.variable import np2var, var2np
from models.pytorch.ctc.ctc import CTC


def do_train(model, params):
    """Run training.
    Args:
        model: the model to train
        params (dict): A dictionary of parameters
    """
    # Load dataset
    train_data = Dataset(
        data_type='train', label_type=params['label_type'],
        batch_size=params['batch_size'],
        max_epoch=params['num_epoch'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, sort_stop_epoch=params['sort_stop_epoch'])
    dev_clean_data = Dataset(
        data_type='dev_clean', data_size=params['data_size'],
        label_type=params['label_type'],
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, num_gpus=1)
    dev_other_data = Dataset(
        data_type='dev_other', data_size=params['data_size'],
        label_type=params['label_type'],
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, num_gpus=1)
    test_clean_data = Dataset(
        data_type='test_clean', data_size=params['data_size'],
        label_type=params['label_type'],
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False)
    test_other_data = Dataset(
        data_type='test_other', data_size=params['data_size'],
        label_type=params['label_type'],
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False)

    # Count total parameters
    for name, num_params in model.num_params_dict.items():
        print("%s %d" % (name, num_params))
    print("Total %.3f M parameters" % (model.total_parameters / 1000000))

    # Define optimizer
    optimizer, _ = model.set_optimizer(
        params['optimizer'],
        learning_rate_init=float(params['learning_rate']),
        weight_decay=params['weight_decay'],
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

    # Initialize parameters
    model.init_weights()

    # GPU setting
    use_cuda = model.use_cuda
    model.set_cuda(deterministic=False)

    # Train model
    csv_steps, csv_loss_train, csv_loss_dev = [], [], []
    start_time_train = time.time()
    start_time_epoch = time.time()
    start_time_step = time.time()
    ler_dev_best = 1
    not_improved_epoch = 0
    learning_rate = float(params['learning_rate'])
    for step, (data, is_new_epoch) in enumerate(train_data):

        # Create feed dictionary for next mini batch (train)
        inputs, labels, inputs_seq_len, labels_seq_len, _ = data
        inputs = np2var(inputs, use_cuda=use_cuda)
        labels = np2var(labels, use_cuda=use_cuda, dtype='int')
        inputs_seq_len = np2var(
            inputs_seq_len, use_cuda=use_cuda, dtype='int')
        labels_seq_len = np2var(labels_seq_len, use_cuda=use_cuda, dtype='int')

        # Clear gradients before
        optimizer.zero_grad()

        # Make prediction
        logits, perm_indices = model(inputs[0], inputs_seq_len[0])

        # Compute loss in the training set
        loss_train = model.compute_loss(
            logits,
            labels[0][perm_indices],
            inputs_seq_len[0][perm_indices],
            labels_seq_len[0][perm_indices])

        # Compute gradient
        optimizer.zero_grad()
        loss_train.backward()

        # Clip gradient norm
        nn.utils.clip_grad_norm(model.parameters(), params['clip_grad_norm'])

        # Update parameters
        optimizer.step()
        # TODO: Add scheduler

        if (step + 1) % params['print_step'] == 0:

            # Create feed dictionary for next mini batch (dev)
            if params['data_size'] in ['100h', '460h']:
                inputs, labels, inputs_seq_len, labels_seq_len, _ = dev_clean_data.next()[
                    0]
            else:
                inputs, labels, inputs_seq_len, labels_seq_len, _ = dev_other_data.next()[
                    0]
            inputs = np2var(inputs, use_cuda=use_cuda, volatile=True)
            labels = np2var(
                labels, use_cuda=use_cuda, volatile=True, dtype='int')
            inputs_seq_len = np2var(
                inputs_seq_len, use_cuda=use_cuda, volatile=True, dtype='int')
            labels_seq_len = np2var(
                labels_seq_len, use_cuda=use_cuda, volatile=True, dtype='int')

            # ***Change to evaluation mode***
            model.eval()

            # Make prediction
            logits, perm_indices = model(inputs[0], inputs_seq_len[0])

            # Compute loss in the dev set
            loss_dev = model.compute_loss(
                logits,
                labels[0][perm_indices],
                inputs_seq_len[0][perm_indices],
                labels_seq_len[0][perm_indices])
            csv_steps.append(step)
            csv_loss_train.append(var2np(loss_train))
            csv_loss_dev.append(var2np(loss_dev))

            # ***Change to training mode***
            model.train()

            duration_step = time.time() - start_time_step
            print("Step %d (epoch: %.3f): loss = %.3f (%.3f) / lr = %.5f (%.3f min)" %
                  (step + 1, train_data.epoch_detail,
                   var2np(loss_train), var2np(loss_dev),
                   learning_rate, duration_step / 60))
            # sys.stdout.flush()
            start_time_step = time.time()

        # Save checkpoint and evaluate model per epoch
        if is_new_epoch:
            duration_epoch = time.time() - start_time_epoch
            print('-----EPOCH:%d (%.3f min)-----' %
                  (train_data.epoch, duration_epoch / 60))

            # Save fugure of loss
            plot_loss(csv_loss_train, csv_loss_dev, csv_steps,
                      save_path=model.save_path)

            if train_data.epoch >= params['eval_start_epoch']:
                # ***Change to evaluation mode***
                model.eval()

                start_time_eval = time.time()
                print('=== Dev Data Evaluation ===')
                if 'char' in params['label_type']:
                    # dev-clean
                    cer_dev_clean_epoch, wer_dev_clean_epoch = do_eval_cer(
                        model=model,
                        model_type='ctc',
                        dataset=dev_clean_data,
                        label_type=params['label_type'],
                        beam_width=1,
                        eval_batch_size=1)
                    print('  CER (clean): %f %%' % (cer_dev_clean_epoch * 100))
                    print('  WER (clean): %f %%' % (wer_dev_clean_epoch * 100))

                    # dev-other
                    cer_dev_other_epoch, wer_dev_other_epoch = do_eval_cer(
                        model=model,
                        model_type='ctc',
                        dataset=dev_other_data,
                        label_type=params['label_type'],
                        beam_width=1,
                        eval_batch_size=1)
                    print('  CER (other): %f %%' % (cer_dev_other_epoch * 100))
                    print('  WER (other): %f %%' % (wer_dev_other_epoch * 100))

                    if params['data_size'] in ['100h', '460h']:
                        metric_epoch = cer_dev_clean_epoch
                    else:
                        metric_epoch = cer_dev_other_epoch

                    if metric_epoch < ler_dev_best:
                        ler_dev_best = metric_epoch
                        not_improved_epoch = 0
                        print('■■■ ↑Best Score (CER)↑ ■■■')

                        # # Save the model
                        saved_path = model.save_checkpoint(
                            model.save_path, epoch=train_data.epoch)
                        print("=> Saved checkpoint (epoch:%d): %s" %
                              (train_data.epoch, saved_path))

                        print('=== Test Data Evaluation ===')
                        # test-clean
                        cer_test_clean_epoch, wer_test_clean_epoch = do_eval_cer(
                            model=model,
                            model_type='ctc',
                            dataset=test_clean_data,
                            label_type=params['label_type'],
                            beam_width=1,
                            is_test=True,
                            eval_batch_size=1)
                        print('  CER (clean): %f %%' %
                              (cer_test_clean_epoch * 100))
                        print('  WER (clean): %f %%' %
                              (wer_test_clean_epoch * 100))

                        # test-other
                        cer_test_other_epoch, wer_test_other_epoch = do_eval_cer(
                            model=model,
                            model_type='ctc',
                            dataset=test_other_data,
                            label_type=params['label_type'],
                            beam_width=1,
                            is_test=True,
                            eval_batch_size=1)
                        print('  CER (other): %f %%' %
                              (cer_test_other_epoch * 100))
                        print('  WER (other): %f %%' %
                              (wer_test_other_epoch * 100))
                    else:
                        not_improved_epoch += 1
                else:
                    raise NotImplementedError

                duration_eval = time.time() - start_time_eval
                print('Evaluation time: %.3f min' % (duration_eval / 60))

                # Early stopping
                if not_improved_epoch == params['not_improved_patient_epoch']:
                    break

                # Update learning rate
                optimizer, learning_rate = lr_controller.decay_lr(
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    epoch=train_data.epoch,
                    value=metric_epoch)

                # ***Change to training mode***
                model.train()

            start_time_step = time.time()
            start_time_epoch = time.time()

    duration_train = time.time() - start_time_train
    print('Total time: %.3f hour' % (duration_train / 3600))

    # Training was finished correctly
    with open(join(model.save_path, 'complete.txt'), 'w') as f:
        f.write('')


def main(config_path, model_save_path):

    # Load a config file (.yml)
    with open(config_path, "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank class
    if params['label_type'] == 'character':
        params['num_classes'] = 28
    elif params['label_type'] == 'character_capital_divide':
        if params['data_size'] == '100h':
            params['num_classes'] = 72
        elif params['data_size'] == '460h':
            params['num_classes'] = 77
        elif params['data_size'] == '960h':
            params['num_classes'] = 77
    elif params['label_type'] == 'word_freq1':
        raise NotImplementedError
    elif params['label_type'] == 'word_freq5':
        raise NotImplementedError
    elif params['label_type'] == 'word_freq10':
        raise NotImplementedError
    elif params['label_type'] == 'word_freq15':
        raise NotImplementedError
    else:
        raise TypeError

    # Model setting
    model = CTC(
        input_size=params['input_size'],
        num_stack=params['num_stack'],
        splice=params['splice'],
        encoder_type=params['encoder_type'],
        bidirectional=params['bidirectional'],
        num_units=params['num_units'],
        num_proj=params['num_proj'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        num_classes=params['num_classes'],
        parameter_init=params['parameter_init'],
        logits_temperature=params['logits_temperature'])

    # Set process name
    setproctitle('pt_libri_ctc_' +
                 params['label_type'] + '_' + params['data_size'])

    model.name += '_' + str(params['num_units'])
    model.name += '_' + str(params['num_layers'])
    model.name += '_' + params['optimizer']
    model.name += '_lr' + str(params['learning_rate'])
    if params['num_proj'] != 0:
        model.name += '_proj' + str(params['num_proj'])
    if params['dropout'] != 0:
        model.name += '_drop' + str(params['dropout'])
    if params['num_stack'] != 1:
        model.name += '_stack' + str(params['num_stack'])
    if params['weight_decay'] != 0:
        model.name += '_wd' + str(params['weight_decay'])
    if params['bottleneck_dim'] != 0:
        model.name += '_bottle' + str(params['bottleneck_dim'])
    if params['logits_temperature'] != 1:
        model.name += '_temp' + str(params['logits_temperature'])

    # Set save path
    model.save_path = mkdir_join(
        model_save_path, 'ctc', params['label_type'], params['data_size'], model.name)

    # Reset model directory
    model_index = 0
    new_model_path = model.save_path
    while True:
        if isfile(join(new_model_path, 'complete.txt')):
            # Training of the first model have been finished
            model_index += 1
            new_model_path = model.save_path + '_' + str(model_index)
        elif isfile(join(new_model_path, 'config.yml')):
            # Training of the first model have not been finished yet
            model_index += 1
            new_model_path = model.save_path + '_' + str(model_index)
        else:
            break
    model.save_path = mkdir(new_model_path)

    # Save config file
    shutil.copyfile(config_path, join(model.save_path, 'config.yml'))

    # sys.stdout = open(join(model.save_path, 'train.log'), 'w')
    # TODO(hirofumi): change to logger
    do_train(model=model, params=params)


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 3:
        raise ValueError('Length of args should be 3.')

    main(config_path=args[1], model_save_path=args[2])
