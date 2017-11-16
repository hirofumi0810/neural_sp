#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the model (Librispeech corpus)."""

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
from models.pytorch.ctc.hierarchical_ctc import HierarchicalCTC
from models.pytorch.attention.hierarchical_attention_seq2seq import HierarchicalAttentionSeq2seq

from examples.librispeech.data.load_dataset_hierarchical_ctc import Dataset as Dataset_hierarchical_ctc
from examples.librispeech.data.load_dataset_hierarchical_attention import Dataset as Dataset_hierarchical_attention

# from examples.librispeech.metrics.cer import do_eval_cer
from examples.librispeech.metrics.wer import do_eval_wer
from utils.training.learning_rate_controller import Controller
from utils.training.plot import plot_loss
from utils.directory import mkdir_join, mkdir
from utils.io.variable import np2var, var2np


def do_train(model, params):
    """Run training.
    Args:
        model: the model to train
        params (dict): A dictionary of parameters
    """
    # Load dataset
    vocab_file_path = '../metrics/vocab_files/' + \
        params['label_type'] + '_' + params['data_size'] + '.txt'
    if params['label_type_sub'] == 'character':
        vocab_file_path_sub = '../metrics/vocab_files/character.txt'
    else:
        vocab_file_path_sub = '../metrics/vocab_files/' + \
            params['label_type_sub'] + '_' + params['data_size'] + '.txt'
    if params['model_type'] == 'hierarchical_ctc':
        Dataset = Dataset_hierarchical_ctc
    elif params['model_type'] == 'hierarchical_attention':
        Dataset = Dataset_hierarchical_attention
    train_data = Dataset(
        data_type='train', data_size=params['data_size'],
        label_type=params['label_type'],
        label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=params['batch_size'],
        max_epoch=params['num_epoch'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, sort_stop_epoch=params['sort_stop_epoch'])
    dev_clean_data = Dataset(
        data_type='dev_clean', data_size=params['data_size'],
        label_type=params['label_type'],
        label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, num_gpus=1)
    dev_other_data = Dataset(
        data_type='dev_other', data_size=params['data_size'],
        label_type=params['label_type'],
        label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, num_gpus=1)
    test_clean_data = Dataset(
        data_type='test_clean', data_size=params['data_size'],
        label_type=params['label_type'],
        label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False)
    test_other_data = Dataset(
        data_type='test_other', data_size=params['data_size'],
        label_type=params['label_type'],
        label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
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
        if params['model_type'] in ['hierarchical_ctc', 'hierarchical_attention']:
            inputs, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub, _ = data

        # Wrap by variable
        inputs = np2var(inputs, use_cuda=use_cuda)
        if params['model_type'] == 'hierarchical_ctc':
            labels = np2var(labels, use_cuda=use_cuda, dtype='int') + 1
            labels_sub = np2var(labels_sub, use_cuda=use_cuda, dtype='int') + 1
            # NOTE: index 0 is reserved for blank
        elif params['model_type'] == 'hierarchical_attention':
            labels = np2var(labels, use_cuda=use_cuda, dtype='long')
            labels_sub = np2var(labels_sub, use_cuda=use_cuda, dtype='long')
        inputs_seq_len = np2var(inputs_seq_len, use_cuda=use_cuda, dtype='int')
        labels_seq_len = np2var(labels_seq_len, use_cuda=use_cuda, dtype='int')
        labels_seq_len_sub = np2var(
            labels_seq_len_sub, use_cuda=use_cuda, dtype='int')

        # Clear gradients before
        optimizer.zero_grad()

        # Compute loss in the training set
        if params['model_type'] == 'hierarchical_ctc':
            logits, logits_sub, perm_indices = model(
                inputs[0], inputs_seq_len[0])
            loss_train = model.compute_loss(
                logits,
                labels[0][perm_indices],
                inputs_seq_len[0][perm_indices],
                labels_seq_len[0][perm_indices])
            loss_train += model.compute_loss(
                logits_sub,
                labels_sub[0][perm_indices],
                inputs_seq_len[0][perm_indices],
                labels_seq_len_sub[0][perm_indices])
        elif params['model_type'] == 'hierarchical_attention':
            logits, att_weights, logits_sub, att_weights_sub, perm_indices = model(
                inputs[0], inputs_seq_len[0], labels[0], labels_sub[0])
            loss_train = model.compute_loss(
                logits,
                labels[0][perm_indices],
                labels_seq_len[0][perm_indices],
                logits_sub,
                labels_sub[0][perm_indices],
                labels_seq_len_sub[0][perm_indices],
                att_weights,
                att_weights_sub,
                coverage_weight=params['coverage_weight'])

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
                if params['model_type'] in ['hierarchical_ctc', 'hierarchical_attention']:
                    inputs, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub, _ = dev_clean_data.next()[
                        0]
            else:
                if params['model_type'] in ['hierarchical_ctc', 'hierarchical_attention']:
                    inputs, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub, _ = dev_other_data.next()[
                        0]

            # Wrap by variable
            inputs = np2var(inputs, use_cuda=use_cuda, volatile=True)
            if params['model_type'] == 'hierarchical_ctc':
                labels = np2var(
                    labels, use_cuda=use_cuda, volatile=True, dtype='int') + 1
                labels_sub = np2var(
                    labels_sub, use_cuda=use_cuda, volatile=True, dtype='int') + 1
                # NOTE: index 0 is reserved for blank
            elif params['model_type'] == 'hierarchical_attention':
                labels = np2var(labels, use_cuda=use_cuda,
                                volatile=True, dtype='long')
                labels_sub = np2var(labels_sub, use_cuda=use_cuda,
                                    volatile=True, dtype='long')
            inputs_seq_len = np2var(
                inputs_seq_len, use_cuda=use_cuda, volatile=True, dtype='int')
            labels_seq_len = np2var(
                labels_seq_len, use_cuda=use_cuda, volatile=True, dtype='int')
            labels_seq_len_sub = np2var(
                labels_seq_len_sub, use_cuda=use_cuda, volatile=True, dtype='int')

            # ***Change to evaluation mode***
            model.eval()

            # Compute loss in the dev set
            if params['model_type'] == 'hierarchical_ctc':
                logits, logits_sub, perm_indices = model(
                    inputs[0], inputs_seq_len[0], volatile=True)
                loss_dev = model.compute_loss(
                    logits,
                    labels[0][perm_indices],
                    inputs_seq_len[0][perm_indices],
                    labels_seq_len[0][perm_indices])
                loss_dev += model.compute_loss(
                    logits_sub,
                    labels_sub[0][perm_indices],
                    inputs_seq_len[0][perm_indices],
                    labels_seq_len_sub[0][perm_indices])
            elif params['model_type'] == 'hierarchical_attention':
                logits, att_weights, logits_sub, att_weights_sub, perm_indices = model(
                    inputs[0], inputs_seq_len[0], labels[0], volatile=True)
                loss_dev = model.compute_loss(
                    logits,
                    labels[0][perm_indices],
                    labels_seq_len[0][perm_indices],
                    logits_sub,
                    labels_sub[0][perm_indices],
                    labels_seq_len_sub[0][perm_indices],
                    att_weights,
                    att_weights_sub,
                    coverage_weight=params['coverage_weight'])

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
            sys.stdout.flush()
            start_time_step = time.time()

        # Save checkpoint and evaluate model per epoch
        if is_new_epoch:
            duration_epoch = time.time() - start_time_epoch
            print('-----EPOCH:%d (%.3f min)-----' %
                  (train_data.epoch, duration_epoch / 60))

            # Save fugure of loss
            plot_loss(csv_loss_train, csv_loss_dev, csv_steps,
                      save_path=model.save_path)

            # Save the model
            saved_path = model.save_checkpoint(
                model.save_path, epoch=train_data.epoch)
            print("=> Saved checkpoint (epoch:%d): %s" %
                  (train_data.epoch, saved_path))

            if train_data.epoch >= params['eval_start_epoch']:
                # ***Change to evaluation mode***
                model.eval()

                start_time_eval = time.time()
                print('=== Dev Data Evaluation ===')

                # dev-clean
                wer_dev_clean_epoch = do_eval_wer(
                    model=model,
                    model_type=params['model_type'],
                    dataset=dev_clean_data,
                    label_type=params['label_type'],
                    data_size=params['data_size'],
                    beam_width=1,
                    eval_batch_size=1)
                print('  WER (clean): %f %%' % (wer_dev_clean_epoch * 100))

                # dev-other
                wer_dev_other_epoch = do_eval_wer(
                    model=model,
                    model_type=params['model_type'],
                    dataset=dev_other_data,
                    label_type=params['label_type'],
                    data_size=params['data_size'],
                    beam_width=1,
                    eval_batch_size=1)
                print('  WER (other): %f %%' % (wer_dev_other_epoch * 100))

                if params['data_size'] in ['100h', '460h']:
                    metric_epoch = wer_dev_clean_epoch
                else:
                    metric_epoch = wer_dev_other_epoch

                if metric_epoch < ler_dev_best:
                    ler_dev_best = metric_epoch
                    not_improved_epoch = 0
                    print('■■■ ↑Best Score (WER)↑ ■■■')

                    print('=== Test Data Evaluation ===')
                    # test-clean
                    wer_test_clean_epoch = do_eval_wer(
                        model=model,
                        model_type=params['model_type'],
                        dataset=test_clean_data,
                        label_type=params['label_type'],
                        data_size=params['data_size'],
                        beam_width=1,
                        is_test=True,
                        eval_batch_size=1)
                    print('  WER (clean): %f %%' %
                          (wer_test_clean_epoch * 100))

                    # test-other
                    wer_test_other_epoch = do_eval_wer(
                        model=model,
                        model_type=params['model_type'],
                        dataset=test_other_data,
                        label_type=params['label_type'],
                        data_size=params['data_size'],
                        beam_width=1,
                        is_test=True,
                        eval_batch_size=1)
                    print('  WER (other): %f %%' %
                          (wer_test_other_epoch * 100))
                else:
                    not_improved_epoch += 1

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

    # Except for blank, <SOS>, <EOS> classes
    if params['label_type'] == 'word_freq1':
        if params['data_size'] == '100h':
            params['num_classes'] = 33779
        elif params['data_size'] == '460h':
            params['num_classes'] = 65988
        elif params['data_size'] == '960h':
            params['num_classes'] = 89115
    elif params['label_type'] == 'word_freq5':
        if params['data_size'] == '100h':
            params['num_classes'] = 11735
        elif params['data_size'] == '460h':
            params['num_classes'] = 27140
        elif params['data_size'] == '960h':
            params['num_classes'] = 37271
    elif params['label_type'] == 'word_freq10':
        if params['data_size'] == '100h':
            params['num_classes'] = 7213
        elif params['data_size'] == '460h':
            params['num_classes'] = 18641
        elif params['data_size'] == '960h':
            params['num_classes'] = 26642
    elif params['label_type'] == 'word_freq15':
        if params['data_size'] == '100h':
            params['num_classes'] = 5219
        elif params['data_size'] == '460h':
            params['num_classes'] = 14498
        elif params['data_size'] == '960h':
            params['num_classes'] = 21409
    else:
        raise TypeError

    if params['label_type_sub'] == 'character':
        params['num_classes_sub'] = 28
    elif params['label_type_sub'] == 'character_capital_divide':
        if params['data_size'] == '100h':
            params['num_classes_sub'] = 72
        elif params['data_size'] == '460h':
            params['num_classes_sub'] = 77
        elif params['data_size'] == '960h':
            params['num_classes_sub'] = 77
    else:
        raise TypeError

    # Model setting
    if params['model_type'] == 'hierarchical_ctc':
        model = HierarchicalCTC(
            input_size=params['input_size'],
            num_stack=params['num_stack'],
            splice=params['splice'],
            encoder_type=params['encoder_type'],
            bidirectional=params['bidirectional'],
            num_units=params['num_units'],
            num_proj=params['num_proj'],
            num_layers=params['num_layers'],
            num_layers_sub=params['num_layers_sub'],
            dropout=params['dropout'],
            num_classes=params['num_classes'],
            num_classes_sub=params['num_classes_sub'],
            parameter_init=params['parameter_init'],
            logits_temperature=params['logits_temperature'])

        # Set process name
        setproctitle('pt_libri_ctc_' +
                     params['label_type'] + '_' + params['label_type_sub'] + '_' + params['data_size'])

        model.name += '_' + str(params['num_units'])
        model.name += '_' + str(params['num_layers'])
        model.name += '_' + str(params['num_layers_sub'])
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

    elif params['model_type'] == 'hierarchical_attention':
        model = HierarchicalAttentionSeq2seq(
            input_size=params['input_size'],
            num_stack=params['num_stack'],
            splice=params['splice'],
            encoder_type=params['encoder_type'],
            encoder_bidirectional=params['encoder_bidirectional'],
            encoder_num_units=params['encoder_num_units'],
            encoder_num_proj=params['encoder_num_proj'],
            encoder_num_layers=params['encoder_num_layers'],
            encoder_num_layers_sub=params['encoder_num_layers_sub'],
            encoder_dropout=params['dropout_encoder'],
            attention_type=params['attention_type'],
            attention_dim=params['attention_dim'],
            decoder_type=params['decoder_type'],
            decoder_num_units=params['decoder_num_units'],
            decoder_num_proj=params['decoder_num_proj'],
            decoder_num_layers=params['decoder_num_layers'],
            decoder_num_units_sub=params['decoder_num_units_sub'],
            decoder_num_proj_sub=params['decoder_num_proj_sub'],
            decoder_num_layers_sub=params['decoder_num_layers_sub'],
            decoder_dropout=params['dropout_decoder'],
            embedding_dim=params['embedding_dim'],
            embedding_dim_sub=params['embedding_dim_sub'],
            embedding_dropout=params['dropout_embedding'],
            num_classes=params['num_classes'],
            num_classes_sub=params['num_classes_sub'],
            max_decode_length=params['max_decode_length'],
            max_decode_length_sub=params['max_decode_length_sub'],
            parameter_init=params['parameter_init'],
            downsample_list=[],
            init_dec_state_with_enc_state=True,
            sharpening_factor=params['sharpening_factor'],
            logits_temperature=params['logits_temperature'],
            sigmoid_smoothing=params['sigmoid_smoothing'],
            input_feeding_approach=params['input_feeding_approach'])

        # Set process name
        setproctitle('pt_libri_att_' +
                     params['label_type'] + '_' + params['label_type_sub'] + '_' + params['data_size'])

        model.name = 'enc' + params['encoder_type'] + \
            str(params['encoder_num_units'])
        model.name += '_' + str(params['encoder_num_layers'])
        model.name += '_' + str(params['encoder_num_layers_sub'])
        model.name += '_att' + str(params['attention_dim'])
        model.name += '_dec' + params['decoder_type'] + \
            str(params['decoder_num_units'])
        model.name += '_' + str(params['decoder_num_layers'])
        model.name += '_' + params['optimizer']
        model.name += '_lr' + str(params['learning_rate'])
        model.name += '_' + params['attention_type']
        if params['dropout_encoder'] != 0:
            model.name += '_dropen' + str(params['dropout_encoder'])
        if params['dropout_decoder'] != 0:
            model.name += '_dropde' + str(params['dropout_decoder'])
        if params['dropout_embedding'] != 0:
            model.name += '_dropem' + str(params['dropout_embedding'])
        if params['num_stack'] != 1:
            model.name += '_stack' + str(params['num_stack'])
        if params['weight_decay'] != 0:
            model.name += 'wd' + str(params['weight_decay'])
        if params['sharpening_factor'] != 1:
            model.name += '_sharp' + str(params['sharpening_factor'])
        if params['logits_temperature'] != 1:
            model.name += '_temp' + str(params['logits_temperature'])
        if bool(params['sigmoid_smoothing']):
            model.name += '_smoothing'
        if bool(params['input_feeding_approach']):
            model.name += '_infeed'

    # Set save path
    model.save_path = mkdir_join(
        model_save_path, params['model_type'],
        params['label_type'] + '_' + params['label_type_sub'],
        params['data_size'], model.name)

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

    sys.stdout = open(join(model.save_path, 'train.log'), 'w')
    # TODO(hirofumi): change to logger
    do_train(model=model, params=params)


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 3:
        raise ValueError('Length of args should be 3.')

    main(config_path=args[1], model_save_path=args[2])
