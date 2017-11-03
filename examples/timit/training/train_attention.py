#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile, abspath
import sys
import time
from setproctitle import setproctitle
import yaml
import shutil

import torch
import torch.nn as nn

sys.path.append(abspath('../../../'))
from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq
from examples.timit.data.load_dataset_attention import Dataset
# from examples.timit.metrics.attention import do_eval_per, do_eval_cer
# from utils.training.learning_rate_controller import Controller
from utils.training.plot import plot_loss, plot_ler
from utils.directory import mkdir_join, mkdir
from utils.io.tensor import to_np


def do_train(model, params):
    """Run training. If target labels are phone, the model is evaluated by PER
    with 39 phones.
    Args:
        model: the model to train
        params (dict): A dictionary of parameters
    """
    # Load dataset
    train_data = Dataset(
        data_type='train', label_type=params['label_type'],
        batch_size=params['batch_size'], max_epoch=params['num_epoch'],
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, sort_stop_epoch=params['sort_stop_epoch'])
    dev_data = Dataset(
        data_type='dev', label_type=params['label_type'],
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False)
    if 'char' in params['label_type']:
        test_data = Dataset(
            data_type='test', label_type=params['label_type'],
            batch_size=1, splice=params['splice'],
            num_stack=params['num_stack'], num_skip=params['num_skip'],
            sort_utt=False)
    else:
        test_data = Dataset(
            data_type='test', label_type='phone39',
            batch_size=1, splice=params['splice'],
            num_stack=params['num_stack'], num_skip=params['num_skip'],
            sort_utt=False)

    # Define optimizer
    optimizer, scheduler = model.set_optimizer(
        params['optimizer'],
        learning_rate_init=params['learning_rate'],
        weight_decay=params['weight_decay'],
        lr_schedule=False, factor=0.1, patience_epoch=5)

    # Initialize parameters
    model.init_weights()

    # Count total parameters
    print("Total %.3f M parameters" % (model.total_parameters / 1000000))

    # GPU setting
    use_cuda = torch.cuda.is_available()
    deterministic = False
    if use_cuda and deterministic:
        print('GPU deterministic mode (no cudnn)')
        torch.backends.cudnn.enabled = False
    elif use_cuda:
        print('GPU mode (faster than the deterministic mode)')
    else:
        print('CPU mode')

    # Train model
    start_time_train = time.time()
    start_time_epoch = time.time()
    start_time_step = time.time()
    ler_dev_best = 1
    learning_rate = float(params['learning_rate'])
    for step, (data, is_new_epoch) in enumerate(train_data):

        # Create feed dictionary for next mini batch (train)
        inputs, labels, inputs_seq_len, _ = data

        if use_cuda:
            model = model.cuda()
            inputs = inputs.cuda()
            labels = labels.cuda()

        # Clear gradients before
        optimizer.zero_grad()

        # Make prediction
        outputs_train, att_weights = model(inputs, labels)

        # Compute loss
        loss = model.compute_loss(outputs_train, labels, att_weights,
                                  coverage_weight=params['coverage_weight'])

        # Compute gradient
        optimizer.zero_grad()
        loss.backward()

        # Clip gradient norm
        nn.utils.clip_grad_norm(model.parameters(), params['clip_grad_norm'])

        # Update parameters
        # if scheduler is not None:
        #     scheduler.step(ler_train_pre)
        # else:
        optimizer.step()
        # TODO: fix this

        if (step + 1) % params['print_step'] == 0:

            # Create feed dictionary for next mini batch (dev)
            (inputs, labels, inputs_seq_len, _), _ = dev_data.next()

            # Change to evaluation mode

            # Decode
            outputs_infer, _ = model.decode_infer(
                inputs, labels,
                beam_width=params['beam_width'])

            # Compute accuracy

            duration_step = time.time() - start_time_step
            print("Step %d (epoch: %.3f): loss = %.3f (%.3f) / ler = %.3f (%.3f) / lr = %.5f (%.3f min)" %
                  (step + 1, train_data.epoch_detail, loss_train, loss_dev, ler_train, ler_dev,
                   learning_rate, duration_step / 60))
            sys.stdout.flush()
            start_time_step = time.time()

        # Save checkpoint and evaluate model per epoch
        if is_new_epoch:
            duration_epoch = time.time() - start_time_epoch
            print('-----EPOCH:%d (%.3f min)-----' %
                  (train_data.epoch, duration_epoch / 60))

            # Save fugure of loss & ler
            plot_loss(csv_loss_train, csv_loss_dev, csv_steps,
                      save_path=model.save_path)
            plot_ler(csv_ler_train, csv_ler_dev, csv_steps,
                     label_type=params['label_type'],
                     save_path=model.save_path)

            # Save the model
            # if save_path is not None:
            #     model.save_checkpoint(save_path, epoch=1)
            #     print("=> Saved checkpoint (epoch:%d): %s" %
            #           (1, save_path))


def main(config_path, model_save_path):
    # Load a config file (.yml)
    with open(config_path, "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank class
    if params['label_type'] == 'phone61':
        params['num_classes'] = 63
    elif params['label_type'] == 'phone48':
        params['num_classes'] = 50
    elif params['label_type'] == 'phone39':
        params['num_classes'] = 41
    elif params['label_type'] == 'character':
        params['num_classes'] = 30
    elif params['label_type'] == 'character_capital_divide':
        params['num_classes'] = 74

    # Model setting
    model = AttentionSeq2seq(
        input_size=params['input_size'] * params['num_stack'],
        splice=params['splice'],
        encoder_type=params['encoder_type'],
        encoder_bidirectional=params['bidirectional'],
        encoder_num_units=params['encoder_num_units'],
        #  encoder_num_proj,
        encoder_num_layers=params['encoder_num_units'],
        encoder_dropout=params['encoder_dropout'],
        attention_type=params['attention_type'],
        attention_dim=params['attention_dim'],
        decoder_type=params['decoder_type'],
        decoder_num_units=params['decoder_num_units'],
        decoder_num_proj=params['decoder_num_proj'],
        #   decdoder_num_layers,
        decoder_dropout=params['decoder_dropout'],
        embedding_dim=params['embedding_dim'],
        num_classes=params['num_classes'],
        eos_index=params['num_classes'],
        max_decode_length=params['max_decode_length'],
        parameter_init=params['parameter_init'],
        init_dec_state_with_enc_state=True,
        # downsample_list=[] if not downsample else [True] * 2,
        sharpening_factor=params['sharpening_factor'],
        logits_softmax_temperature=params['logits_temperature'])

    # Set process name
    setproctitle('pt_timit_' + model.name + '_' + params['label_type'])

    model.name += '_en' + str(params['encoder_num_units'])
    model.name += '_' + str(params['encoder_num_layers'])
    model.name += '_de' + str(params['decoder_num_units'])
    model.name += '_' + str(params['decoder_num_layers'])
    model.name += '_attdim' + str(params['attention_dim'])
    model.name += '_atttype' + str(params['attention_type'])
    model.name += '_' + params['optimizer']
    model.name += '_lr' + str(params['learning_rate'])
    # if params['dropout'] != 1:
    #     model.name += '_drop' + str(params['dropout'])
    # if params['num_stack'] != 1:
    #     model.name += '_stack' + str(params['num_stack'])
    # if params['weight_decay'] != 0:
    #     model.name += '_wd' + str(params['weight_decay'])

    # Set save path
    model.save_path = mkdir_join(
        model_save_path, 'attention', params['label_type'], model.name)

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
