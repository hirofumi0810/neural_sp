#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot attention weights (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isdir
import sys
import numpy as np
import yaml
import argparse
import shutil

import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_style("white")
blue = '#4682B4'
orange = '#D2691E'
green = '#006400'

sys.path.append(abspath('../../../'))
from examples.timit.data.load_dataset_attention import Dataset
from utils.io.labels.phone import Idx2phone
from utils.io.variable import np2var
from utils.directory import mkdir_join, mkdir
from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--eval_batch_size', type=str, default=1,
                    help='the size of mini-batch in evaluation')


def do_plot(model, params, epoch, eval_batch_size):
    """Decode the Attention outputs.
    Args:
        model: the model to restore
        params (dict): A dictionary of parameters
        epoch (int): the epoch to restore
        eval_batch_size (int): the size of mini-batch when evaluation
    """
    if 'kana' in params['label_type']:
        vocab_file_path = '../metrics/vocab_files/' + \
            params['label_type'] + '.txt'
    else:
        vocab_file_path = '../metrics/vocab_files/' + \
            params['label_type'] + '_' + params['data_size'] + '.txt'

    # Load dataset
    test_data = Dataset(
        data_type='eval1', label_type=params['label_type'],
        data_size=params['data_size'],
        batch_size=eval_batch_size,
        vocab_file_path=vocab_file_path,
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False)

    # GPU setting
    model.set_cuda(deterministic=False)

    # Load the saved model
    checkpoint = model.load_checkpoint(save_path=model.save_path, epoch=epoch)
    model.load_state_dict(checkpoint['state_dict'])

    # Change to evaluation mode
    model.eval()

    # Visualize
    plot(model=model,
         dataset=test_data,
         label_type=params['label_type'],
         data_size=params['data_size'],
         is_test=test_data.is_test,
         # save_path=mkdir_join(model.save_path, 'attention_weights'),
         save_path=None,
         show=True)


def plot(model, dataset, label_type,
         is_test=False, save_path=None, show=False):
    """Visualize attention weights of Attetnion-based model.
    Args:
        model: model to evaluate
        dataset: An instance of a `Dataset` class
        label_type (string, optional): phone39 or phone48 or phone61 or character or
            character_capital_divide
        is_test (bool, optional):
        save_path (string, optional): path to save attention weights plotting
        show (bool, optional): if True, show each figure
    """
    # Clean directory
    if save_path is not None and isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    idx2phone = Idx2phone('../metrics/vocab_files/' + label_type + '.txt')

    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        inputs, labels_true, _, labels_seq_len, input_names = data
        inputs = np2var(
            inputs, use_cuda=model.use_cuda, volatile=True)

        batch_size = inputs[0].size(0)

        # Make prediction
        labels_pred, attention_weights = model.decode_infer(
            inputs[0], beam_width=1)

        for i_batch in range(batch_size):

            # Check if the sum of attention weights equals to 1
            # print(np.sum(attention_weights[i_batch], axis=1))

            str_pred = idx2phone(labels_pred[i_batch]).split('>')[0]
            # NOTE: Trancate by <EOS>

            # Remove the last space
            if len(str_pred) > 0 and str_pred[-1] == ' ':
                str_pred = str_pred[:-1]

            plt.clf()
            plt.figure(figsize=(10, 4))
            sns.heatmap(attention_weights[i_batch],
                        cmap='Blues',
                        xticklabels=False,
                        yticklabels=str_pred.split(' '))

            plt.xlabel('Input frames', fontsize=12)
            plt.ylabel('Output labels (top to bottom)', fontsize=12)

            if show:
                plt.show()

            # Save as a png file
            if save_path is not None:
                plt.savefig(join(save_path, input_names[0] + '.png'), dvi=500)

            plt.close()

        if is_new_epoch:
            break


def main():

    args = parser.parse_args()

    # Load config file
    with open(join(args.model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a <SOS> and <EOS> class
    if params['label_type'] == 'kana':
        params['num_classes'] = 146
    elif params['label_type'] == 'kana_divide':
        params['num_classes'] = 147
    elif params['label_type'] == 'kanji':
        if params['data_size'] == 'subset':
            params['num_classes'] = 2978
        elif params['data_size'] == 'fullset':
            params['num_classes'] = 3383
    elif params['label_type'] == 'kanji_divide':
        if params['data_size'] == 'subset':
            params['num_classes'] = 2979
        elif params['data_size'] == 'fullset':
            params['num_classes'] = 3384
    elif params['label_type'] == 'word_freq1':
        if params['data_size'] == 'subset':
            params['num_classes'] = 39169
        elif params['data_size'] == 'fullset':
            params['num_classes'] = 66277
    elif params['label_type'] == 'word_freq5':
        if params['data_size'] == 'subset':
            params['num_classes'] = 12877
        elif params['data_size'] == 'fullset':
            params['num_classes'] = 23528
    elif params['label_type'] == 'word_freq10':
        if params['data_size'] == 'subset':
            params['num_classes'] = 8542
        elif params['data_size'] == 'fullset':
            params['num_classes'] = 15536
    elif params['label_type'] == 'word_freq15':
        if params['data_size'] == 'subset':
            params['num_classes'] = 6726
        elif params['data_size'] == 'fullset':
            params['num_classes'] = 12111
    else:
        TypeError

    # Model setting
    model = AttentionSeq2seq(
        input_size=params['input_size'],
        num_stack=params['num_stack'],
        splice=params['splice'],
        encoder_type=params['encoder_type'],
        encoder_bidirectional=params['encoder_bidirectional'],
        encoder_num_units=params['encoder_num_units'],
        encoder_num_proj=params['encoder_num_proj'],
        encoder_num_layers=params['encoder_num_layers'],
        encoder_dropout=params['dropout_encoder'],
        attention_type=params['attention_type'],
        attention_dim=params['attention_dim'],
        decoder_type=params['decoder_type'],
        decoder_num_units=params['decoder_num_units'],
        decoder_num_proj=params['decoder_num_proj'],
        decdoder_num_layers=params['decoder_num_layers'],
        decoder_dropout=params['dropout_decoder'],
        embedding_dim=params['embedding_dim'],
        embedding_dropout=params['dropout_embedding'],
        num_classes=params['num_classes'],
        sos_index=params['num_classes'],
        eos_index=params['num_classes'] + 1,
        max_decode_length=params['max_decode_length'],
        parameter_init=params['parameter_init'],
        downsample_list=[],
        init_dec_state_with_enc_state=True,
        sharpening_factor=params['sharpening_factor'],
        logits_temperature=params['logits_temperature'],
        sigmoid_smoothing=params['sigmoid_smoothing'],
        input_feeding_approach=params['input_feeding_approach'])

    model.save_path = args.model_path
    do_plot(model=model, params=params,
            epoch=args.epoch, eval_batch_size=args.eval_batch_size)


if __name__ == '__main__':
    main()
