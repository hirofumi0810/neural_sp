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
from models.pytorch.load_model import load
from examples.csj.data.load_dataset_attention import Dataset
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.io.variable import var2np
from utils.directory import mkdir_join, mkdir


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=str, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--max_decode_length', type=int, default=100,  # or 60
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')


def main():

    args = parser.parse_args()

    # Load config file
    with open(join(args.model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Get voabulary number (excluding blank, <SOS>, <EOS> classes)
    with open('../metrics/vocab_num.yml', "r") as f:
        vocab_num = yaml.load(f)
        params['num_classes'] = vocab_num[params['data_size']
                                          ][params['label_type']]

    # Model setting
    model = load(model_type=params['model_type'], params=params)

    # Load dataset
    test_data = Dataset(
        data_type='eval1',
        # data_type='eval2',
        # data_type='eval3',
        data_size=params['data_size'],
        label_type=params['label_type'], num_classes=params['num_classes'],
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False,
        use_cuda=model.use_cuda, volatile=True)

    # GPU setting
    model.set_cuda(deterministic=False)

    # Load the saved model
    checkpoint = model.load_checkpoint(
        save_path=args.model_path, epoch=args.epoch)
    model.load_state_dict(checkpoint['state_dict'])

    # Change to evaluation mode
    model.eval()

    # Visualize
    plot(model=model,
         dataset=test_data,
         label_type=params['label_type'],
         data_size=params['data_size'],
         # save_path=mkdir_join(model.save_path, 'att_weights'),
         save_path=None,
         show=True)


def plot(model, dataset, data_size, label_type, save_path=None, show=False):
    """Visualize attention weights of Attetnion-based model.
    Args:
        model: model to evaluate
        dataset: An instance of a `Dataset` class
        data_size (string): subset or fullset
        label_type (string): kanji or kanji or kanji_divide or kana_divide or
            word_freq1 or word_freq5 or word_freq10 or word_freq15
        save_path (string, optional): path to save attention weights plotting
        show (bool, optional): if True, show each figure
    """
    # Clean directory
    if save_path is not None and isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    if label_type == 'character':
        vocab_file_path = '../metrics/vocab_files/character.txt'
    else:
        vocab_file_path = '../metrics/vocab_files/' + \
            label_type + '_' + data_size + '.txt'

    if 'char' in label_type:
        map_fn = Idx2char(vocab_file_path)
    else:
        map_fn = Idx2word(vocab_file_path)

    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        inputs, _, inputs_seq_len, _, input_names = data

        batch_size = inputs[0].size(0)

        # Decode
        labels_pred, att_weights, _ = model.attention_weights(
            inputs[0], inputs_seq_len[0], beam_width=1, max_decode_length=100)

        for i_batch in range(batch_size):

            # Check if the sum of attention weights equals to 1
            # print(np.sum(att_weights[i_batch], axis=1))

            str_pred = map_fn(labels_pred[i_batch]).split('>')[0]
            # NOTE: Trancate by <EOS>

            # Remove the last space
            if len(str_pred) > 0 and str_pred[-1] == ' ':
                str_pred = str_pred[:-1]

            plt.clf()
            plt.figure(figsize=(10, 4))
            sns.heatmap(att_weights[i_batch],
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


if __name__ == '__main__':
    main()
