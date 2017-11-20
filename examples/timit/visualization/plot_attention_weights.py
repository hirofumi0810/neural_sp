#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot attention weights (TIMIT corpus)."""

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
from examples.timit.data.load_dataset import Dataset
from utils.io.labels.phone import Idx2phone
from utils.io.variable import np2var, var2np
from utils.directory import mkdir_join, mkdir


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--max_decode_length', type=int, default=40,
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
        params['num_classes'] = vocab_num[params['label_type']]

    # Load model
    model = load(model_type=params['model_type'], params=params)

    # Load dataset
    vocab_file_path = '../metrics/vocab_files/' + params['label_type'] + '.txt'
    test_data = Dataset(
        model_type='attetnion',
        data_type='test', label_type=params['label_type'],
        vocab_file_path=vocab_file_path,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False,
        use_cuda=model.use_cuda, volatile=True)

    # GPU setting
    model.set_cuda(deterministic=False)

    # Restore the saved model
    checkpoint = model.load_checkpoint(
        save_path=args.model_path, epoch=args.epoch)
    model.load_state_dict(checkpoint['state_dict'])

    # Change to evaluation mode
    model.eval()

    # Visualize
    plot(model=model,
         dataset=test_data,
         label_type=params['label_type'],
         # save_path=mkdir_join(model.save_path, 'att_weights'),
         save_path=None,
         show=True)


def plot(model, dataset, label_type, save_path=None, show=False):
    """Visualize attention weights of attetnion-based model.
    Args:
        model: model to evaluate
        dataset: An instance of a `Dataset` class
        label_type (string, optional): phone39 or phone48 or phone61
        is_test (bool, optional):
        save_path (string, optional): path to save attention weights plotting
        show (bool, optional): if True, show each figure
    """
    # Clean directory
    if save_path is not None and isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    idx2phone = Idx2phone(
        vocab_file_path='../metrics/vocab_files/' + label_type + '.txt')

    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        inputs, _, inputs_seq_len, _, input_names = data

        # Decode
        labels_pred, att_weights, _ = model.attention_weights(
            inputs, inputs_seq_len, beam_width=1, max_decode_length=40)

        for i_batch in range(inputs.size(0)):

            # Check if the sum of attention weights equals to 1
            # print(np.sum(att_weights[i_batch], axis=1))

            str_pred = idx2phone(labels_pred[i_batch]).split('>')[0]
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
