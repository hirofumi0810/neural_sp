#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot attention weights (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isdir
import sys
import argparse
import shutil

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.timit.data.load_dataset import Dataset
from utils.io.labels.phone import Idx2phone
from utils.io.variable import np2var, var2np
from utils.directory import mkdir_join, mkdir
from utils.visualization.attention import plot_attention_weights
from utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--max_decode_len', type=int, default=40,
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'))

    # Load model
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    # Load dataset
    vocab_file_path = '../metrics/vocab_files/' + params['label_type'] + '.txt'
    test_data = Dataset(
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type='attetnion',
        data_type='test', label_type=params['label_type'],
        vocab_file_path=vocab_file_path,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, save_format=params['save_format'])

    # Visualize
    plot_attention(model=model,
                   dataset=test_data,
                   max_decode_len=args.max_decode_len,
                   eval_batch_size=args.eval_batch_size,
                   save_path=mkdir_join(args.model_path, 'att_weights'))


def plot_attention(model, dataset, max_decode_len,
                   eval_batch_size=None, save_path=None):
    """Visualize attention weights of the attetnion-based model.
    Args:
        model: model to evaluate
        dataset: An instance of a `Dataset` class
        max_decode_len (int): the length of output sequences
            to stop prediction when EOS token have not been emitted.
        eval_batch_size (int, optional): the batch size when evaluating the model
        save_path (string, optional): path to save attention weights plotting
    """
    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    # Clean directory
    if save_path is not None and isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    idx2phone = Idx2phone(
        vocab_file_path='../metrics/vocab_files/' + dataset.label_type + '.txt')

    for batch, is_new_epoch in dataset:

        # Decode
        best_hyps, attention_weights = model.attention_weights(
            batch['xs'], batch['x_lens'], max_decode_len=max_decode_len)
        # NOTE: attention_weights: `[B, T_out, T_in]`

        for i_batch in range(len(batch['xs'])):

            # Check if the sum of attention weights equals to 1
            # print(np.sum(attention_weights[i_batch], axis=1))

            str_pred = idx2phone(best_hyps[i_batch])
            eos = True if '>' in str_pred else False

            str_pred = str_pred.split('>')[0]
            # NOTE: Trancate by <EOS>

            # Remove the last space
            if len(str_pred) > 0 and str_pred[-1] == ' ':
                str_pred = str_pred[:-1]

            if eos:
                str_pred += ' >'

            plot_attention_weights(
                spectrogram=batch['xs'][i_batch],
                attention_weights=attention_weights[i_batch, :len(
                    str_pred.split(' ')), :batch['x_lens'][i_batch] - 1],
                label_list=str_pred.split(' '),
                save_path=join(
                    save_path, batch['input_names'][i_batch] + '.png'),
                fig_size=(14, 7))

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
