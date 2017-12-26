#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot attention weights (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isdir
import sys
import yaml
import argparse
import shutil

sys.path.append(abspath('../../../'))
from models.pytorch.load_model import load
from examples.timit.data.load_dataset import Dataset
from utils.io.labels.phone import Idx2phone
from utils.io.variable import np2var, var2np
from utils.directory import mkdir_join, mkdir
from utils.visualization.attention import plot_attention_weights


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--beam_width', type=int, default=1,
                    help='beam_width (int, optional): beam width for beam search.' +
                    ' 1 disables beam search, which mean greedy decoding.')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--max_decode_len', type=int, default=40,
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

    # GPU setting
    model.set_cuda(deterministic=False)

    # Restore the saved model
    checkpoint = model.load_checkpoint(
        save_path=args.model_path, epoch=args.epoch)
    model.load_state_dict(checkpoint['state_dict'])

    # ***Change to evaluation mode***
    model.eval()

    # Load dataset
    vocab_file_path = '../metrics/vocab_files/' + params['label_type'] + '.txt'
    test_data = Dataset(
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
                   label_type=params['label_type'],
                   beam_width=args.beam_width,
                   max_decode_len=args.max_decode_len,
                   eval_batch_size=args.eval_batch_size,
                   save_path=mkdir_join(args.model_path, 'att_weights'))


def plot_attention(model, dataset, label_type, beam_width, max_decode_len,
                   eval_batch_size=None, save_path=None):
    """Visualize attention weights of attetnion-based model.
    Args:
        model: model to evaluate
        dataset: An instance of a `Dataset` class
        label_type (string, optional): phone39 or phone48 or phone61
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
        vocab_file_path='../metrics/vocab_files/' + label_type + '.txt')

    for batch, is_new_epoch in dataset:

        inputs, _, inputs_seq_len, _, input_names = batch

        # Decode
        labels_pred, attention_weights = model.attention_weights(
            inputs, inputs_seq_len,
            beam_width=beam_width,
            max_decode_len=max_decode_len)
        # NOTE: attention_weights: `[B, T_out, T_in]`

        for i_batch in range(inputs.shape[0]):

            # Check if the sum of attention weights equals to 1
            # print(np.sum(attention_weights[i_batch], axis=1))

            str_pred = idx2phone(labels_pred[i_batch]).split('>')[0]
            # NOTE: Trancate by <EOS>

            # Remove the last space
            if len(str_pred) > 0 and str_pred[-1] == ' ':
                str_pred = str_pred[:-1]

            plot_attention_weights(
                spectrogram=inputs[i_batch],
                attention_weights=attention_weights[i_batch, :len(
                    str_pred.split(' ')), :inputs_seq_len[i_batch]],
                label_list=str_pred.split(' '),
                save_path=join(save_path, input_names[i_batch] + '.png'),
                fig_size=(14, 7))

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
