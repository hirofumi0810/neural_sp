#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot attention weights (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isdir
import sys
import yaml
import argparse
import shutil

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.csj.data.load_dataset import Dataset
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.io.variable import var2np
from utils.directory import mkdir_join, mkdir
from utils.visualization.attention import plot_attention_weights


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--max_decode_len', type=int, default=100,  # or 60
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

    # Load model
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    # Restore the saved model
    checkpoint = model.load_checkpoint(
        save_path=args.model_path, epoch=args.epoch)
    model.load_state_dict(checkpoint['state_dict'])

    # Load dataset
    vocab_file_path = '../metrics/vocab_files/' + \
        params['label_type'] + '_' + params['data_size'] + '.txt'
    test_data = Dataset(
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='eval1',
        # data_type='eval2',
        # data_type='eval3',
        data_size=params['data_size'],
        label_type=params['label_type'], vocab_file_path=vocab_file_path,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, save_format=params['save_format'])

    # Visualize
    plot(model=model,
         dataset=test_data,
         max_decode_len=args.max_decode_len,
         eval_batch_size=args.eval_batch_size,
         save_path=mkdir_join(args.model_path, 'att_weights'))
    # save_path=None)


def plot(model, dataset, max_decode_len,
         eval_batch_size=None, save_path=None):
    """Visualize attention weights of Attetnion-based model.
    Args:
        model: model to evaluate
        dataset: An instance of a `Dataset` class
        max_decode_len (int): the length of output sequences
            to stop prediction when EOS token have not been emitted.
        eval_batch_size (int, optional): the batch size when evaluating the model
        save_path (string, optional): path to save attention weights plotting
    """
    # Clean directory
    if save_path is not None and isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    vocab_file_path = '../metrics/vocab_files/' + \
        dataset.label_type + '_' + dataset.data_size + '.txt'

    if 'word' in dataset.label_type:
        map_fn = Idx2word(vocab_file_path)
    else:
        map_fn = Idx2char(vocab_file_path)

    for batch, is_new_epoch in dataset:

        # Decode
        best_hyps, att_weights = model.attention_weights(
            batch['xs'], batch['x_lens'], max_decode_len=max_decode_len)

        for i_batch in range(len(batch['xs'])):

            # Check if the sum of attention weights equals to 1
            # print(np.sum(att_weights[i_batch], axis=1))

            str_pred = map_fn(best_hyps[i_batch])
            eos = True if '>' in str_pred else False

            str_pred = str_pred.split('>')[0]
            # NOTE: Trancate by <EOS>

            # Remove the last space
            if len(str_pred) > 0 and str_pred[-1] == ' ':
                str_pred = str_pred[:-1]

            if eos:
                str_pred += '_>'

            speaker = batch['input_names'][i_batch].split('_')[0]
            plot_attention_weights(
                spectrogram=batch['xs'][i_batch],
                attention_weights=att_weights[i_batch, :len(
                    str_pred.split('_')), :batch['x_lens'][i_batch]],
                label_list=str_pred.split('_'),
                save_path=mkdir_join(save_path, speaker,
                                     batch['input_names'][i_batch] + '.png'),
                fig_size=(20, 8))

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
