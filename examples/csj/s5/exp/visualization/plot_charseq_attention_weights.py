#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot attention weights (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isdir
import sys
import argparse
import shutil

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_style("white")
blue = '#4682B4'
orange = '#D2691E'
green = '#006400'

# sns.set(font='IPAMincho')
sns.set(font='Noto Sans CJK JP')

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.csj.data.load_dataset_hierarchical import Dataset
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.io.variable import var2np
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
parser.add_argument('--max_decode_len', type=int, default=60,
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')
parser.add_argument('--max_decode_len_sub', type=int, default=100,
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    vocab_file_path = '../metrics/vocab_files/' + \
        params['label_type'] + '_' + params['data_size'] + '.txt'
    vocab_file_path_sub = '../metrics/vocab_files/' + \
        params['label_type_sub'] + '_' + params['data_size'] + '.txt'
    test_data = Dataset(
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        data_type='eval1',
        # data_type='eval2',
        # data_type='eval3',
        data_size=params['data_size'],
        label_type=params['label_type'], label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, reverse=True, save_format=params['save_format'])
    params['num_classes'] = test_data.num_classes
    params['num_classes_sub'] = test_data.num_classes_sub

    # Load model
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    # Visualize
    plot(model=model,
         dataset=test_data,
         max_decode_len=args.max_decode_len,
         max_decode_len_sub=args.max_decode_len_sub,
         eval_batch_size=args.eval_batch_size,
         save_path=mkdir_join(args.model_path, 'att_weights'))
    # save_path=None)


def plot(model, dataset, max_decode_len, max_decode_len_sub,
         eval_batch_size=None, save_path=None):
    """Visualize attention weights of Attetnion-based model.
    Args:
        model: model to evaluate
        dataset: An instance of a `Dataset` class
        max_decode_len (int): the length of output sequences
            to stop prediction when EOS token have not been emitted.
        max_decode_len_sub (int):
        eval_batch_size (int, optional): the batch size when evaluating the model
        save_path (string, optional): path to save attention weights plotting
    """
    # Clean directory
    if save_path is not None and isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    idx2word = Idx2word(
        vocab_file_path='../metrics/vocab_files/' +
        dataset.label_type + '_' + dataset.data_size + '.txt')
    idx2char = Idx2char(
        vocab_file_path='../metrics/vocab_files/' +
        dataset.label_type_sub + '_' + dataset.data_size + '.txt')

    for batch, is_new_epoch in dataset:

        if model.model_type == 'charseq_attention':
            best_hyps, best_hyps_sub, att_weights, char_att_weights = model.attention_weights(
                batch['xs'], batch['x_lens'],
                max_decode_len=max_decode_len,
                max_decode_len_sub=max_decode_len_sub)
        else:
            raise ValueError

        for b in range(len(batch['xs'])):

            # Check if the sum of attention weights equals to 1
            # print(np.sum(att_weights[b], axis=1))

            str_hyp = idx2word(best_hyps[b])
            str_hyp_sub = idx2char(best_hyps_sub[b])

            # TODO: eosで区切ってもattention weightsは打ち切られていない．

            speaker = batch['input_names'][b].split('_')[0]
            plot_attention_weights(
                att_weights[b, :len(str_hyp.split('_')), :batch['x_lens'][b]],
                frame_num=batch['x_lens'][b],
                num_stack=dataset.num_stack,
                label_list=str_hyp.split('_'),
                spectrogram=batch['xs'][b, :, :80],
                save_path=mkdir_join(save_path, speaker,
                                     batch['input_names'][b] + '.png'),
                figsize=(20, 8)
                # figsize=(14, 7)
            )

            plot_char_attention_weights(
                char_att_weights[b, :len(str_hyp.split(
                    '_')), :len(list(str_hyp_sub))],
                label_list=str_hyp.split('_'),
                label_list_sub=list(str_hyp_sub),
                save_path=mkdir_join(save_path, speaker,
                                     batch['input_names'][b] + '_char_attend.png'),
                figsize=(20, 8)
                # figsize=(14, 7)
            )

        if is_new_epoch:
            break


def plot_char_attention_weights(attention_weights, label_list, label_list_sub,
                                save_path=None, figsize=(10, 4)):
    """Plot attention weights.
    Args:
        attention_weights (np.ndarray): A tensor of size `[T_out, T_in]`
        label_list (list):
        label_list_sub (list):
        save_path (string): path to save a figure of CTC posterior (utterance)
        figsize (tuple):
    """
    plt.clf()
    plt.figure(figsize=figsize)

    # Plot attention weights
    sns.heatmap(attention_weights,
                # cmap='Blues',
                cmap='viridis',
                xticklabels=label_list_sub,
                yticklabels=label_list)
    # cbar_kws={"orientation": "horizontal"}
    plt.ylabel('Output labels (←)', fontsize=12)
    plt.yticks(rotation=0)

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path, dvi=500)

    plt.close()


if __name__ == '__main__':
    main()
