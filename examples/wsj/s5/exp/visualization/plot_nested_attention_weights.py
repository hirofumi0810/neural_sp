#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot attention weights of the nested attention model (WSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isdir
import sys
import argparse
import shutil
import numpy as np

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
from examples.wsj.s5.exp.dataset.load_dataset_hierarchical import Dataset
from utils.directory import mkdir_join, mkdir
from utils.visualization.attention import plot_hierarchical_attention_weights
from utils.config import load_config
from examples.csj.s5.exp.visualization.plot_nested_attention_weights import plot_word2char_attention_weights

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str,
                    help='path to saved data')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--beam_width', type=int, default=1,
                    help='the size of beam in the main task')
parser.add_argument('--beam_width_sub', type=int, default=1,
                    help='the size of beam in the sub task')
parser.add_argument('--length_penalty', type=float, default=0,
                    help='length penalty in beam search decoding')
parser.add_argument('--coverage_penalty', type=float, default=0,
                    help='coverage penalty in beam search decoding')

MAX_DECODE_LEN_WORD = 32
MIN_DECODE_LEN_WORD = 2
MAX_DECODE_LEN_CHAR = 199
MIN_DECODE_LEN_CHAR = 10


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    dataset = Dataset(
        data_save_path=args.data_save_path,
        backend=params['backend'],
        input_freq=params['input_freq'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        data_type='test_eval92',
        data_size=params['data_size'],
        label_type=params['label_type'], label_type_sub=params['label_type_sub'],
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, reverse=False, tool=params['tool'])

    params['num_classes'] = dataset.num_classes
    params['num_classes_sub'] = dataset.num_classes_sub

    # Load model
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    a2c_oracle = False

    save_path = mkdir_join(args.model_path, 'att_weights')

    ######################################################################

    # Clean directory
    if save_path is not None and isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    for batch, is_new_epoch in dataset:
        batch_size = len(batch['xs'])

        if a2c_oracle:
            if dataset.is_test:
                max_label_num = 0
                for b in range(batch_size):
                    if max_label_num < len(list(batch['ys_sub'][b][0])):
                        max_label_num = len(
                            list(batch['ys_sub'][b][0]))

                ys_sub = np.zeros(
                    (batch_size, max_label_num), dtype=np.int32)
                ys_sub -= 1  # pad with -1
                y_lens_sub = np.zeros((batch_size,), dtype=np.int32)
                for b in range(batch_size):
                    indices = dataset.char2idx(batch['ys_sub'][b][0])
                    ys_sub[b, :len(indices)] = indices
                    y_lens_sub[b] = len(indices)
                    # NOTE: transcript is seperated by space('_')
            else:
                ys_sub = batch['ys_sub']
                y_lens_sub = batch['y_lens_sub']
        else:
            ys_sub = None
            y_lens_sub = None

        best_hyps, aw, best_hyps_sub, aw_sub, aw_dec, _ = model.decode(
            batch['xs'], batch['x_lens'],
            beam_width=args.beam_width,
            max_decode_len=MAX_DECODE_LEN_WORD,
            min_decode_len=MIN_DECODE_LEN_WORD,
            beam_width_sub=args.beam_width_sub,
            max_decode_len_sub=MAX_DECODE_LEN_CHAR,
            min_decode_len_sub=MIN_DECODE_LEN_CHAR,
            length_penalty=args.length_penalty,
            coverage_penalty=args.coverage_penalty,
            teacher_forcing=a2c_oracle,
            ys_sub=ys_sub,
            y_lens_sub=y_lens_sub)

        for b in range(batch_size):
            word_list = dataset.idx2word(best_hyps[b])
            if dataset.label_type_sub == 'word':
                char_list = dataset.idx2word(
                    best_hyps_sub[b], return_list=True)
            else:
                char_list = dataset.idx2char(
                    best_hyps_sub[b], return_list=True)

            # word to acoustic & character to acoustic
            plot_hierarchical_attention_weights(
                aw[b][:len(word_list), :batch['x_lens'][b]],
                aw_sub[b][:len(char_list), :batch['x_lens'][b]],
                label_list=word_list,
                label_list_sub=char_list,
                spectrogram=batch['xs'][b, :, :dataset.input_freq],
                save_path=mkdir_join(
                    save_path, batch['input_names'][b] + '.png'),
                figsize=(40, 8)
            )

            # word to characater
            plot_word2char_attention_weights(
                aw_dec[b][:len(word_list), :len(char_list)],
                label_list=word_list,
                label_list_sub=char_list,
                save_path=mkdir_join(
                    save_path, batch['input_names'][b] + '_word2char.png'),
                figsize=(40, 8)
            )

            # with open(join(save_path, speaker, batch['input_names'][b] + '.txt'), 'w') as f:
            #     f.write(batch['ys'][b][0])

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
