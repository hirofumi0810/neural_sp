#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot attention weights of the hierarchical attention model (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isdir
import sys
import argparse
import shutil
from distutils.util import strtobool

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.csj.s5.exp.dataset.load_dataset_hierarchical import Dataset
from utils.directory import mkdir_join, mkdir
from utils.visualization.attention import plot_hierarchical_attention_weights
from utils.config import load_config
from utils.io.labels.word import Word2char

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
                    help='length penalty in the beam search decoding')
parser.add_argument('--coverage_penalty', type=float, default=0,
                    help='coverage penalty in the beam search decoding')
parser.add_argument('--rnnlm_weight', type=float, default=0,
                    help='the weight of RNNLM score of the main task in the beam search decoding')
parser.add_argument('--rnnlm_weight_sub', type=float, default=0,
                    help='the weight of RNNLM score of the sub task in the beam search decoding')
parser.add_argument('--rnnlm_path', default=None, type=str, nargs='?',
                    help='path to the RMMLM of the main task')
parser.add_argument('--rnnlm_path_sub', default=None, type=str, nargs='?',
                    help='path to the RMMLM of the sub task')
parser.add_argument('--joint_decoding', type=strtobool, default=False)
parser.add_argument('--score_sub_weight', type=float, default=0)

MAX_DECODE_LEN_WORD = 100
MIN_DECODE_LEN_WORD = 1
MAX_DECODE_LEN_CHAR = 200
MIN_DECODE_LEN_CHAR = 1


def main():

    args = parser.parse_args()

    # Load a ASR config file
    config = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    dataset = Dataset(data_save_path=args.data_save_path,
                      input_freq=config['input_freq'],
                      use_delta=config['use_delta'],
                      use_double_delta=config['use_double_delta'],
                      data_type='eval1',
                      # data_type='eval2',
                      # data_type='eval3',
                      data_size=config['data_size'],
                      label_type=config['label_type'],
                      label_type_sub=config['label_type_sub'],
                      batch_size=args.eval_batch_size,
                      sort_utt=False, reverse=False, tool=config['tool'])
    config['num_classes'] = dataset.num_classes
    config['num_classes_sub'] = dataset.num_classes_sub

    # For cold fusion
    if config['rnnlm_fusion_type'] and config['rnnlm_path']:
        # Load a RNNLM config file
        rnnlm_config = load_config(join(args.model_path, 'config_rnnlm.yml'))

        assert config['label_type'] == rnnlm_config['label_type']
        rnnlm_config['num_classes'] = dataset.num_classes
        config['rnnlm_config'] = rnnlm_config
    else:
        config['rnnlm_config'] = None

    if config['rnnlm_fusion_type'] and config['rnnlm_path_sub']:
        # Load a RNNLM config file
        rnnlm_config_sub = load_config(
            join(args.model_path, 'config_rnnlm_sub.yml'))

        assert config['label_type_sub'] == rnnlm_config_sub['label_type']
        rnnlm_config_sub['num_classes'] = dataset.num_classes_sub
        config['rnnlm_config_sub'] = rnnlm_config_sub
    else:
        config['rnnlm_config_sub'] = None

    # Load the ASR model
    model = load(model_type=config['model_type'],
                 config=config,
                 backend=config['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # For shallow fusion
    if not (config['rnnlm_fusion_type'] and config['rnnlm_path']) and args.rnnlm_path is not None and args.rnnlm_weight > 0:
        # Load a RNNLM config file
        config_rnnlm = load_config(
            join(args.rnnlm_path, 'config.yml'), is_eval=True)

        assert config['label_type'] == config_rnnlm['label_type']
        config_rnnlm['num_classes'] = dataset.num_classes

        # Load the pre-trianed RNNLM
        rnnlm = load(model_type=config_rnnlm['model_type'],
                     config=config_rnnlm,
                     backend=config_rnnlm['backend'])
        rnnlm.load_checkpoint(save_path=args.rnnlm_path, epoch=-1)
        rnnlm.rnn.flatten_parameters()
        model.rnnlm_0 = rnnlm

    if not (config['rnnlm_fusion_type'] and config['rnnlm_path_sub']) and args.rnnlm_path_sub is not None and args.rnnlm_weight_sub > 0:
        # Load a RNNLM config file
        config_rnnlm_sub = load_config(
            join(args.rnnlm_path_sub, 'config.yml'), is_eval=True)

        assert config['label_type_sub'] == config_rnnlm_sub['label_type']
        config_rnnlm_sub['num_classes'] = dataset.num_classes_sub

        # Load the pre-trianed RNNLM
        rnnlm_sub = load(model_type=config_rnnlm_sub['model_type'],
                         config=config_rnnlm_sub,
                         backend=config_rnnlm_sub['backend'])
        rnnlm_sub.load_checkpoint(save_path=args.rnnlm_path_sub, epoch=-1)
        rnnlm_sub.rnn.flatten_parameters()
        model.rnnlm_1 = rnnlm_sub

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    save_path = mkdir_join(args.model_path, 'att_weights')

    ######################################################################

    word2char = Word2char(dataset.vocab_file_path,
                          dataset.vocab_file_path_sub)

    # Clean directory
    if save_path is not None and isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    for batch, is_new_epoch in dataset:
        # Decode
        if model.model_type == 'hierarchical_attention' and args.joint_decoding is not None:
            best_hyps, aw, best_hyps_sub, aw_sub, _ = model.decode(
                batch['xs'],
                beam_width=args.beam_width,
                max_decode_len=MAX_DECODE_LEN_WORD,
                min_decode_len=MIN_DECODE_LEN_WORD,
                length_penalty=args.length_penalty,
                coverage_penalty=args.coverage_penalty,
                rnnlm_weight=args.rnnlm_weight,
                joint_decoding=args.joint_decoding,
                space_index=dataset.char2idx('_')[0],
                oov_index=dataset.word2idx('OOV')[0],
                word2char=word2char,
                idx2word=dataset.idx2word,
                idx2char=dataset.idx2char,
                score_sub_weight=args.score_sub_weight)
        else:
            best_hyps, aw, perm_idx = model.decode(
                batch['xs'],
                beam_width=args.beam_width,
                max_decode_len=MAX_DECODE_LEN_WORD,
                min_decode_len=MIN_DECODE_LEN_WORD,
                length_penalty=args.length_penalty,
                coverage_penalty=args.coverage_penalty,
                rnnlm_weight=args.rnnlm_weight)
            best_hyps_sub, aw_sub, _ = model.decode(
                batch['xs'],
                beam_width=args.beam_width_sub,
                max_decode_len=MAX_DECODE_LEN_CHAR,
                min_decode_len=MIN_DECODE_LEN_CHAR,
                length_penalty=args.length_penalty,
                coverage_penalty=args.coverage_penalty,
                rnnlm_weight=args.rnnlm_weight_sub,
                task_index=1)

        for b in range(len(batch['xs'])):
            word_list = dataset.idx2word(best_hyps[b], return_list=True)
            char_list = dataset.idx2char(best_hyps_sub[b], return_list=True)

            speaker = batch['input_names'][b].split('_')[0]

            plot_hierarchical_attention_weights(
                aw[b][:len(word_list), :batch['x_lens'][b]],  # TODO: fix this
                aw_sub[b][:len(char_list), :batch['x_lens'][b]],
                label_list=word_list,
                label_list_sub=char_list,
                spectrogram=batch['xs'][b, :, :dataset.input_freq],
                save_path=mkdir_join(save_path, speaker,
                                     batch['input_names'][b] + '.png'),
                figsize=(40, 8)
            )

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
