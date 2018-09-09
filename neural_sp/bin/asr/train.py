#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"Set options for ASR training."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cProfile
import os
import warnings

from neural_sp.bin.asr.train_main import train
from neural_sp.utils.config import load_config


parser = argparse.ArgumentParser()
# general
parser.add_argument('--ngpus', type=int, default=0,
                    help='number of GPUs (0 indicates CPU)')
parser.add_argument('--config', type=str, default=None,
                    help='path to a yaml file for configuration')
parser.add_argument('--model', type=str, default=None,
                    help='directory to save a model')
parser.add_argument('--saved_model', type=str, default=None,
                    help='saved model to restart training')
# dataset
parser.add_argument('--corpus', type=str,
                    help='the name of corpus')
parser.add_argument('--train_set', type=str,
                    help='path to a csv file for the training set')
parser.add_argument('--train_set_sub', type=str, default=None,
                    help='path to a csv file for the training set for th sub tsk')
parser.add_argument('--dev_set', type=str,
                    help='path to a csv file for the development set')
parser.add_argument('--dev_set_sub', type=str, default=None,
                    help='path to a csv file for the development set for the sub task')
parser.add_argument('--eval_sets', type=str, default=[], nargs='+',
                    help='path to csv files for the evaluation sets')
parser.add_argument('--dict', type=str,
                    help='path to a dictionary file')
parser.add_argument('--dict_sub', type=str, default=None,
                    help='path to a dictionary file for the sub task')
parser.add_argument('--label_type', type=str, default='word',
                    choices=['word', 'bpe', 'char', 'phone'],
                    help='')
parser.add_argument('--label_type_sub', type=str, default='char',
                    choices=['bpe', 'char', 'phone'],
                    help='')
# features
parser.add_argument('--input_type', type=str, choices=['speech', 'text'],
                    help='')
parser.add_argument('--num_splice', type=int, default=1,
                    help='')
parser.add_argument('--num_stack', type=int, default=1,
                    help='')
parser.add_argument('--num_skip', type=int, default=1,
                    help='')
parser.add_argument('--max_num_frames', type=int, default=2000,
                    help='')
parser.add_argument('--min_num_frames', type=int, default=40,
                    help='')
parser.add_argument('--dynamic_batching', type=bool, default=False,
                    help='')
# topology (encoder)
parser.add_argument('--conv_in_channel', type=int, default=1,
                    help='')
parser.add_argument('--conv_channels', type=list, default=[],
                    help='')
parser.add_argument('--conv_kernel_sizes', type=list, default=[],
                    help='')
parser.add_argument('--conv_strides', type=list, default=[],
                    help='')
parser.add_argument('--conv_poolings', type=list, default=[],
                    help='')
parser.add_argument('--conv_batch_norm', type=bool, default=False,
                    help='')
parser.add_argument('--enc_type', type=str, default='blstm',
                    help='')
parser.add_argument('--enc_num_units', type=int, default=320,
                    help='')
parser.add_argument('--enc_num_projs', type=int, default=0,
                    help='')
parser.add_argument('--enc_num_layers', type=int, default=5,
                    help='')
parser.add_argument('--enc_num_layers_sub', type=int, default=0,
                    help='')
parser.add_argument('--enc_residual', type=bool, default=False,
                    help='')
parser.add_argument('--subsample', type=list, default=[False] * 5,
                    help='')
parser.add_argument('--subsample_type', type=str, default='drop',
                    choices=['drop', 'concat'],
                    help='')
# topology (decoder)
parser.add_argument('--att_type', type=str, default='location',
                    help='')
parser.add_argument('--att_dim', type=int, default=128,
                    help='')
parser.add_argument('--att_conv_num_channels', type=int, default=10,
                    help='')
parser.add_argument('--att_conv_width', type=int, default=201,
                    help='')
parser.add_argument('--att_num_heads', type=int, default=1,
                    help='')
parser.add_argument('--att_num_heads_sub', type=int, default=1,
                    help='')
parser.add_argument('--att_sharpening_factor', type=int, default=1,
                    help='')
parser.add_argument('--att_sigmoid_smoothing', type=bool, default=False,
                    help='')
parser.add_argument('--bridge_layer', type=bool, default=False,
                    help='')
parser.add_argument('--dec_type', type=str, default='lstm',
                    choices=['lstm' 'gru'],
                    help='')
parser.add_argument('--dec_num_units', type=int, default=320,
                    help='')
parser.add_argument('--dec_num_projs', type=int, default=0,
                    help='')
parser.add_argument('--dec_num_layers', type=int, default=1,
                    help='')
parser.add_argument('--dec_num_layers_sub', type=int, default=1,
                    help='')
parser.add_argument('--dec_residual', type=bool, default=False,
                    help='')
parser.add_argument('--emb_dim', type=int, default=320,
                    help='')
parser.add_argument('--ctc_fc_list', type=list, default=[],
                    help='')
# optimization
parser.add_argument('--batch_size', type=int, default=50,
                    help='')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='')
parser.add_argument('--num_epochs', type=int, default=25,
                    help='')
parser.add_argument('--convert_to_sgd_epoch', type=int, default=20,
                    help='')
parser.add_argument('--print_step', type=int, default=100,
                    help='')
# initialization
parser.add_argument('--param_init', type=float, default=0.1,
                    help='')
parser.add_argument('--param_init_dist', type=str, default='uniform',
                    help='')
parser.add_argument('--rec_weight_orthogonal', type=bool, default=False,
                    help='')
parser.add_argument('--pretrained_model', default=False,
                    help='')
# regularization
parser.add_argument('--clip_grad_norm', type=float, default=5.0,
                    help='')
parser.add_argument('--clip_activation_enc', type=float, default=50.0,
                    help='')
parser.add_argument('--clip_activation_dec', type=float, default=50.0,
                    help='')
parser.add_argument('--dropout_in', type=float, default=0.0,
                    help='')
parser.add_argument('--dropout_enc', type=float, default=0.0,
                    help='')
parser.add_argument('--dropout_dec', type=float, default=0.0,
                    help='')
parser.add_argument('--dropout_emb', type=float, default=0.0,
                    help='')
parser.add_argument('--weight_decay', type=float, default=1e-6,
                    help='')
parser.add_argument('--logits_temp', type=float, default=1.0,
                    help='')
parser.add_argument('--ss_prob', type=float, default=0.0,
                    help='')
parser.add_argument('--lsm_prob', type=float, default=0.0,
                    help='')
parser.add_argument('--lsm_type', type=str, default='uniform',
                    help='')
# annealing
parser.add_argument('--decay_type', type=str, default='per_epoch',
                    choices=['per_epoch'],
                    help='')
parser.add_argument('--decay_start_epoch', type=int, default=10,
                    help='')
parser.add_argument('--decay_rate', type=float, default=0.9,
                    help='')
parser.add_argument('--decay_patient_epoch', type=int, default=0,
                    help='')
parser.add_argument('--sort_stop_epoch', type=int, default=100,
                    help='')
parser.add_argument('--not_improved_patient_epoch', type=int, default=5,
                    help='')
parser.add_argument('--eval_start_epoch', type=int, default=1,
                    help='')
# MTL
parser.add_argument('--ctc_weight', type=float, default=0.0,
                    help='')
parser.add_argument('--ctc_weight_sub', type=float, default=0.0,
                    help='')
parser.add_argument('--bwd_weight', type=float, default=0.0,
                    help='')
parser.add_argument('--bwd_weight_sub', type=float, default=0.0,
                    help='')
parser.add_argument('--main_task_weight', type=float, default=1.0,
                    help='')
# cold fusion
parser.add_argument('--cold_fusion_type', type=str, default='prob',
                    choices=['hidden', 'prob'],
                    help='')
parser.add_argument('--rnnlm_cf', type=str, default=False,
                    help='')
# RNNLM initialization & RNNLM objective
parser.add_argument('--internal_lm', type=bool, default=False,
                    help='')
parser.add_argument('--rnnlm_init', type=str, default=False,
                    help='')
parser.add_argument('--rnnlm_weight', type=float, default=0.0,
                    help='')
parser.add_argument('--share_softmax', type=bool, default=False,
                    help='')

args = parser.parse_args()


def main():

    # Load a config file
    if args.saved_model is None:
        config = load_config(args.config)
    else:
        # Restart from the last checkpoint
        config = load_config(os.path.join(args.saved_model, 'config.yml'))

    # Check differences between args and yaml comfiguraiton
    for k, v in vars(args).items():
        if k not in config.keys():
            warnings.warn("key %s is automatically set to %s" % (k, str(v)))

    # Merge config with args
    for k, v in config.items():
        setattr(args, k, v)

    return train(args)


if __name__ == '__main__':
    # Setting for profiling
    pr = cProfile.Profile()
    save_path = pr.runcall(main)
    pr.dump_stats(os.path.join(save_path, 'train.profile'))
