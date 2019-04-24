#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Args option for the LM task."""

import argparse
from distutils.util import strtobool


def parse():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--corpus', type=str,
                        help='name of corpus')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='number of GPUs (0 indicates CPU)')
    parser.add_argument('--model', type=str, default=False,
                        help='directory to save a model')
    parser.add_argument('--resume', type=str, default=False, nargs='?',
                        help='path to the model to resume training')
    parser.add_argument('--job_name', type=str, default='',
                        help='name of job')
    # dataset
    parser.add_argument('--train_set', type=str,
                        help='path to a tsv file for the training set')
    parser.add_argument('--dev_set', type=str,
                        help='path to a tsv file for the development set')
    parser.add_argument('--eval_sets', type=str, default=[], nargs='+',
                        help='path to tsv files for the evaluation sets')
    parser.add_argument('--nlsyms', type=str, nargs='?',
                        help='path to a non-linguistic symbols file')
    parser.add_argument('--dict', type=str,
                        help='path to a dictionary file')
    parser.add_argument('--unit', type=str, default='word',
                        choices=['word', 'wp', 'char', 'word_char'],
                        help='Output unit')
    parser.add_argument('--wp_model', type=str, default=False, nargs='?',
                        help='path to of the wordpiece model')
    # features
    parser.add_argument('--min_n_tokens', type=int, default=1,
                        help='minimum number of input tokens')
    parser.add_argument('--dynamic_batching', type=strtobool, default=False,
                        help='')
    # topology
    parser.add_argument('--lm_type', type=str, default='lstm',
                        choices=['lstm', 'gru', 'gated_cnn'],
                        help='type of language model')
    parser.add_argument('--n_units', type=int, default=512,
                        help='number of units in each layer')
    parser.add_argument('--n_projs', type=int, default=0,
                        help='number of units in the projection layer')
    parser.add_argument('--n_layers', type=int, default=5,
                        help='number of layers')
    parser.add_argument('--emb_dim', type=int, default=5,
                        help='number of dimensions in the embedding layer')
    parser.add_argument('--tie_embedding', type=strtobool, default=False, nargs='?',
                        help='Tie input and output embedding')
    parser.add_argument('--residual', type=strtobool, default=False, nargs='?',
                        help='')
    parser.add_argument('--use_glu', type=strtobool, default=False, nargs='?',
                        help='Use Gated Linear Unit (GLU) for fully-connected layers')
    # optimization
    parser.add_argument('--batch_size', type=int, default=256,
                        help='')
    parser.add_argument('--bptt', type=int, default=100,
                        help='')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adadelta', 'sgd', 'momentum', 'nesterov'],
                        help='')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='')
    parser.add_argument('--eps', type=float, default=1e-6,
                        help='')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--convert_to_sgd_epoch', type=int, default=20,
                        help='')
    parser.add_argument('--print_step', type=int, default=100,
                        help='step to print log')
    parser.add_argument('--decay_type', type=str, default='epoch',
                        choices=['epoch', 'metric'],
                        help='')
    parser.add_argument('--decay_start_epoch', type=int, default=10,
                        help='')
    parser.add_argument('--decay_rate', type=float, default=0.9,
                        help='')
    parser.add_argument('--decay_patient_n_epochs', type=int, default=0,
                        help='')
    parser.add_argument('--sort_stop_epoch', type=int, default=10000,
                        help='')
    parser.add_argument('--not_improved_patient_n_epochs', type=int, default=5,
                        help='')
    parser.add_argument('--eval_start_epoch', type=int, default=1,
                        help='')
    # initialization
    parser.add_argument('--param_init', type=float, default=0.1,
                        help='')
    parser.add_argument('--param_init_dist', type=str, default='uniform',
                        choices=['uniform', 'he', 'glorot', 'lecun',
                                 'xavier_uniform', 'xavier_normal',
                                 'kaiming_uniform', 'kaiming_normal'],
                        help='')
    parser.add_argument('--rec_weight_orthogonal', type=strtobool, default=False,
                        help='')
    parser.add_argument('--pretrained_model', default=False, nargs='?',
                        help='')
    # regularization
    parser.add_argument('--clip_grad_norm', type=float, default=5.0,
                        help='')
    parser.add_argument('--dropout_hidden', type=float, default=0.0,
                        help='dropout probability for the hidden layers')
    parser.add_argument('--dropout_out', type=float, default=0.0,
                        help='dropout probability for the output layer')
    parser.add_argument('--dropout_emb', type=float, default=0.0,
                        help='dropout probability for the embedding layer')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='')
    parser.add_argument('--logits_temp', type=float, default=1.0,
                        help='')
    parser.add_argument('--backward', type=strtobool, default=False, nargs='?',
                        help='')
    parser.add_argument('--adaptive_softmax', type=strtobool, default=False,
                        help='use adaptive softmax')
    # contextualization
    parser.add_argument('--serialize', type=strtobool, default=False, nargs='?',
                        help='serialize text according to onset in dialogue')
    # evaluation
    parser.add_argument('--recog_sets', type=str, default=[], nargs='+',
                        help='path to tsv files for the evaluation sets')
    parser.add_argument('--recog_model', type=str, default=None, nargs='+',
                        help='path to the model')
    parser.add_argument('--recog_dir', type=str, default=None,
                        help='directory to save decoding results')
    parser.add_argument('--recog_batch_size', type=int, default=1,
                        help='size of mini-batch in evaluation')
    # cache
    parser.add_argument('--recog_n_caches', type=int, default=0,
                        help='number of tokens for cache')
    parser.add_argument('--recog_cache_theta', type=float, default=0.2,
                        help='theta paramter for cache')
    parser.add_argument('--recog_cache_lambda', type=float, default=0.2,
                        help='lambda paramter for cache')
    args = parser.parse_args()
    return args
