#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Args option for the LM task."""

import configargparse
from distutils.util import strtobool


def parse():
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add('--config', is_config_file=True, help='config file path')
    # general
    parser.add_argument('--corpus', type=str,
                        help='corpus name')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='number of GPUs (0 indicates CPU)')
    parser.add_argument('--model_save_dir', type=str, default=False,
                        help='directory to save a model')
    parser.add_argument('--resume', type=str, default=False, nargs='?',
                        help='model path to resume training')
    parser.add_argument('--job_name', type=str, default=False,
                        help='job name')
    parser.add_argument('--stdout', type=strtobool, default=False,
                        help='print to standard output')
    parser.add_argument('--recog_stdout', type=strtobool, default=False,
                        help='print to standard output during evaluation')
    # dataset
    parser.add_argument('--train_set', type=str,
                        help='tsv file path for the training set')
    parser.add_argument('--dev_set', type=str,
                        help='tsv file path for the development set')
    parser.add_argument('--eval_sets', type=str, default=[], nargs='+',
                        help='tsv file paths for the evaluation sets')
    parser.add_argument('--nlsyms', type=str, default=False, nargs='?',
                        help='non-linguistic symbols file path')
    parser.add_argument('--dict', type=str,
                        help='dictionary file path')
    parser.add_argument('--unit', type=str, default='word',
                        choices=['word', 'wp', 'char', 'word_char'],
                        help='output unit')
    parser.add_argument('--wp_model', type=str, default=False, nargs='?',
                        help='wordpiece model path')
    # features
    parser.add_argument('--min_n_tokens', type=int, default=1,
                        help='minimum number of input tokens')
    parser.add_argument('--dynamic_batching', type=strtobool, default=False,
                        help='')
    # topology
    parser.add_argument('--lm_type', type=str, default='lstm',
                        choices=['lstm', 'gru', 'gated_conv_custom',
                                 'gated_conv_8', 'gated_conv_8B', 'gated_conv_9',
                                 'gated_conv_13', 'gated_conv_14', 'gated_conv_14B',
                                 'transformer'],
                        help='type of language model')
    parser.add_argument('--kernel_size', type=int, default=4,
                        help='kernel size for GatedConvLM')
    parser.add_argument('--n_units', type=int, default=1024,
                        help='number of units in each layer')
    parser.add_argument('--n_projs', type=int, default=0,
                        help='number of units in the projection layer')
    parser.add_argument('--n_layers', type=int, default=5,
                        help='number of layers')
    parser.add_argument('--emb_dim', type=int, default=1024,
                        help='number of dimensions in the embedding layer')
    parser.add_argument('--n_units_null_context', type=int, default=0, nargs='?',
                        help='')
    parser.add_argument('--tie_embedding', type=strtobool, default=False, nargs='?',
                        help='tie input and output embedding')
    parser.add_argument('--residual', type=strtobool, default=False, nargs='?',
                        help='')
    parser.add_argument('--use_glu', type=strtobool, default=False, nargs='?',
                        help='use Gated Linear Unit (GLU) for fully-connected layers')
    # optimization
    parser.add_argument('--batch_size', type=int, default=256,
                        help='mini-batch size')
    parser.add_argument('--bptt', type=int, default=100,
                        help='BPTT length')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adadelta', 'adagrad', 'sgd', 'momentum', 'nesterov', 'noam'],
                        help='type of optimizer')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='number of epochs to train the model')
    parser.add_argument('--convert_to_sgd_epoch', type=int, default=100,
                        help='epoch to converto to SGD fine-tuning')
    parser.add_argument('--print_step', type=int, default=100,
                        help='print log per this value')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--lr_factor', type=float, default=10.0,
                        help='factor of learning rate for Transformer')
    parser.add_argument('--eps', type=float, default=1e-6,
                        help='epsilon parameter for Adadelta optimizer')
    parser.add_argument('--lr_decay_type', type=str, default='always',
                        choices=['always', 'metric', 'warmup'],
                        help='type of learning rate decay')
    parser.add_argument('--lr_decay_start_epoch', type=int, default=10,
                        help='epoch to start to decay learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9,
                        help='decay rate of learning rate')
    parser.add_argument('--lr_decay_patient_n_epochs', type=int, default=0,
                        help='number of epochs to tolerate learning rate decay when validation perfomance is not improved')
    parser.add_argument('--early_stop_patient_n_epochs', type=int, default=5,
                        help='number of epochs to tolerate stopping training when validation perfomance is not improved')
    parser.add_argument('--sort_stop_epoch', type=int, default=10000,
                        help='epoch to stop soring utterances by length')
    parser.add_argument('--eval_start_epoch', type=int, default=1,
                        help='first epoch to start evalaution')
    parser.add_argument('--warmup_start_lr', type=float, default=0,
                        help='initial learning rate for learning rate warm up')
    parser.add_argument('--warmup_n_steps', type=int, default=0,
                        help='number of steps to warm up learing rate')
    parser.add_argument('--accum_grad_n_steps', type=int, default=1,
                        help='total number of steps to accumulate gradients')
    # initialization
    parser.add_argument('--param_init', type=float, default=0.1,
                        help='')
    parser.add_argument('--rec_weight_orthogonal', type=strtobool, default=False,
                        help='')
    parser.add_argument('--pretrained_model', type=str, default=False, nargs='?',
                        help='')
    # regularization
    parser.add_argument('--clip_grad_norm', type=float, default=5.0,
                        help='')
    parser.add_argument('--dropout_in', type=float, default=0.0,
                        help='dropout probability for the input embedding layer')
    parser.add_argument('--dropout_hidden', type=float, default=0.0,
                        help='dropout probability for the hidden layers')
    parser.add_argument('--dropout_out', type=float, default=0.0,
                        help='dropout probability for the output layer')
    parser.add_argument('--dropout_att', type=float, default=0.1,
                        help='dropout probability for the attention weights (for Transformer)')
    parser.add_argument('--dropout_residual', type=float, default=0.0,
                        help='dropout probability for the stochasitc residual connections (for Transformer)')
    parser.add_argument('--dropout_head', type=float, default=0.0,
                        help='dropout probability for MHA in the decoder (for Transformer)')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='')
    parser.add_argument('--lsm_prob', type=float, default=0.0,
                        help='probability of label smoothing')
    parser.add_argument('--logits_temp', type=float, default=1.0,
                        help='')
    parser.add_argument('--backward', type=strtobool, default=False, nargs='?',
                        help='')
    parser.add_argument('--adaptive_softmax', type=strtobool, default=False,
                        help='use adaptive softmax')
    # transformer
    parser.add_argument('--transformer_d_model', type=int, default=256,
                        help='number of units in self-attention layers in Transformer')
    parser.add_argument('--transformer_d_ff', type=int, default=2048,
                        help='number of units in feed-forward fully-conncected layers in Transformer')
    parser.add_argument('--transformer_attn_type', type=str, default='scaled_dot',
                        choices=['scaled_dot', 'add', 'average'],
                        help='type of attention for Transformer')
    parser.add_argument('--transformer_n_heads', type=int, default=4,
                        help='number of heads in the attention layer for Transformer')
    parser.add_argument('--transformer_pe_type', type=str, default='add',
                        choices=['add', 'concat', 'none', '1dconv3L'],
                        help='type of positional encoding')
    parser.add_argument('--transformer_layer_norm_eps', type=float, default=1e-12,
                        help='epsilon value for layer narmalization')
    parser.add_argument('--transformer_ffn_activation', type=str, default='relu',
                        choices=['relu', 'gelu', 'gelu_accurate', 'glu'],
                        help='nonlinear activation for position wise feed-forward layer')
    parser.add_argument('--transformer_param_init', type=str, default='xavier_uniform',
                        choices=['xavier_uniform', 'pytorch'],
                        help='parameter initializatin for Transformer')
    # contextualization
    parser.add_argument('--shuffle', type=strtobool, default=False, nargs='?',
                        help='shuffle utterances per epoch')
    parser.add_argument('--serialize', type=strtobool, default=False, nargs='?',
                        help='serialize text according to onset in dialogue')
    # evaluation parameters
    parser.add_argument('--recog_sets', type=str, default=[], nargs='+',
                        help='tsv file paths for the evaluation sets')
    parser.add_argument('--recog_model', type=str, default=False, nargs='+',
                        help='model path')
    parser.add_argument('--recog_dir', type=str, default=False,
                        help='directory to save decoding results')
    parser.add_argument('--recog_batch_size', type=int, default=1,
                        help='size of mini-batch in evaluation')
    parser.add_argument('--recog_n_average', type=int, default=5,
                        help='number of models for the model averaging of Transformer')
    # cache parameters
    parser.add_argument('--recog_n_caches', type=int, default=0,
                        help='number of tokens for cache')
    parser.add_argument('--recog_cache_theta', type=float, default=0.2,
                        help='theta paramter for cache')
    parser.add_argument('--recog_cache_lambda', type=float, default=0.2,
                        help='lambda paramter for cache')

    args = parser.parse_args()
    # args, _ = parser.parse_known_args(parser)
    return args
