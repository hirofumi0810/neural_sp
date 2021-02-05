# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Args option for the LM task."""

import configargparse
from distutils.util import strtobool
import logging
from omegaconf import OmegaConf
import os

from neural_sp.bin.train_utils import load_config

logger = logging.getLogger(__name__)


def parse_args_train(input_args):
    parser = build_parser()
    user_args, _ = parser.parse_known_args(input_args)

    config = OmegaConf.load(user_args.config)

    # register module specific arguments
    parser = register_args_lm(parser, user_args, user_args.lm_type)
    user_args = parser.parse_args()

    # merge to omegaconf
    for k, v in vars(user_args).items():
        if k not in config:
            config[k] = v

    return config


def parse_args_eval(input_args):
    parser = build_parser()
    user_args, _ = parser.parse_known_args(input_args)

    # Load a yaml config file
    dir_name = os.path.dirname(user_args.recog_model[0])
    config = load_config(os.path.join(dir_name, 'conf.yml'))

    # register module specific arguments to support new args after training
    parser = register_args_lm(parser, user_args, config.lm_type)
    user_args = parser.parse_args()

    # Overwrite to omegaconf
    for k, v in vars(user_args).items():
        if 'recog' in k or k not in config:
            config[k] = v
            logger.info('Overwrite configration: %s => %s' % (k, v))

    return config, dir_name


def register_args_lm(parser, args, lm_type):
    if 'gated_conv' in lm_type:
        from neural_sp.models.lm.gated_convlm import GatedConvLM as module
    elif lm_type == 'transformer':
        from neural_sp.models.lm.transformerlm import TransformerLM as module
    elif lm_type == 'transformer_xl':
        from neural_sp.models.lm.transformer_xl import TransformerXL as module
    else:
        from neural_sp.models.lm.rnnlm import RNNLM as module
    if hasattr(module, 'add_args'):
        parser = module.add_args(parser, args)
    return parser


def build_parser():
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add('--config', is_config_file=True, help='config file path')
    # general
    parser.add_argument('--corpus', type=str,
                        help='corpus name')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='number of GPUs (0 indicates CPU)')
    parser.add_argument('--cudnn_benchmark', type=strtobool, default=True,
                        help='use CuDNN benchmark mode')
    parser.add_argument("--train_dtype", default="float32",
                        choices=["float16", "float32", "float64", "O0", "O1", "O2", "O3"],
                        help="Data type for training")
    parser.add_argument('--model_save_dir', type=str, default=False,
                        help='directory to save a model')
    parser.add_argument('--resume', type=str, default=False, nargs='?',
                        help='model path to resume training')
    parser.add_argument('--job_name', type=str, default=False,
                        help='job name')
    parser.add_argument('--stdout', type=strtobool, default=False,
                        help='print to standard output')
    parser.add_argument('--remove_old_checkpoints', type=strtobool, default=True,
                        help='remove old checkpoints to save disk (turned off when training Transformer')
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
                                 'transformer', 'transformer_xl'],
                        help='type of language model')
    parser.add_argument('--n_layers', type=int, default=5,
                        help='number of layers')
    parser.add_argument('--emb_dim', type=int, default=1024,
                        help='number of dimensions in the embedding layer')
    parser.add_argument('--n_units_null_context', type=int, default=0, nargs='?',
                        help='')
    parser.add_argument('--tie_embedding', type=strtobool, default=False, nargs='?',
                        help='tie input and output embedding')
    # optimization
    parser.add_argument('--batch_size', type=int, default=256,
                        help='mini-batch size')
    parser.add_argument('--bptt', type=int, default=200,
                        help='BPTT length')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adadelta', 'adagrad', 'sgd', 'momentum', 'nesterov', 'noam'],
                        help='type of optimizer')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='number of epochs to train the model')
    parser.add_argument('--convert_to_sgd_epoch', type=int, default=100,
                        help='epoch to convert to SGD fine-tuning')
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
                        help='number of epochs to tolerate learning rate decay when validation performance is not improved')
    parser.add_argument('--early_stop_patient_n_epochs', type=int, default=5,
                        help='number of epochs to tolerate stopping training when validation performance is not improved')
    parser.add_argument('--sort_stop_epoch', type=int, default=10000,
                        help='epoch to stop soring utterances by length')
    parser.add_argument('--eval_start_epoch', type=int, default=1,
                        help='first epoch to start evaluation')
    parser.add_argument('--warmup_start_lr', type=float, default=0,
                        help='initial learning rate for learning rate warm up')
    parser.add_argument('--warmup_n_steps', type=int, default=0,
                        help='number of steps to warm up learning rate')
    parser.add_argument('--accum_grad_n_steps', type=int, default=1,
                        help='total number of steps to accumulate gradients')
    # initialization
    parser.add_argument('--param_init', type=float, default=0.1,
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
    # contextualization
    parser.add_argument('--shuffle', type=strtobool, default=False, nargs='?',
                        help='shuffle utterances per epoch')
    parser.add_argument('--serialize', type=strtobool, default=False, nargs='?',
                        help='serialize text according to onset in dialogue')
    # evaluation parameters
    parser.add_argument('--recog_stdout', type=strtobool, default=False,
                        help='print to standard output during evaluation')
    parser.add_argument('--recog_n_gpus', type=int, default=0,
                        help='number of GPUs (0 indicates CPU)')
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
    parser.add_argument('--recog_n_caches', type=int, default=0,
                        help='number of tokens for cache')
    parser.add_argument('--recog_cache_theta', type=float, default=0.2,
                        help='theta parameter for cache')
    parser.add_argument('--recog_cache_lambda', type=float, default=0.2,
                        help='lambda parameter for cache')
    parser.add_argument('--recog_mem_len', type=int, default=0,
                        help='number of tokens for memory in TransformerXL during evaluation')
    return parser
