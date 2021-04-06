# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Common args options."""

from distutils.util import strtobool


def add_args_common(parser):
    # general
    parser.add('--config', is_config_file=True, help='config file path')
    parser.add_argument('--corpus', type=str,
                        help='corpus name')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='number of GPUs (0 indicates CPU)')
    parser.add_argument('--cudnn_benchmark', type=strtobool, default=True,
                        help='use CuDNN benchmark mode')
    parser.add_argument('--cudnn_deterministic', type=strtobool, default=False,
                        help='use CuDNN deterministic mode')
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
                        help='print to standard output during training')
    parser.add_argument('--remove_old_checkpoints', type=strtobool, default=True,
                        help='remove old checkpoints to save disk (turned off when training Transformer')
    parser.add_argument('--use_wandb', type=strtobool, default=False,
                        help='use wandb for reporting')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for torch')
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
                        choices=['word', 'wp', 'char', 'phone', 'word_char', 'char_space'],
                        help='output unit')
    parser.add_argument('--wp_model', type=str, default=False, nargs='?',
                        help='wordpiece model path')

    return parser
