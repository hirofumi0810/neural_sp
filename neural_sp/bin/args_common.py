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
    # optimization
    parser.add_argument('--batch_size', type=int, default=50,
                        help='mini-batch size')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adadelta', 'adagrad', 'sgd', 'momentum', 'nesterov', 'noam'],
                        help='type of optimizer')
    parser.add_argument('--n_epochs', type=int, default=25,
                        help='number of epochs to train the model')
    parser.add_argument('--convert_to_sgd_epoch', type=int, default=100,
                        help='epoch to convert to SGD fine-tuning')
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
    parser.add_argument('--warmup_start_lr', type=float, default=0,
                        help='initial learning rate for learning rate warm up')
    parser.add_argument('--warmup_n_steps', type=int, default=0,
                        help='number of steps to warm up learning rate')
    parser.add_argument('--accum_grad_n_steps', type=int, default=1,
                        help='total number of steps to accumulate gradients')
    parser.add_argument('--early_stop_patient_n_epochs', type=int, default=5,
                        help='number of epochs to tolerate stopping training when validation performance is not improved')
    # evaluation
    parser.add_argument('--print_step', type=int, default=200,
                        help='print log per this value')
    parser.add_argument('--eval_start_epoch', type=int, default=1,
                        help='first epoch to start evaluation')
    # initialization
    parser.add_argument('--param_init', type=float, default=0.1,
                        help='')
    # regularization
    parser.add_argument('--clip_grad_norm', type=float, default=5.0,
                        help='')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay parameter')
    parser.add_argument('--lsm_prob', type=float, default=0.0,
                        help='probability of label smoothing')
    # decoding parameters
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
    parser.add_argument('--recog_n_average', type=int, default=1,
                        help='number of models for the model averaging of Transformer')
    return parser
