#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Train the ASR model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cProfile
import os
# from setproctitle import setproctitle
import shutil
from tensorboardX import SummaryWriter
import time
import torch
from tqdm import tqdm
import warnings

from neural_sp.bin.asr.train_utils import Controller
from neural_sp.bin.asr.train_utils import Reporter
from neural_sp.bin.asr.train_utils import Updater
from neural_sp.datasets.loader_asr import Dataset
from neural_sp.evaluators.character import eval_char
from neural_sp.evaluators.loss import eval_loss
from neural_sp.evaluators.phone import eval_phone
from neural_sp.evaluators.word import eval_word
from neural_sp.evaluators.wordpiece import eval_wordpiece
from neural_sp.models.data_parallel import CustomDataParallel
from neural_sp.models.seq2seq.seq2seq import Seq2seq
from neural_sp.models.rnnlm.rnnlm import RNNLM
from neural_sp.utils.config import load_config
from neural_sp.utils.config import save_config
from neural_sp.utils.general import mkdir_join
from neural_sp.utils.general import set_logger

parser = argparse.ArgumentParser()
# general
parser.add_argument('--ngpus', type=int, default=0,
                    help='number of GPUs (0 indicates CPU)')
parser.add_argument('--config', type=str, default=None,
                    help='path to a yaml file for configuration')
parser.add_argument('--model', type=str, default=None,
                    help='directory to save a model')
parser.add_argument('--resume_model', type=str, default=None,
                    help='path to the model to resume training')
parser.add_argument('--job_name', type=str, default='',
                    help='name of job')
# dataset
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
                    choices=['word', 'wordpiece', 'char', 'phone'],
                    help='')
parser.add_argument('--label_type_sub', type=str, default='char',
                    choices=['wordpiece', 'char', 'phone'],
                    help='')
parser.add_argument('--wp_model', type=str, default=False, nargs='?',
                    help='path to of the wordpiece model')
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
parser.add_argument('--init_with_enc', type=bool, default=False,
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
parser.add_argument('--metric', type=str, default='ler',
                    choices=['ler', 'loss', 'acc', 'ppl'],
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

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

decode_params = {
    'batch_size': 1,
    'beam_width': 1,
    'min_len_ratio': 0.0,
    'max_len_ratio': 1.0,
    'length_penalty': 0.0,
    'coverage_penalty': 0.0,
    'coverage_threshold': 0.0,
    'rnnlm_weight': 0.0,
    'resolving_unk': False,
}


def main():

    # Load a config file
    if args.resume_model is None:
        config = load_config(args.config)
    else:
        # Restart from the last checkpoint
        config = load_config(os.path.join(args.resume_model, 'config.yml'))

    # Check differences between args and yaml comfiguraiton
    for k, v in vars(args).items():
        if k not in config.keys():
            warnings.warn("key %s is automatically set to %s" % (k, str(v)))

    # Merge config with args
    for k, v in config.items():
        setattr(args, k, v)

    # Automatically reduce batch size in multi-GPU setting
    if args.ngpus > 1:
        args.batch_size -= 10
        args.print_step //= args.ngpus

    subsample_factor = 1
    subsample_factor_sub = 1
    for p in args.conv_poolings:
        if len(p) > 0:
            subsample_factor *= p[0]
    if args.train_set_sub is not None:
        subsample_factor_sub = subsample_factor * (2**sum(args.subsample[:args.enc_num_layers_sub - 1]))
    subsample_factor *= 2**sum(args.subsample)

    # Load dataset
    train_set = Dataset(csv_path=args.train_set,
                        dict_path=args.dict,
                        label_type=args.label_type,
                        batch_size=args.batch_size * args.ngpus,
                        max_epoch=args.num_epochs,
                        max_num_frames=args.max_num_frames,
                        min_num_frames=args.min_num_frames,
                        sort_by_input_length=True,
                        short2long=True,
                        sort_stop_epoch=args.sort_stop_epoch,
                        dynamic_batching=True,
                        use_ctc=args.ctc_weight > 0,
                        subsample_factor=subsample_factor,
                        csv_path_sub=args.train_set_sub,
                        dict_path_sub=args.dict_sub,
                        label_type_sub=args.label_type_sub,
                        use_ctc_sub=args.ctc_weight_sub > 0,
                        subsample_factor_sub=subsample_factor_sub,
                        skip_speech=(args.input_type != 'speech'))
    dev_set = Dataset(csv_path=args.dev_set,
                      dict_path=args.dict,
                      label_type=args.label_type,
                      batch_size=args.batch_size * args.ngpus,
                      max_epoch=args.num_epochs,
                      max_num_frames=args.max_num_frames,
                      min_num_frames=args.min_num_frames,
                      shuffle=True,
                      use_ctc=args.ctc_weight > 0,
                      subsample_factor=subsample_factor,
                      csv_path_sub=args.dev_set_sub,
                      dict_path_sub=args.dict_sub,
                      label_type_sub=args.label_type_sub,
                      use_ctc_sub=args.ctc_weight_sub > 0,
                      subsample_factor_sub=subsample_factor_sub,
                      skip_speech=(args.input_type != 'speech'))
    eval_sets = []
    for set in args.eval_sets:
        eval_sets += [Dataset(csv_path=set,
                              dict_path=args.dict,
                              label_type=args.label_type,
                              batch_size=1,
                              max_epoch=args.num_epochs,
                              is_test=True,
                              skip_speech=(args.input_type != 'speech'))]

    args.num_classes = train_set.num_classes
    args.input_dim = train_set.input_dim
    args.num_classes_sub = train_set.num_classes_sub

    # Load a RNNLM config file for cold fusion & RNNLM initialization
    # if config['rnnlm_cf']:
    #     if args.model is not None:
    #         config['rnnlm_config_cold_fusion'] = load_config(
    #             os.path.join(config['rnnlm_cf'], 'config.yml'), is_eval=True)
    #     elif args.resume_model is not None:
    #         config = load_config(os.path.join(
    #             args.resume_model, 'config_rnnlm_cf.yml'))
    #     assert args.label_type == config['rnnlm_config_cold_fusion']['label_type']
    #     config['rnnlm_config_cold_fusion']['num_classes'] = train_set.num_classes
    args.rnnlm_cf = None
    args.rnnlm_init = None

    # Model setting
    model = Seq2seq(args)
    model.name = args.enc_type
    if len(args.conv_channels) > 0:
        tmp = model.name
        model.name = 'conv' + str(len(args.conv_channels)) + 'L'
        if args.conv_batch_norm:
            model.name += 'bn'
        model.name += tmp
    model.name += str(args.enc_num_units) + 'H'
    model.name += str(args.enc_num_projs) + 'P'
    model.name += str(args.enc_num_layers) + 'L'
    model.name += '_subsample' + str(subsample_factor)
    model.name += '_' + args.dec_type
    model.name += str(args.dec_num_units) + 'H'
    # model.name += str(args.dec_num_projs) + 'P'
    model.name += str(args.dec_num_layers) + 'L'
    model.name += '_' + args.att_type
    if args.att_num_heads > 1:
        model.name += '_head' + str(args.att_num_heads)
    model.name += '_' + args.optimizer
    model.name += '_lr' + str(args.learning_rate)
    model.name += '_bs' + str(args.batch_size)
    model.name += '_ss' + str(args.ss_prob)
    model.name += '_ls' + str(args.lsm_prob)
    if args.ctc_weight > 0:
        model.name += '_ctc' + str(args.ctc_weight)
    if args.bwd_weight > 0:
        model.name += '_bwd' + str(args.bwd_weight)
    if args.main_task_weight < 1:
        model.name += '_main' + str(args.main_task_weight)
        if args.ctc_weight_sub > 0:
            model.name += '_ctcsub' + str(args.ctc_weight_sub * (1 - args.main_task_weight))
        else:
            model.name += '_attsub' + str(1 - args.main_task_weight)

    if args.resume_model is None:
        # Load pre-trained RNNLM
        # if config['rnnlm_cf']:
        #     rnnlm = RNNLM(args)
        #     rnnlm.load_checkpoint(save_path=config['rnnlm_cf'], epoch=-1)
        #     rnnlm.flatten_parameters()
        #
        #     # Fix RNNLM parameters
        #     for param in rnnlm.parameters():
        #         param.requires_grad = False
        #
        #     # Set pre-trained parameters
        #     if config['rnnlm_config_cold_fusion']['backward']:
        #         model.dec_0_bwd.rnnlm = rnnlm
        #     else:
        #         model.dec_0_fwd.rnnlm = rnnlm
        # TODO(hirofumi): 最初にRNNLMのモデルをコピー

        # Set save path
        save_path = mkdir_join(args.model, '_'.join(os.path.basename(args.train_set).split('.')[:-1]), model.name)
        model.set_save_path(save_path)  # avoid overwriting

        # Save the config file as a yaml file
        save_config(vars(args), model.save_path)

        # Save the dictionary & wp_model
        shutil.copy(args.dict, os.path.join(save_path, 'dict.txt'))
        if args.label_type == 'wordpiece':
            shutil.copy(args.wp_model, os.path.join(save_path, 'wp.model'))

        # Setting for logging
        logger = set_logger(os.path.join(model.save_path, 'train.log'), key='training')

        for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
            logger.info('%s: %s' % (k, str(v)))

        # if os.path.isdir(args.pretrained_model):
        #     # NOTE: Start training from the pre-trained model
        #     # This is defferent from resuming training
        #     model.load_checkpoint(args.pretrained_model, epoch=-1,
        #                           load_pretrained_model=True)

        # Count total parameters
        for name in sorted(list(model.num_params_dict.keys())):
            num_params = model.num_params_dict[name]
            logger.info("%s %d" % (name, num_params))
        logger.info("Total %.2f M parameters" % (model.total_parameters / 1000000))

        # Set optimizer
        model.set_optimizer(optimizer=args.optimizer,
                            learning_rate_init=float(args.learning_rate),
                            weight_decay=float(args.weight_decay),
                            clip_grad_norm=args.clip_grad_norm,
                            lr_schedule=False,
                            factor=args.decay_rate,
                            patience_epoch=args.decay_patient_epoch)

        epoch, step = 1, 0
        learning_rate = float(args.learning_rate)
        metric_dev_best = 10000

    # NOTE: Restart from the last checkpoint
    # elif args.resume_model is not None:
    #     # Set save path
    #     model.save_path = args.resume_model
    #
    #     # Setting for logging
    #     logger = set_logger(os.path.join(model.save_path, 'train.log'), key='training')
    #
    #     # Set optimizer
    #     model.set_optimizer(
    #         optimizer=config['optimizer'],
    #         learning_rate_init=float(config['learning_rate']),  # on-the-fly
    #         weight_decay=float(config['weight_decay']),
    #         clip_grad_norm=config['clip_grad_norm'],
    #         lr_schedule=False,
    #         factor=config['decay_rate'],
    #         patience_epoch=config['decay_patient_epoch'])
    #
    #     # Restore the last saved model
    #     epoch, step, learning_rate, metric_dev_best = model.load_checkpoint(
    #         save_path=args.resume_model, epoch=-1, restart=True)
    #
    #     if epoch >= config['convert_to_sgd_epoch']:
    #         model.set_optimizer(
    #             optimizer='sgd',
    #             learning_rate_init=float(config['learning_rate']),  # on-the-fly
    #             weight_decay=float(config['weight_decay']),
    #             clip_grad_norm=config['clip_grad_norm'],
    #             lr_schedule=False,
    #             factor=config['decay_rate'],
    #             patience_epoch=config['decay_patient_epoch'])
    #
    #     if config['rnnlm_cf']:
    #         if config['rnnlm_config_cold_fusion']['backward']:
    #             model.rnnlm_0_bwd.flatten_parameters()
    #         else:
    #             model.rnnlm_0_fwd.flatten_parameters()

    train_set.epoch = epoch - 1  # start from index:0

    # GPU setting
    if args.ngpus >= 1:
        model = CustomDataParallel(model,
                                   device_ids=list(range(0, args.ngpus, 1)),
                                   deterministic=False,
                                   benchmark=True)
        model.cuda()

    logger.info('PID: %s' % os.getpid())
    logger.info('USERNAME: %s' % os.uname()[1])

    # Set process name
    # setproctitle(args.job_name)

    # Set learning rate controller
    lr_controller = Controller(learning_rate_init=learning_rate,
                               decay_type=args.decay_type,
                               decay_start_epoch=args.decay_start_epoch,
                               decay_rate=args.decay_rate,
                               decay_patient_epoch=args.decay_patient_epoch,
                               lower_better=True,
                               best_value=metric_dev_best)

    # Set reporter
    reporter = Reporter(model.module.save_path, max_loss=300)

    # Set the updater
    updater = Updater(args.clip_grad_norm)

    # Setting for tensorboard
    tf_writer = SummaryWriter(model.module.save_path)

    start_time_train = time.time()
    start_time_epoch = time.time()
    start_time_step = time.time()
    not_improved_epoch = 0
    loss_train_mean, acc_train_mean = 0, 0
    pbar_epoch = tqdm(total=len(train_set))
    pbar_all = tqdm(total=len(train_set) * args.num_epochs)
    while True:
        # Compute loss in the training set (including parameter update)
        batch_train, is_new_epoch = train_set.next()
        model, loss_train, acc_train = updater(model, batch_train)
        loss_train_mean += loss_train
        acc_train_mean += acc_train
        pbar_epoch.update(len(batch_train['utt_ids']))

        if (step + 1) % args.print_step == 0:
            # Compute loss in the dev set
            batch_dev = dev_set.next()[0]
            model, loss_dev, acc_dev = updater(model, batch_dev, is_eval=True)

            loss_train_mean /= args.print_step
            acc_train_mean /= args.print_step
            reporter.step(step, loss_train_mean, loss_dev, acc_train_mean, acc_dev)

            # Logging by tensorboard
            tf_writer.add_scalar('train/loss', loss_train_mean, step + 1)
            tf_writer.add_scalar('dev/loss', loss_dev, step + 1)
            # for n, p in model.module.named_parameters():
            #     n = n.replace('.', '/')
            #     if p.grad is not None:
            #         tf_writer.add_histogram(n, p.data.cpu().numpy(), step + 1)
            #         tf_writer.add_histogram(n + '/grad', p.grad.data.cpu().numpy(), step + 1)

            duration_step = time.time() - start_time_step
            if args.input_type == 'speech':
                x_len = max(len(x) for x in batch_train['xs'])
            elif args.input_type == 'text':
                x_len = max(len(x) for x in batch_train['ys_sub'])
            logger.info("...Step:%d(ep:%.2f) loss:%.2f(%.2f)/acc:%.2f(%.2f)/lr:%.5f/bs:%d/x_len:%d (%.2f min)" %
                        (step + 1, train_set.epoch_detail,
                         loss_train_mean, loss_dev, acc_train_mean, acc_dev,
                         learning_rate, train_set.current_batch_size,
                         x_len, duration_step / 60))
            start_time_step = time.time()
            loss_train_mean, acc_train_mean = 0, 0
        step += args.ngpus

        # Save checkpoint and evaluate model per epoch
        if is_new_epoch:
            duration_epoch = time.time() - start_time_epoch
            logger.info('===== EPOCH:%d (%.2f min) =====' % (epoch, duration_epoch / 60))

            # Save fugures of loss and accuracy
            reporter.epoch()

            if epoch < args.eval_start_epoch:
                # Save the model
                model.module.save_checkpoint(model.module.save_path, epoch, step,
                                             learning_rate, metric_dev_best)
            else:
                start_time_eval = time.time()
                # dev
                if args.metric == 'ler':
                    if args.label_type == 'word':
                        metric_dev = eval_word([model.module], dev_set, decode_params,
                                               epoch=epoch + 1)[0]
                        logger.info('  WER (%s): %.3f %%' % (dev_set.set, metric_dev))
                    elif args.label_type == 'wordpiece':
                        metric_dev = eval_wordpiece([model.module], dev_set, decode_params,
                                                    args.wp_model, epoch=epoch + 1)[0]
                        logger.info('  WER (%s): %.3f %%' % (dev_set.set, metric_dev))
                    elif 'char' in args.label_type:
                        metric_dev = eval_char([model.module], dev_set, decode_params,
                                               epoch=epoch)[1][0]
                        logger.info('  CER (%s): %.3f %%' % (dev_set.set, metric_dev))
                    elif 'phone' in args.label_type:
                        metric_dev = eval_phone([model.module], dev_set, decode_params,
                                                epoch=epoch + 1)[0]
                        logger.info('  PER (%s): %.3f %%' % (dev_set.set, metric_dev))
                elif args.metric == 'loss':
                    metric_dev = eval_loss([model.module], dev_set, decode_params)
                    logger.info('  Loss (%s): %.3f %%' % (dev_set.set, metric_dev))
                else:
                    raise NotImplementedError()

                if metric_dev < metric_dev_best:
                    metric_dev_best = metric_dev
                    not_improved_epoch = 0
                    logger.info('||||| Best Score |||||')

                    # Update learning rate
                    model.module.optimizer, learning_rate = lr_controller.decay_lr(
                        optimizer=model.module.optimizer,
                        learning_rate=learning_rate,
                        epoch=epoch,
                        value=metric_dev)

                    # Save the model
                    model.module.save_checkpoint(model.module.save_path, epoch, step,
                                                 learning_rate, metric_dev_best)

                    # test
                    for eval_set in eval_sets:
                        if args.metric == 'ler':
                            if args.label_type == 'word':
                                wer_test = eval_word([model.module], eval_set, decode_params,
                                                     epoch=epoch + 1)[0]
                                logger.info('  WER (%s): %.3f %%' % (eval_set.set, wer_test))
                            elif args.label_type == 'wordpiece':
                                wer_test = eval_wordpiece([model.module], eval_set, decode_params,
                                                          epoch=epoch + 1)[0]
                                logger.info('  WER (%s): %.3f %%' % (eval_set.set, wer_test))
                            elif 'char' in args.label_type:
                                cer_test = eval_char([model.module], eval_set, decode_params,
                                                     epoch=epoch)[1][0]
                                logger.info('  CER (%s): %.3f / %.3f %%' % (eval_set.set, cer_test))
                            elif 'phone' in args.label_type:
                                per_test = eval_phone([model.module], eval_set, decode_params,
                                                      epoch=epoch + 1)[0]
                                logger.info('  PER (%s): %.3f %%' % (eval_set.set, per_test))
                        elif args.metric == 'loss':
                            loss_test = eval_loss([model.module], eval_set, decode_params)
                            logger.info('  Loss (%s): %.3f %%' % (eval_set.set, loss_test))
                        else:
                            raise NotImplementedError()
                else:
                    # Update learning rate
                    model.module.optimizer, learning_rate = lr_controller.decay_lr(
                        optimizer=model.module.optimizer,
                        learning_rate=learning_rate,
                        epoch=epoch,
                        value=metric_dev)

                    not_improved_epoch += 1

                duration_eval = time.time() - start_time_eval
                logger.info('Evaluation time: %.2f min' % (duration_eval / 60))

                # Early stopping
                if not_improved_epoch == args.not_improved_patient_epoch:
                    break

                if epoch == args.convert_to_sgd_epoch:
                    # Convert to fine-tuning stage
                    model.module.set_optimizer(
                        'sgd',
                        learning_rate_init=float(args.learning_rate),
                        weight_decay=float(args.weight_decay),
                        clip_grad_norm=args.clip_grad_norm,
                        lr_schedule=False,
                        factor=args.decay_rate,
                        patience_epoch=args.decay_patient_epoch)
                    logger.info('========== Convert to SGD ==========')

            pbar_epoch = tqdm(total=len(train_set))
            pbar_all.update(len(train_set))

            if epoch == args.num_epochs:
                break

            start_time_step = time.time()
            start_time_epoch = time.time()
            epoch += 1

    duration_train = time.time() - start_time_train
    logger.info('Total time: %.2f hour' % (duration_train / 3600))

    tf_writer.close()
    pbar_epoch.close()
    pbar_all.close()

    return model.module.save_path


if __name__ == '__main__':
    # Setting for profiling
    pr = cProfile.Profile()
    save_path = pr.runcall(main)
    pr.dump_stats(os.path.join(save_path, 'train.profile'))
