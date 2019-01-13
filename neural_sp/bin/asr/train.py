#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Train the ASR model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import cProfile
import numpy as np
import os
# from setproctitle import setproctitle
import shutil
import time
import torch
from tqdm import tqdm

from neural_sp.bin.asr.args import parse
from neural_sp.bin.asr.train_utils import Controller
from neural_sp.bin.asr.train_utils import load_config
from neural_sp.bin.asr.train_utils import Reporter
from neural_sp.bin.asr.train_utils import save_config
from neural_sp.bin.asr.train_utils import set_logger
from neural_sp.datasets.loader_asr import Dataset
from neural_sp.evaluators.character import eval_char
from neural_sp.evaluators.loss import eval_loss
from neural_sp.evaluators.phone import eval_phone
from neural_sp.evaluators.word import eval_word
from neural_sp.evaluators.wordpiece import eval_wordpiece
from neural_sp.models.data_parallel import CustomDataParallel
from neural_sp.models.seq2seq.seq2seq import Seq2seq
from neural_sp.models.seq2seq.transformer import Transformer
from neural_sp.models.rnnlm.rnnlm import RNNLM
from neural_sp.utils.general import mkdir_join


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

decode_params = {
    'recog_batch_size': 1,
    'beam_width': 1,
    'min_len_ratio': 0.0,
    'max_len_ratio': 1.0,
    'length_penalty': 0.0,
    'coverage_penalty': 0.0,
    'coverage_threshold': 0.0,
    'rnnlm_weight': 0.0,
    'resolving_unk': False,
    'fwd_bwd_attention': False
}


def main():

    args = parse()
    args_pre = copy.deepcopy(args)

    # Load a config file
    if args.resume:
        config = load_config(os.path.join(args.resume, 'config.yml'))
        for k, v in config.items():
            setattr(args, k, v)

    # Automatically reduce batch size in multi-GPU setting
    if args.ngpus > 1:
        args.batch_size -= 10
        args.print_step //= args.ngpus

    subsample_factor = 1
    subsample_factor_sub1 = 1
    subsample_factor_sub2 = 1
    subsample_factor_sub3 = 1
    subsample = [int(s) for s in args.subsample.split('_')]
    if args.conv_poolings:
        for p in args.conv_poolings.split('_'):
            p = int(p.split(',')[0].replace('(', ''))
            if p > 1:
                subsample_factor *= p
    if args.train_set_sub1:
        subsample_factor_sub1 = subsample_factor * np.prod(subsample[:args.enc_nlayers_sub1 - 1])
    if args.train_set_sub2:
        subsample_factor_sub2 = subsample_factor * np.prod(subsample[:args.enc_nlayers_sub2 - 1])
    if args.train_set_sub3:
        subsample_factor_sub3 = subsample_factor * np.prod(subsample[:args.enc_nlayers_sub3 - 1])
    subsample_factor *= np.prod(subsample)

    # Load dataset
    train_set = Dataset(csv_path=args.train_set,
                        csv_path_sub1=args.train_set_sub1,
                        csv_path_sub2=args.train_set_sub2,
                        csv_path_sub3=args.train_set_sub3,
                        dict_path=args.dict,
                        dict_path_sub1=args.dict_sub1,
                        dict_path_sub2=args.dict_sub2,
                        dict_path_sub3=args.dict_sub3,
                        unit=args.unit,
                        unit_sub1=args.unit_sub1,
                        unit_sub2=args.unit_sub2,
                        unit_sub3=args.unit_sub3,
                        wp_model=args.wp_model,
                        wp_model_sub1=args.wp_model_sub1,
                        wp_model_sub2=args.wp_model_sub2,
                        wp_model_sub3=args.wp_model_sub3,
                        batch_size=args.batch_size * args.ngpus,
                        nepochs=args.nepochs,
                        min_nframes=args.min_nframes,
                        max_nframes=args.max_nframes,
                        sort_by_input_length=True,
                        short2long=True,
                        sort_stop_epoch=args.sort_stop_epoch,
                        dynamic_batching=args.dynamic_batching,
                        ctc=args.ctc_weight > 0,
                        ctc_sub1=args.ctc_weight_sub1 > 0,
                        ctc_sub2=args.ctc_weight_sub2 > 0,
                        ctc_sub3=args.ctc_weight_sub3 > 0,
                        subsample_factor=subsample_factor,
                        subsample_factor_sub1=subsample_factor_sub1,
                        subsample_factor_sub2=subsample_factor_sub2,
                        subsample_factor_sub3=subsample_factor_sub3,
                        skip_speech=(args.input_type != 'speech'))
    dev_set = Dataset(csv_path=args.dev_set,
                      csv_path_sub1=args.dev_set_sub1,
                      csv_path_sub2=args.dev_set_sub2,
                      csv_path_sub3=args.dev_set_sub3,
                      dict_path=args.dict,
                      dict_path_sub1=args.dict_sub1,
                      dict_path_sub2=args.dict_sub2,
                      dict_path_sub3=args.dict_sub3,
                      unit=args.unit,
                      unit_sub1=args.unit_sub1,
                      unit_sub2=args.unit_sub2,
                      unit_sub3=args.unit_sub3,
                      wp_model=args.wp_model,
                      wp_model_sub1=args.wp_model_sub1,
                      wp_model_sub2=args.wp_model_sub2,
                      wp_model_sub3=args.wp_model_sub3,
                      batch_size=args.batch_size * args.ngpus,
                      min_nframes=args.min_nframes,
                      max_nframes=args.max_nframes,
                      shuffle=True,
                      ctc=args.ctc_weight > 0,
                      ctc_sub1=args.ctc_weight_sub1 > 0,
                      ctc_sub2=args.ctc_weight_sub2 > 0,
                      ctc_sub3=args.ctc_weight_sub3 > 0,
                      subsample_factor=subsample_factor,
                      subsample_factor_sub1=subsample_factor_sub1,
                      subsample_factor_sub2=subsample_factor_sub2,
                      subsample_factor_sub3=subsample_factor_sub3,
                      skip_speech=(args.input_type != 'speech'))
    eval_sets = []
    for set in args.eval_sets:
        eval_sets += [Dataset(csv_path=set,
                              dict_path=args.dict,
                              unit=args.unit,
                              wp_model=args.wp_model,
                              batch_size=1,
                              is_test=True,
                              skip_speech=(args.input_type != 'speech'))]

    args.vocab = train_set.vocab
    args.vocab_sub1 = train_set.vocab_sub1
    args.vocab_sub2 = train_set.vocab_sub2
    args.vocab_sub3 = train_set.vocab_sub3
    args.input_dim = train_set.input_dim

    # Load a RNNLM config file for cold fusion & RNNLM initialization
    # if config['rnnlm_cold_fusion']:
    #     if args.model:
    #         config['rnnlm_config_cold_fusion'] = load_config(
    #             os.path.join(config['rnnlm_cold_fusion'], 'config.yml'), is_eval=True)
    #     elif args.resume:
    #         config = load_config(os.path.join(
    #             args.resume, 'config_rnnlm_cf.yml'))
    #     assert args.unit == config['rnnlm_config_cold_fusion']['unit']
    #     config['rnnlm_config_cold_fusion']['vocab'] = train_set.vocab
    args.rnnlm_cold_fusion = False

    # Model setting
    if args.transformer:
        model = Transformer(args)
        dir_name = 'transformer'
        if len(args.conv_channels) > 0:
            tmp = dir_name
            dir_name = 'conv' + str(len(args.conv_channels.split('_'))) + 'L'
            if args.conv_batch_norm:
                dir_name += 'bn'
            dir_name += tmp
        dir_name += str(args.d_model) + 'H'
        dir_name += str(args.enc_nlayers) + 'L'
        dir_name += str(args.dec_nlayers) + 'L'
        dir_name += '_head' + str(args.attn_nheads)
        dir_name += '_' + args.optimizer
        dir_name += '_lr' + str(args.learning_rate)
        dir_name += '_bs' + str(args.batch_size)
        dir_name += '_ls' + str(args.lsm_prob)
        dir_name += '_' + str(args.pre_process) + 't' + str(args.post_process)
        if args.nstacks > 1:
            dir_name += '_stack' + str(args.nstacks)
        if args.bwd_weight > 0:
            dir_name += '_bwd' + str(args.bwd_weight)
    else:
        model = Seq2seq(args)
        dir_name = args.enc_type
        if args.conv_channels and len(args.conv_channels.split('_')) > 0:
            tmp = dir_name
            dir_name = 'conv' + str(len(args.conv_channels.split('_'))) + 'L'
            if args.conv_batch_norm:
                dir_name += 'bn'
            dir_name += tmp
        dir_name += str(args.enc_nunits) + 'H'
        dir_name += str(args.enc_nprojs) + 'P'
        dir_name += str(args.enc_nlayers) + 'L'
        if args.enc_residual:
            dir_name += '_res'
        dir_name += '_' + args.subsample_type + str(subsample_factor)
        dir_name += '_' + args.dec_type
        dir_name += str(args.dec_nunits) + 'H'
        # dir_name += str(args.dec_nprojs) + 'P'
        dir_name += str(args.dec_nlayers) + 'L'
        if args.dec_loop_type == 'conditional':
            dir_name += '_cond'
        elif args.dec_loop_type == 'lmdecoder':
            dir_name += '_lmdec'
        elif args.dec_loop_type == 'rnmt':
            dir_name += '_rnmt'
        if args.dec_residual:
            dir_name += '_res'
        if args.input_feeding:
            dir_name += '_inputfeed'
        if args.tie_embedding:
            dir_name += '_tie'
        dir_name += '_' + args.attn_type
        if args.attn_nheads > 1:
            dir_name += '_head' + str(args.attn_nheads)
        if args.attn_sigmoid:
            dir_name += '_sig'
        dir_name += '_' + args.optimizer
        dir_name += '_lr' + str(args.learning_rate)
        dir_name += '_bs' + str(args.batch_size)
        dir_name += '_ss' + str(args.ss_prob)
        dir_name += '_ls' + str(args.lsm_prob)
        if args.focal_loss_weight > 0:
            dir_name += '_fl' + str(args.focal_loss_weight)
        if args.layer_norm:
            dir_name += '_layernorm'
        # MTL
        if args.mtl_per_batch:
            dir_name += '_mtlperbatch'
            if args.ctc_weight > 0:
                dir_name += '_' + args.unit + 'ctc'
            if args.bwd_weight > 0:
                dir_name += '_' + args.unit + 'bwd'
            if args.lmobj_weight > 0:
                dir_name += '_' + args.unit + 'lmobj'
            if args.train_set_sub1:
                dir_name += '_' + args.unit_sub1 + str(args.vocab_sub1)
                if args.ctc_weight_sub1 == 0:
                    dir_name += 'att'
                elif args.ctc_weight_sub1 == args.sub1_weight:
                    dir_name += 'ctc'
                else:
                    dir_name += 'attctc'
            if args.train_set_sub2:
                dir_name += '_' + args.unit_sub2 + str(args.vocab_sub2)
                if args.ctc_weight_sub2 == 0:
                    dir_name += 'att'
                elif args.ctc_weight_sub2 == args.sub2_weight:
                    dir_name += 'ctc'
                else:
                    dir_name += 'attctc'
            if args.train_set_sub3:
                dir_name += '_' + args.unit_sub3 + str(args.vocab_sub3)
                if args.ctc_weight_sub3 == 0:
                    dir_name += 'att'
                elif args.ctc_weight_sub3 == args.sub3_weight:
                    dir_name += 'ctc'
                else:
                    dir_name += 'attctc'
        else:
            if args.ctc_weight > 0:
                dir_name += '_ctc' + str(args.ctc_weight)
            if args.bwd_weight > 0:
                dir_name += '_bwd' + str(args.bwd_weight)
            if args.lmobj_weight > 0:
                dir_name += '_lmobj' + str(args.lmobj_weight)
            if args.sub1_weight > 0:
                if args.ctc_weight_sub1 == args.sub1_weight:
                    dir_name += '_' + str(args.unit_sub1) + str(args.vocab_sub1) + 'ctc' + str(args.ctc_weight_sub1)
                elif args.ctc_weight_sub1 == 0:
                    dir_name += '_' + str(args.unit_sub1) + str(args.vocab_sub1) + 'att' + str(args.sub1_weight)
                else:
                    dir_name += '_' + str(args.unit_sub1) + str(args.vocab_sub1) + 'ctc' + str(args.ctc_weight_sub1) + str(
                        args.unit_sub1) + str(args.vocab_sub1) + 'att' + str(args.sub1_weight - args.ctc_weight_sub1)
            if args.sub2_weight > 0:
                if args.ctc_weight_sub2 == args.sub2_weight:
                    dir_name += '_' + str(args.unit_sub2) + str(args.vocab_sub2) + 'ctc' + str(args.ctc_weight_sub2)
                elif args.ctc_weight_sub2 == 0:
                    dir_name += '_' + str(args.unit_sub2) + str(args.vocab_sub2) + 'att' + str(args.sub2_weight)
                else:
                    dir_name += '_' + str(args.unit_sub2) + str(args.vocab_sub2) + 'ctc' + str(args.ctc_weight_sub2) + str(
                        args.unit_sub2) + str(args.vocab_sub2) + 'att' + str(args.sub2_weight - args.ctc_weight_sub2)
            if args.sub3_weight > 0:
                if args.ctc_weight_sub3 == args.sub3_weight:
                    dir_name += '_' + str(args.unit_sub3) + str(args.vocab_sub3) + 'ctc' + str(args.ctc_weight_sub3)
                elif args.ctc_weight_sub3 == 0:
                    dir_name += '_' + str(args.unit_sub3) + str(args.vocab_sub3) + 'att' + str(args.sub3_weight)
                else:
                    dir_name += '_' + str(args.unit_sub3) + str(args.vocab_sub3) + 'ctc' + str(args.ctc_weight_sub3) + str(
                        args.unit_sub3) + str(args.vocab_sub3) + 'att' + str(args.sub3_weight - args.ctc_weight_sub3)
        if args.task_specific_layer:
            dir_name += '_tsl'
        # Pre-training
        if args.pretrained_model and os.path.isdir(args.pretrained_model):
            # Load a config file
            config_pre = load_config(os.path.join(args.pretrained_model, 'config.yml'))
            dir_name += '_' + config_pre['unit'] + 'pt'
        if args.twin_net_weight > 0:
            dir_name += '_twin' + str(args.twin_net_weight)

    if not args.resume:
        # Load pre-trained RNNLM
        # if config['rnnlm_cold_fusion']:
        #     rnnlm = RNNLM(args)
        #     rnnlm.load_checkpoint(save_path=config['rnnlm_cold_fusion'], epoch=-1)
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
        save_path = mkdir_join(args.model, '_'.join(os.path.basename(args.train_set).split('.')[:-1]), dir_name)
        model.set_save_path(save_path)  # avoid overwriting

        # Save the config file as a yaml file
        save_config(vars(args), model.save_path)

        # Save the dictionary & wp_model
        shutil.copy(args.dict, os.path.join(model.save_path, 'dict.txt'))
        if args.dict_sub1:
            shutil.copy(args.dict_sub1, os.path.join(model.save_path, 'dict_sub1.txt'))
        if args.dict_sub2:
            shutil.copy(args.dict_sub2, os.path.join(model.save_path, 'dict_sub2.txt'))
        if args.dict_sub3:
            shutil.copy(args.dict_sub3, os.path.join(model.save_path, 'dict_sub3.txt'))
        if args.unit == 'wp':
            shutil.copy(args.wp_model, os.path.join(model.save_path, 'wp.model'))
        if args.unit_sub1 == 'wp':
            shutil.copy(args.wp_model_sub1, os.path.join(model.save_path, 'wp_sub1.model'))
        if args.unit_sub2 == 'wp':
            shutil.copy(args.wp_model_sub2, os.path.join(model.save_path, 'wp_sub2.model'))
        if args.unit_sub3 == 'wp':
            shutil.copy(args.wp_model_sub3, os.path.join(model.save_path, 'wp_sub3.model'))

        # Setting for logging
        logger = set_logger(os.path.join(model.save_path, 'train.log'), key='training')

        for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
            logger.info('%s: %s' % (k, str(v)))

        # Count total parameters
        for n in sorted(list(model.num_params_dict.keys())):
            nparams = model.num_params_dict[n]
            logger.info("%s %d" % (n, nparams))
        logger.info("Total %.2f M parameters" % (model.total_parameters / 1000000))
        logger.info(model)

        # Initialize with pre-trained model's parameters
        if args.pretrained_model and os.path.isdir(args.pretrained_model):
            # Merge config with args
            for k, v in config_pre.items():
                setattr(args_pre, k, v)

            # Load the ASR model
            model_pre = Seq2seq(args_pre)
            model_pre.load_checkpoint(args.pretrained_model, epoch=-1)

            # Overwrite parameters
            param_dict = dict(model_pre.named_parameters())
            for n, p in model.named_parameters():
                if n in param_dict.keys() and p.size() == param_dict[n].size():
                    p.data = param_dict[n].data
                    logger.info('Overwrite %s' % n)

        # Set optimizer
        model.set_optimizer(optimizer=args.optimizer,
                            learning_rate_init=float(args.learning_rate),
                            weight_decay=float(args.weight_decay),
                            clip_grad_norm=args.clip_grad_norm,
                            lr_schedule=False,
                            factor=args.decay_rate,
                            patience_epoch=args.decay_patient_epoch)

        epoch, step = 1, 1
        learning_rate = float(args.learning_rate)
        metric_dev_best = 10000

    # NOTE: Restart from the last checkpoint
    # elif args.resume:
    #     # Set save path
    #     model.save_path = args.resume
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
    #         save_path=args.resume, epoch=-1, restart=True)
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
    #     if config['rnnlm_cold_fusion']:
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
    # if args.job_name:
    #     setproctitle(args.job_name)
    # else:
    #     setproctitle(dir_name)

    # Set learning rate controller
    lr_controller = Controller(learning_rate_init=learning_rate,
                               decay_type=args.decay_type,
                               decay_start_epoch=args.decay_start_epoch,
                               decay_rate=args.decay_rate,
                               decay_patient_epoch=args.decay_patient_epoch,
                               lower_better=True,
                               best_value=metric_dev_best,
                               model_size=args.d_model,
                               warmup_step=args.warmup_step,
                               factor=1)

    # Set reporter
    reporter = Reporter(model.module.save_path, tensorboard=True)

    if args.mtl_per_batch:
        # NOTE: from easier to harder tasks
        tasks = []
        if 1 - args.bwd_weight - args.ctc_weight - args.sub1_weight - args.sub2_weight - args.sub3_weight > 0:
            tasks += ['ys']
        if args.bwd_weight > 0:
            tasks = ['ys.bwd'] + tasks
        if args.ctc_weight > 0:
            tasks = ['ys.ctc'] + tasks
        if args.lmobj_weight > 0:
            tasks = ['ys.lmobj'] + tasks
        if args.train_set_sub1:
            if args.sub1_weight - args.bwd_weight_sub1 - args.ctc_weight_sub1 > 0:
                tasks = ['ys_sub1'] + tasks
            if args.bwd_weight_sub1 > 0:
                tasks = ['ys_sub1.bwd'] + tasks
            if args.ctc_weight_sub1 > 0:
                tasks = ['ys_sub1.ctc'] + tasks
            if args.lmobj_weight_sub1 > 0:
                tasks = ['ys_sub1.lmobj'] + tasks
        if args.train_set_sub2:
            if args.sub2_weight - args.bwd_weight_sub2 - args.ctc_weight_sub2 > 0:
                tasks = ['ys_sub2'] + tasks
            if args.bwd_weight_sub2 > 0:
                tasks = ['ys_sub2.bwd'] + tasks
            if args.ctc_weight_sub2 > 0:
                tasks = ['ys_sub2.ctc'] + tasks
            if args.lmobj_weight_sub2 > 0:
                tasks = ['ys_sub2.lmobj'] + tasks
        if args.train_set_sub3:
            if args.sub3_weight - args.bwd_weight_sub3 - args.ctc_weight_sub3 > 0:
                tasks = ['ys_sub3'] + tasks
            if args.bwd_weight_sub3 > 0:
                tasks = ['ys_sub3.bwd'] + tasks
            if args.ctc_weight_sub3 > 0:
                tasks = ['ys_sub3.ctc'] + tasks
            if args.lmobj_weight_sub3 > 0:
                tasks = ['ys_sub2.lmobj'] + tasks
    else:
        tasks = ['all']

    start_time_train = time.time()
    start_time_epoch = time.time()
    start_time_step = time.time()
    not_improved_epoch = 0
    pbar_epoch = tqdm(total=len(train_set))
    while True:
        # Compute loss in the training set
        batch_train, is_new_epoch = train_set.next()

        # Change tasks depending on task
        for task in tasks:
            model.module.optimizer.zero_grad()
            loss, reporter = model(batch_train, reporter=reporter, task=task)
            if len(model.device_ids) > 1:
                loss.backward(torch.ones(len(model.device_ids)))
            else:
                loss.backward()
            loss.detach()  # Trancate the graph
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.clip_grad_norm)
            model.module.optimizer.step()
            loss_train = loss.item()
            del loss
        reporter.step(is_eval=False)

        # Update learning rate
        if args.decay_type == 'warmup':
            model.module.optimizer, learning_rate = lr_controller.warmup_lr(
                optimizer=model.module.optimizer,
                learning_rate=learning_rate,
                step=step)

        if step % args.print_step == 0:
            # Compute loss in the dev set
            batch_dev = dev_set.next()[0]
            # Change tasks depending on task
            for task in tasks:
                loss, reporter = model(batch_dev, reporter=reporter, task=task,
                                       is_eval=True)
                loss_dev = loss.item()
                del loss
            reporter.step(is_eval=True)

            duration_step = time.time() - start_time_step
            if args.input_type == 'speech':
                x_len = max(len(x) for x in batch_train['xs'])
            elif args.input_type == 'text':
                x_len = max(len(x) for x in batch_train['ys'])
            logger.info("step:%d(ep:%.2f) loss:%.3f(%.3f)/lr:%.5f/bs:%d/x_len:%d (%.2f min)" %
                        (step, train_set.epoch_detail,
                         loss_train, loss_dev,
                         learning_rate, len(batch_train['utt_ids']),
                         x_len, duration_step / 60))
            start_time_step = time.time()
        step += args.ngpus
        pbar_epoch.update(len(batch_train['utt_ids']))

        # Save fugures of loss and accuracy
        if step % (args.print_step * 10) == 0:
            reporter.snapshot()

        # Save checkpoint and evaluate model per epoch
        if is_new_epoch:
            duration_epoch = time.time() - start_time_epoch
            logger.info('========== EPOCH:%d (%.2f min) ==========' % (epoch, duration_epoch / 60))

            if epoch < args.eval_start_epoch:
                # Save the model
                model.module.save_checkpoint(model.module.save_path, epoch, step - 1,
                                             learning_rate, metric_dev_best)
            else:
                start_time_eval = time.time()
                # dev
                if args.metric == 'edit_distance':
                    if args.unit in ['word', 'word_char']:
                        metric_dev = eval_word([model.module], dev_set, decode_params,
                                               epoch=epoch)[0]
                        logger.info('WER (%s): %.3f %%' % (dev_set.set, metric_dev))
                    elif args.unit == 'wp':
                        metric_dev = eval_wordpiece([model.module], dev_set, decode_params,
                                                    epoch=epoch)[0]
                        logger.info('WER (%s): %.3f %%' % (dev_set.set, metric_dev))
                    elif 'char' in args.unit:
                        dev_results = eval_char([model.module], dev_set, decode_params,
                                                epoch=epoch)
                        metric_dev = dev_results[1][0]
                        wer_dev = dev_results[0][0]
                        logger.info('CER (%s): %.3f %%' % (dev_set.set, metric_dev))
                        logger.info('WER (%s): %.3f %%' % (dev_set.set, wer_dev))
                    elif 'phone' in args.unit:
                        metric_dev = eval_phone([model.module], dev_set, decode_params,
                                                epoch=epoch)[0]
                        logger.info('PER (%s): %.3f %%' % (dev_set.set, metric_dev))
                elif args.metric == 'loss':
                    metric_dev = eval_loss([model.module], dev_set, decode_params)
                    logger.info('Loss (%s): %.3f %%' % (dev_set.set, metric_dev))
                else:
                    raise NotImplementedError()

                # Update learning rate
                if args.decay_type != 'warmup':
                    model.module.optimizer, learning_rate = lr_controller.decay_lr(
                        optimizer=model.module.optimizer,
                        learning_rate=learning_rate,
                        epoch=epoch,
                        value=metric_dev)

                if metric_dev < metric_dev_best:
                    metric_dev_best = metric_dev
                    not_improved_epoch = 0
                    logger.info('||||| Best Score |||||')

                    # Save the model
                    model.module.save_checkpoint(model.module.save_path, epoch, step - 1,
                                                 learning_rate, metric_dev_best)

                    # test
                    for eval_set in eval_sets:
                        if args.metric == 'edit_distance':
                            if args.unit in ['word', 'word_char']:
                                wer_test = eval_word([model.module], eval_set, decode_params,
                                                     epoch=epoch)[0]
                                logger.info('WER (%s): %.3f %%' % (eval_set.set, wer_test))
                            elif args.unit == 'wp':
                                wer_test = eval_wordpiece([model.module], eval_set, decode_params,
                                                          epoch=epoch)[0]
                                logger.info('WER (%s): %.3f %%' % (eval_set.set, wer_test))
                            elif 'char' in args.unit:
                                test_results = eval_char([model.module], eval_set, decode_params,
                                                         epoch=epoch)
                                cer_test = test_results[1][0]
                                wer_test = test_results[0][0]
                                logger.info('CER (%s): %.3f %%' % (eval_set.set, cer_test))
                                logger.info('WER (%s): %.3f %%' % (eval_set.set, wer_test))
                            elif 'phone' in args.unit:
                                per_test = eval_phone([model.module], eval_set, decode_params,
                                                      epoch=epoch)[0]
                                logger.info('PER (%s): %.3f %%' % (eval_set.set, per_test))
                        elif args.metric == 'loss':
                            loss_test = eval_loss([model.module], eval_set, decode_params)
                            logger.info('Loss (%s): %.3f %%' % (eval_set.set, loss_test))
                        else:
                            raise NotImplementedError()
                else:
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
                        learning_rate_init=float(args.learning_rate),  # TODO: ?
                        weight_decay=float(args.weight_decay),
                        clip_grad_norm=args.clip_grad_norm,
                        lr_schedule=False,
                        factor=args.decay_rate,
                        patience_epoch=args.decay_patient_epoch)
                    logger.info('========== Convert to SGD ==========')

            pbar_epoch = tqdm(total=len(train_set))

            if epoch == args.nepochs:
                break

            start_time_step = time.time()
            start_time_epoch = time.time()
            epoch += 1

    duration_train = time.time() - start_time_train
    logger.info('Total time: %.2f hour' % (duration_train / 3600))

    if reporter.tensorboard:
        reporter.tf_writer.close()
    pbar_epoch.close()

    return model.module.save_path


if __name__ == '__main__':
    # Setting for profiling
    pr = cProfile.Profile()
    save_path = pr.runcall(main)
    pr.dump_stats(os.path.join(save_path, 'train.profile'))
