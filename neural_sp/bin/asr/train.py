#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Train the ASR model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import cProfile
import numpy as np
import os
from setproctitle import setproctitle
import shutil
import time
import torch
from tqdm import tqdm

from neural_sp.bin.args_asr import parse
from neural_sp.bin.lr_controller import Controller
from neural_sp.bin.train_utils import load_config
from neural_sp.bin.train_utils import save_config
from neural_sp.bin.train_utils import set_logger
from neural_sp.bin.train_utils import set_save_path
from neural_sp.bin.train_utils import load_checkpoint
from neural_sp.bin.train_utils import save_checkpoint
from neural_sp.bin.reporter import Reporter
from neural_sp.datasets.loader_asr import Dataset
from neural_sp.evaluators.character import eval_char
from neural_sp.evaluators.phone import eval_phone
from neural_sp.evaluators.ppl import eval_ppl
from neural_sp.evaluators.word import eval_word
from neural_sp.evaluators.wordpiece import eval_wordpiece
from neural_sp.models.data_parallel import CustomDataParallel
from neural_sp.models.seq2seq.seq2seq import Seq2seq
from neural_sp.models.seq2seq.skip_thought import SkipThought
from neural_sp.utils import mkdir_join


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


def main():

    args = parse()
    args_pt = copy.deepcopy(args)
    args_teacher = copy.deepcopy(args)

    # Load a conf file
    if args.resume:
        conf = load_config(os.path.join(os.path.dirname(args.resume), 'conf.yml'))
        for k, v in conf.items():
            if k != 'resume':
                setattr(args, k, v)
    recog_params = vars(args)

    # Automatically reduce batch size in multi-GPU setting
    if args.n_gpus > 1:
        args.batch_size -= 10
        args.print_step //= args.n_gpus

    # Compute subsampling factor
    subsample_factor = 1
    subsample_factor_sub1 = 1
    subsample_factor_sub2 = 1
    subsample = [int(s) for s in args.subsample.split('_')]
    if args.conv_poolings and 'conv' in args.enc_type:
        for p in args.conv_poolings.split('_'):
            subsample_factor *= int(p.split(',')[0].replace('(', ''))
    else:
        subsample_factor = np.prod(subsample)
    if args.train_set_sub1:
        if args.conv_poolings and 'conv' in args.enc_type:
            subsample_factor_sub1 = subsample_factor * np.prod(subsample[:args.enc_n_layers_sub1 - 1])
        else:
            subsample_factor_sub1 = subsample_factor
    if args.train_set_sub2:
        if args.conv_poolings and 'conv' in args.enc_type:
            subsample_factor_sub2 = subsample_factor * np.prod(subsample[:args.enc_n_layers_sub2 - 1])
        else:
            subsample_factor_sub2 = subsample_factor

    skip_thought = 'skip' in args.enc_type

    # Load dataset
    train_set = Dataset(corpus=args.corpus,
                        tsv_path=args.train_set,
                        tsv_path_sub1=args.train_set_sub1,
                        tsv_path_sub2=args.train_set_sub2,
                        dict_path=args.dict,
                        dict_path_sub1=args.dict_sub1,
                        dict_path_sub2=args.dict_sub2,
                        nlsyms=args.nlsyms,
                        unit=args.unit,
                        unit_sub1=args.unit_sub1,
                        unit_sub2=args.unit_sub2,
                        wp_model=args.wp_model,
                        wp_model_sub1=args.wp_model_sub1,
                        wp_model_sub2=args.wp_model_sub2,
                        batch_size=args.batch_size * args.n_gpus,
                        n_epochs=args.n_epochs,
                        min_n_frames=args.min_n_frames,
                        max_n_frames=args.max_n_frames,
                        sort_by_input_length=True,
                        short2long=True,
                        sort_stop_epoch=args.sort_stop_epoch,
                        dynamic_batching=args.dynamic_batching,
                        ctc=args.ctc_weight > 0,
                        ctc_sub1=args.ctc_weight_sub1 > 0,
                        ctc_sub2=args.ctc_weight_sub2 > 0,
                        subsample_factor=subsample_factor,
                        subsample_factor_sub1=subsample_factor_sub1,
                        subsample_factor_sub2=subsample_factor_sub2,
                        contextualize=args.contextualize,
                        skip_thought=skip_thought)
    dev_set = Dataset(corpus=args.corpus,
                      tsv_path=args.dev_set,
                      tsv_path_sub1=args.dev_set_sub1,
                      tsv_path_sub2=args.dev_set_sub2,
                      dict_path=args.dict,
                      dict_path_sub1=args.dict_sub1,
                      dict_path_sub2=args.dict_sub2,
                      nlsyms=args.nlsyms,
                      unit=args.unit,
                      unit_sub1=args.unit_sub1,
                      unit_sub2=args.unit_sub2,
                      wp_model=args.wp_model,
                      wp_model_sub1=args.wp_model_sub1,
                      wp_model_sub2=args.wp_model_sub2,
                      batch_size=args.batch_size * args.n_gpus,
                      min_n_frames=args.min_n_frames,
                      max_n_frames=args.max_n_frames,
                      shuffle=True if args.contextualize else False,
                      ctc=args.ctc_weight > 0,
                      ctc_sub1=args.ctc_weight_sub1 > 0,
                      ctc_sub2=args.ctc_weight_sub2 > 0,
                      subsample_factor=subsample_factor,
                      subsample_factor_sub1=subsample_factor_sub1,
                      subsample_factor_sub2=subsample_factor_sub2,
                      contextualize=args.contextualize,
                      skip_thought=skip_thought)
    eval_sets = []
    for s in args.eval_sets:
        eval_sets += [Dataset(corpus=args.corpus,
                              tsv_path=s,
                              dict_path=args.dict,
                              nlsyms=args.nlsyms,
                              unit=args.unit,
                              wp_model=args.wp_model,
                              batch_size=1,
                              contextualize=args.contextualize,
                              skip_thought=skip_thought,
                              is_test=True)]

    args.vocab = train_set.vocab
    args.vocab_sub1 = train_set.vocab_sub1
    args.vocab_sub2 = train_set.vocab_sub2
    args.input_dim = train_set.input_dim

    # Load a LM conf file for cold fusion & LM initialization
    if args.lm_fusion:
        if args.model:
            lm_conf = load_config(os.path.join(os.path.dirname(args.lm_fusion), 'conf.yml'))
        elif args.resume:
            lm_conf = load_config(os.path.join(os.path.dirname(args.resume), 'conf_lm.yml'))
        args.lm_conf = argparse.Namespace()
        for k, v in lm_conf.items():
            setattr(args.lm_conf, k, v)
        assert args.unit == args.lm_conf.unit
        assert args.vocab == args.lm_conf.vocab

    # Set save path
    if args.resume:
        save_path = os.path.dirname(args.resume)
        dir_name = os.path.basename(save_path)
    else:
        dir_name = make_model_name(args, subsample_factor)
        save_path = mkdir_join(args.model, '_'.join(os.path.basename(args.train_set).split('.')[:-1]), dir_name)
        save_path = set_save_path(save_path)  # avoid overwriting

    # Set logger
    logger = set_logger(os.path.join(save_path, 'train.log'), key='training')

    # Model setting
    model = SkipThought(args) if skip_thought else Seq2seq(args)
    model.save_path = save_path

    if args.resume:
        # Set optimizer
        epoch = int(args.resume.split('-')[-1])
        model.set_optimizer(optimizer='sgd' if epoch > conf['convert_to_sgd_epoch'] + 1 else conf['optimizer'],
                            learning_rate=float(conf['learning_rate']),  # on-the-fly
                            weight_decay=float(conf['weight_decay']))

        # Restore the last saved model
        model, checkpoint = load_checkpoint(model, args.resume, resume=True)
        lr_controller = checkpoint['lr_controller']
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        metric_dev_best = checkpoint['metric_dev_best']

        # Resume between convert_to_sgd_epoch and convert_to_sgd_epoch + 1
        if epoch == conf['convert_to_sgd_epoch'] + 1:
            model.set_optimizer(optimizer='sgd',
                                learning_rate=args.learning_rate,
                                weight_decay=float(conf['weight_decay']))
            logger.info('========== Convert to SGD ==========')
    else:
        # Save the conf file as a yaml file
        save_config(vars(args), os.path.join(model.save_path, 'conf.yml'))
        if args.lm_fusion:
            save_config(args.lm_conf, os.path.join(model.save_path, 'conf_lm.yml'))

        # Save the nlsyms, dictionar, and wp_model
        if args.nlsyms:
            shutil.copy(args.nlsyms, os.path.join(model.save_path, 'nlsyms.txt'))
        for sub in ['', '_sub1', '_sub2']:
            if getattr(args, 'dict' + sub):
                shutil.copy(getattr(args, 'dict' + sub), os.path.join(model.save_path, 'dict' + sub + '.txt'))
            if getattr(args, 'unit' + sub) == 'wp':
                shutil.copy(getattr(args, 'wp_model' + sub), os.path.join(model.save_path, 'wp' + sub + '.model'))

        for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
            logger.info('%s: %s' % (k, str(v)))

        # Count total parameters
        for n in sorted(list(model.num_params_dict.keys())):
            nparams = model.num_params_dict[n]
            logger.info("%s %d" % (n, nparams))
        logger.info("Total %.2f M parameters" % (model.total_parameters / 1000000))
        logger.info(model)

        # Initialize with pre-trained model's parameters
        if args.pretrained_model and os.path.isfile(args.pretrained_model):
            # Load the ASR model
            conf_pt = load_config(os.path.join(os.path.dirname(args.pretrained_model), 'conf.yml'))
            for k, v in conf_pt.items():
                setattr(args_pt, k, v)
            model_pt = Seq2seq(args_pt)
            model_pt, _ = load_checkpoint(model_pt, args.pretrained_model)

            # Overwrite parameters
            only_enc = (args.enc_n_layers != args_pt.enc_n_layers) or (
                args.unit != args_pt.unit) or args_pt.ctc_weight == 1
            param_dict = dict(model_pt.named_parameters())
            for n, p in model.named_parameters():
                if n in param_dict.keys() and p.size() == param_dict[n].size():
                    if only_enc and 'enc' not in n:
                        continue
                    if args.lm_fusion_type == 'cache' and 'output' in n:
                        continue
                    p.data = param_dict[n].data
                    logger.info('Overwrite %s' % n)

        if args.teacher and os.path.isfile(args.teacher):
            # Load the ASR model
            conf_teacher = load_config(os.path.join(os.path.dirname(args.teacher), 'conf.yml'))
            for k, v in conf_teacher.items():
                setattr(args_teacher, k, v)
            # Setting for knowledge distillation
            args_teacher.ss_prob = 0
            args.lsm_prob = 0
            model_teacher = Seq2seq(args_teacher)
            model_teacher, _ = load_checkpoint(model_teacher, args.teacher)
        else:
            model_teacher = None

        # Set optimizer
        model.set_optimizer(
            optimizer=args.optimizer,
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            transformer='transformer' in args.enc_type or args.dec_type == 'transformer')

        epoch, step = 1, 1
        metric_dev_best = 10000

        # Set learning rate controller
        lr_controller = Controller(learning_rate=float(args.learning_rate),
                                   decay_type=args.decay_type,
                                   decay_start_epoch=args.decay_start_epoch,
                                   decay_rate=args.decay_rate,
                                   decay_patient_n_epochs=args.decay_patient_n_epochs,
                                   lower_better=True,
                                   best_value=metric_dev_best,
                                   model_size=args.d_model,
                                   warmup_start_learning_rate=args.warmup_start_learning_rate,
                                   warmup_n_steps=args.warmup_n_steps,
                                   lr_init_factor=10,
                                   transformer='transformer' in args.enc_type or args.dec_type == 'transformer')

    train_set.epoch = epoch - 1  # start from index:0

    # GPU setting
    if args.n_gpus >= 1:
        model = CustomDataParallel(model,
                                   device_ids=list(range(0, args.n_gpus, 1)),
                                   deterministic=False,
                                   benchmark=True)
        model.cuda()
        if model_teacher is not None:
            model_teacher.cuda()

    logger.info('PID: %s' % os.getpid())
    logger.info('USERNAME: %s' % os.uname()[1])

    # Set process name
    if args.job_name:
        setproctitle(args.job_name)
    else:
        setproctitle(dir_name)

    # Set reporter
    reporter = Reporter(model.module.save_path, tensorboard=True)

    if args.mtl_per_batch:
        # NOTE: from easier to harder tasks
        tasks = []
        if 1 - args.bwd_weight - args.ctc_weight - args.sub1_weight - args.sub2_weight > 0:
            tasks += ['ys']
        if args.bwd_weight > 0:
            tasks = ['ys.bwd'] + tasks
        if args.ctc_weight > 0:
            tasks = ['ys.ctc'] + tasks
        if args.lmobj_weight > 0:
            tasks = ['ys.lmobj'] + tasks
        for sub in ['sub1', 'sub2']:
            if getattr(args, 'train_set_' + sub):
                if getattr(args, sub + '_weight') - getattr(args, 'ctc_weight_' + sub) > 0:
                    tasks = ['ys_' + sub] + tasks
                if getattr(args, 'ctc_weight_' + sub) > 0:
                    tasks = ['ys_' + sub + '.ctc'] + tasks
    else:
        tasks = ['all']

    start_time_train = time.time()
    start_time_epoch = time.time()
    start_time_step = time.time()
    not_improved_n_epochs = 0
    pbar_epoch = tqdm(total=len(train_set))
    while True:
        # Compute loss in the training set
        batch_train, is_new_epoch = train_set.next()

        # Change tasks depending on task
        for task in tasks:
            if skip_thought:
                loss, reporter = model(batch_train['ys'],
                                       ys_prev=batch_train['ys_prev'],
                                       ys_next=batch_train['ys_next'],
                                       reporter=reporter)
            else:
                loss, reporter = model(batch_train, reporter=reporter, task=task,
                                       teacher=model_teacher)
            loss /= args.accum_grad_n_steps
            if len(model.device_ids) > 1:
                loss.backward(torch.ones(len(model.device_ids)))
            else:
                loss.backward()
            loss.detach()  # Trancate the graph
            if step % args.accum_grad_n_steps == 0:
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.clip_grad_norm)
                model.module.optimizer.step()
                model.module.optimizer.zero_grad()
            loss_train = loss.item()
            del loss

        reporter.step(is_eval=False)

        # Update learning rate
        if step < args.warmup_n_steps:
            model.module.optimizer = lr_controller.warmup(model.module.optimizer, step=step)

        if step % args.print_step == 0:
            # Compute loss in the dev set
            batch_dev = dev_set.next()[0]
            # Change tasks depending on task
            for task in tasks:
                if skip_thought:
                    loss, reporter = model(batch_dev['ys'],
                                           ys_prev=batch_dev['ys_prev'],
                                           ys_next=batch_dev['ys_next'],
                                           reporter=reporter,
                                           is_eval=True)
                else:
                    loss, reporter = model(batch_dev, reporter=reporter, task=task,
                                           is_eval=True)
                loss_dev = loss.item()
                del loss
            reporter.step(is_eval=True)

            duration_step = time.time() - start_time_step
            if args.input_type == 'speech':
                xlen = max(len(x) for x in batch_train['xs'])
                ylen = max(len(y) for y in batch_train['ys'])
            elif args.input_type == 'text':
                xlen = max(len(x) for x in batch_train['ys'])
                ylen = max(len(y) for y in batch_train['ys_sub1'])
            logger.info("step:%d(ep:%.2f) loss:%.3f(%.3f)/lr:%.5f/bs:%d/xlen:%d/ylen:%d (%.2f min)" %
                        (step, train_set.epoch_detail,
                         loss_train, loss_dev,
                         lr_controller.lr, len(batch_train['utt_ids']),
                         xlen, ylen, duration_step / 60))
            start_time_step = time.time()
        step += args.n_gpus
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
                save_checkpoint(model.module, model.module.save_path, lr_controller,
                                epoch, step - 1, metric_dev_best,
                                remove_old_checkpoints=True)
                reporter._epoch += 1
                # TODO(hirofumi): fix later
            else:
                start_time_eval = time.time()
                # dev
                if args.metric == 'edit_distance':
                    if args.unit in ['word', 'word_char']:
                        metric_dev = eval_word([model.module], dev_set, recog_params,
                                               epoch=epoch)[0]
                        logger.info('WER (%s): %.2f %%' % (dev_set.set, metric_dev))
                    elif args.unit == 'wp':
                        metric_dev, cer_dev = eval_wordpiece([model.module], dev_set, recog_params,
                                                             epoch=epoch)
                        logger.info('WER (%s): %.2f %%' % (dev_set.set, metric_dev))
                        logger.info('CER (%s): %.2f %%' % (dev_set.set, cer_dev))
                    elif 'char' in args.unit:
                        metric_dev, cer_dev = eval_char([model.module], dev_set, recog_params,
                                                        epoch=epoch)
                        logger.info('WER (%s): %.2f %%' % (dev_set.set, metric_dev))
                        logger.info('CER (%s): %.2f %%' % (dev_set.set, cer_dev))
                    elif 'phone' in args.unit:
                        metric_dev = eval_phone([model.module], dev_set, recog_params,
                                                epoch=epoch)
                        logger.info('PER (%s): %.2f %%' % (dev_set.set, metric_dev))
                elif args.metric == 'ppl':
                    metric_dev = eval_ppl([model.module], dev_set, recog_params=recog_params)[0]
                    logger.info('PPL (%s): %.2f' % (dev_set.set, metric_dev))
                elif args.metric == 'loss':
                    metric_dev = eval_ppl([model.module], dev_set, recog_params=recog_params)[1]
                    logger.info('Loss (%s): %.2f' % (dev_set.set, metric_dev))
                else:
                    raise NotImplementedError(args.metric)
                reporter.epoch(metric_dev)

                # Update learning rate
                model.module.optimizer = lr_controller.decay(
                    model.module.optimizer, epoch=epoch, value=metric_dev)

                if metric_dev < metric_dev_best:
                    metric_dev_best = metric_dev
                    not_improved_n_epochs = 0
                    logger.info('||||| Best Score |||||')

                    # Save the model
                    save_checkpoint(model.module, model.module.save_path, lr_controller,
                                    epoch, step - 1, metric_dev_best,
                                    remove_old_checkpoints=True)

                    # test
                    for s in eval_sets:
                        if args.metric == 'edit_distance':
                            if args.unit in ['word', 'word_char']:
                                wer_test = eval_word([model.module], s, recog_params,
                                                     epoch=epoch)[0]
                                logger.info('WER (%s): %.2f %%' % (s.set, wer_test))
                            elif args.unit == 'wp':
                                wer_test, cer_test = eval_wordpiece([model.module], s, recog_params,
                                                                    epoch=epoch)
                                logger.info('WER (%s): %.2f %%' % (s.set, wer_test))
                                logger.info('CER (%s): %.2f %%' % (s.set, cer_test))
                            elif 'char' in args.unit:
                                wer_test, cer_test = eval_char([model.module], s, recog_params,
                                                               epoch=epoch)
                                logger.info('WER (%s): %.2f %%' % (s.set, wer_test))
                                logger.info('CER (%s): %.2f %%' % (s.set, cer_test))
                            elif 'phone' in args.unit:
                                per_test = eval_phone([model.module], s, recog_params,
                                                      epoch=epoch)
                                logger.info('PER (%s): %.2f %%' % (s.set, per_test))
                        elif args.metric == 'ppl':
                            ppl_test = eval_ppl([model.module], s, recog_params=recog_params)[0]
                            logger.info('PPL (%s): %.2f' % (s.set, ppl_test))
                        elif args.metric == 'loss':
                            loss_test = eval_ppl([model.module], s, recog_params=recog_params)[1]
                            logger.info('Loss (%s): %.2f' % (s.set, loss_test))
                        else:
                            raise NotImplementedError(args.metric)
                else:
                    not_improved_n_epochs += 1

                    # start scheduled sampling
                    if args.ss_prob > 0:
                        model.module.scheduled_sampling_trigger()

                duration_eval = time.time() - start_time_eval
                logger.info('Evaluation time: %.2f min' % (duration_eval / 60))

                # Early stopping
                if not_improved_n_epochs == args.not_improved_patient_n_epochs:
                    break

                # Convert to fine-tuning stage
                if epoch == args.convert_to_sgd_epoch:
                    model.module.set_optimizer('sgd',
                                               learning_rate=args.learning_rate,
                                               weight_decay=float(args.weight_decay))
                    lr_controller = Controller(learning_rate=args.learning_rate,
                                               decay_type='epoch',
                                               decay_start_epoch=epoch,
                                               decay_rate=0.5,
                                               lower_better=True)
                    logger.info('========== Convert to SGD ==========')

            pbar_epoch = tqdm(total=len(train_set))

            if epoch == args.n_epochs:
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


def make_model_name(args, subsample_factor):

    # encoder
    dir_name = args.enc_type.replace('conv_', '')
    if args.conv_channels and len(args.conv_channels.split('_')) > 0 and 'conv' in args.enc_type:
        tmp = dir_name
        dir_name = 'conv' + str(len(args.conv_channels.split('_'))) + 'L'
        if args.conv_batch_norm:
            dir_name += 'bn'
        if args.conv_residual:
            dir_name += 'res'
        dir_name += tmp
    if 'transformer' in args.enc_type:
        dir_name += str(args.d_model) + 'H'
        dir_name += str(args.transformer_enc_n_layers) + 'L'
    else:
        dir_name += str(args.enc_n_units) + 'H'
        dir_name += str(args.enc_n_projs) + 'P'
        dir_name += str(args.enc_n_layers) + 'L'
        if args.enc_residual:
            dir_name += 'res'
        if args.enc_nin:
            dir_name += 'NiN'
    if args.n_stacks > 1:
        dir_name += '_stack' + str(args.n_stacks)
    else:
        dir_name += '_' + args.subsample_type + str(subsample_factor)
    if args.sequence_summary_network:
        dir_name += '_ssn'

    # decoder
    if args.ctc_weight < 1:
        dir_name += '_' + args.dec_type
        if args.dec_type == 'transformer':
            dir_name += str(args.d_model) + 'H'
            dir_name += str(args.transformer_dec_n_layers) + 'L'
            dir_name += '_' + args.transformer_attn_type
        else:
            dir_name += str(args.dec_n_units) + 'H'
            dir_name += str(args.dec_n_projs) + 'P'
            dir_name += str(args.dec_n_layers) + 'L'
            dir_name += '_' + args.dec_loop_type
            if args.dec_residual:
                dir_name += 'res'
            if args.input_feeding:
                dir_name += '_inputfeed'
            dir_name += '_' + args.attn_type
            if args.attn_sigmoid:
                dir_name += '_sig'
        if args.attn_n_heads > 1:
            dir_name += '_head' + str(args.attn_n_heads)
        if args.tie_embedding:
            dir_name += '_tie'

    # optimization and regularization
    dir_name += '_' + args.optimizer
    dir_name += '_lr' + str(args.learning_rate)
    dir_name += '_bs' + str(args.batch_size)
    if args.ctc_weight < 1:
        dir_name += '_ss' + str(args.ss_prob)
    dir_name += '_ls' + str(args.lsm_prob)
    if args.focal_loss_weight > 0:
        dir_name += '_fl' + str(args.focal_loss_weight)
    if args.warmup_n_steps > 0:
        dir_name += '_warmpup' + str(args.warmup_n_steps)

    # LM integration
    if args.lm_fusion:
        dir_name += '_' + args.lm_fusion_type

    # MTL
    if args.mtl_per_batch:
        if args.ctc_weight > 0:
            dir_name += '_' + args.unit + 'ctc'
        if args.bwd_weight > 0:
            dir_name += '_' + args.unit + 'bwd'
        if args.lmobj_weight > 0:
            dir_name += '_' + args.unit + 'lmobj'
        for sub in ['sub1', 'sub2']:
            if getattr(args, 'train_set_' + sub):
                dir_name += '_' + getattr(args, 'unit_' + sub) + str(getattr(args, 'vocab_' + sub))
                if getattr(args, 'ctc_weight_' + sub) > 0:
                    dir_name += 'ctc'
                if getattr(args, sub + '_weight') - getattr(args, 'ctc_weight_' + sub) > 0:
                    dir_name += 'fwd'
    else:
        if args.ctc_weight > 0:
            dir_name += '_ctc' + str(args.ctc_weight)
        if args.bwd_weight > 0:
            dir_name += '_bwd' + str(args.bwd_weight)
        if args.lmobj_weight > 0:
            dir_name += '_lmobj' + str(args.lmobj_weight)
        for sub in ['sub1', 'sub2']:
            if getattr(args, sub + '_weight') > 0:
                dir_name += '_' + getattr(args, 'unit_' + sub) + str(getattr(args, 'vocab_' + sub))
                if getattr(args, 'ctc_weight_' + sub) > 0:
                    dir_name += 'ctc' + str(getattr(args, 'ctc_weight_' + sub))
                if getattr(args, sub + '_weight') - getattr(args, 'ctc_weight_' + sub) > 0:
                    dir_name += 'fwd' + str(1 - getattr(args, sub + '_weight') - getattr(args, 'ctc_weight_' + sub))
    if args.task_specific_layer:
        dir_name += '_tsl'

    # contextualization
    if args.contextualize:
        dir_name += '_' + str(args.contextualize)

    # Pre-training
    if args.pretrained_model and os.path.isfile(args.pretrained_model):
        conf_pt = load_config(os.path.join(os.path.dirname(args.pretrained_model), 'conf.yml'))
        dir_name += '_' + conf_pt['unit'] + 'pt'
    if args.freeze_encoder:
        dir_name += '_encfreeze'

    if args.adaptive_softmax:
        dir_name += '_adaptiveSM'

    # knowledge distillation
    if args.teacher:
        dir_name += '_distillation'

    return dir_name


if __name__ == '__main__':
    # Setting for profiling
    pr = cProfile.Profile()
    save_path = pr.runcall(main)
    pr.dump_stats(os.path.join(save_path, 'train.profile'))
