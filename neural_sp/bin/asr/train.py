#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Train ASR model."""

import argparse
import copy
import cProfile
from distutils.version import LooseVersion
import logging
import os
from setproctitle import setproctitle
import shutil
import sys
import time
import torch
from tqdm import tqdm

from neural_sp.bin.args_asr import parse_args_train
from neural_sp.bin.model_name import set_asr_model_name
from neural_sp.bin.train_utils import (
    compute_subsampling_factor,
    load_checkpoint,
    load_config,
    save_config,
    set_logger,
    set_save_path
)
from neural_sp.evaluators.accuracy import eval_accuracy
from neural_sp.evaluators.character import eval_char
from neural_sp.evaluators.phone import eval_phone
from neural_sp.evaluators.ppl import eval_ppl
from neural_sp.evaluators.word import eval_word
from neural_sp.evaluators.wordpiece import eval_wordpiece
from neural_sp.evaluators.wordpiece_bleu import eval_wordpiece_bleu
from neural_sp.datasets.asr import build_dataloader
from neural_sp.models.data_parallel import (
    CustomDataParallel,
    CPUWrapperASR
)
from neural_sp.models.lm.build import build_lm
from neural_sp.models.seq2seq.speech2text import Speech2Text
from neural_sp.trainers.lr_scheduler import LRScheduler
from neural_sp.trainers.optimizer import set_optimizer
from neural_sp.trainers.reporter import Reporter
from neural_sp.utils import mkdir_join

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

logger = logging.getLogger(__name__)


def main():

    args = parse_args_train(sys.argv[1:])
    args_init = copy.deepcopy(args)
    args_teacher = copy.deepcopy(args)

    # Load a conf file
    if args.resume:
        conf = load_config(os.path.join(os.path.dirname(args.resume), 'conf.yml'))
        for k, v in conf.items():
            if k != 'resume':
                setattr(args, k, v)
    recog_params = vars(args)

    args = compute_subsampling_factor(args)

    # for multi-GPUs
    if args.n_gpus > 1:
        batch_size = args.batch_size * args.n_gpus
        accum_grad_n_steps = max(1, args.accum_grad_n_steps // args.n_gpus)
    else:
        batch_size = args.batch_size
        accum_grad_n_steps = args.accum_grad_n_steps

    # Load dataset
    train_set = build_dataloader(args=args,
                                 tsv_path=args.train_set,
                                 tsv_path_sub1=args.train_set_sub1,
                                 tsv_path_sub2=args.train_set_sub2,
                                 batch_size=batch_size,
                                 n_epochs=args.n_epochs,
                                 sort_by='input',
                                 short2long=args.sort_short2long,
                                 sort_stop_epoch=args.sort_stop_epoch,
                                 num_workers=args.n_gpus,
                                 pin_memory=False,
                                 word_alignment_dir=args.train_word_alignment,
                                 ctc_alignment_dir=args.train_ctc_alignment)
    dev_set = build_dataloader(args=args,
                               tsv_path=args.dev_set,
                               tsv_path_sub1=args.dev_set_sub1,
                               tsv_path_sub2=args.dev_set_sub2,
                               batch_size=batch_size,
                               num_workers=args.n_gpus,
                               pin_memory=False,
                               word_alignment_dir=args.dev_word_alignment,
                               ctc_alignment_dir=args.dev_ctc_alignment)
    eval_sets = [build_dataloader(args=args,
                                  tsv_path=s,
                                  batch_size=1,
                                  is_test=True) for s in args.eval_sets]

    args.vocab = train_set.vocab
    args.vocab_sub1 = train_set.vocab_sub1
    args.vocab_sub2 = train_set.vocab_sub2
    args.input_dim = train_set.input_dim

    # Set save path
    if args.resume:
        save_path = os.path.dirname(args.resume)
        dir_name = os.path.basename(save_path)
    else:
        dir_name = set_asr_model_name(args)
        if args.mbr_training:
            assert args.asr_init
            save_path = mkdir_join(os.path.dirname(args.asr_init), dir_name)
        else:
            save_path = mkdir_join(args.model_save_dir, '_'.join(
                os.path.basename(args.train_set).split('.')[:-1]), dir_name)
        save_path = set_save_path(save_path)  # avoid overwriting

    # Set logger
    set_logger(os.path.join(save_path, 'train.log'), stdout=args.stdout)

    # Load a LM conf file for LM fusion & LM initialization
    if not args.resume and args.external_lm:
        lm_conf = load_config(os.path.join(os.path.dirname(args.external_lm), 'conf.yml'))
        args.lm_conf = argparse.Namespace()
        for k, v in lm_conf.items():
            setattr(args.lm_conf, k, v)
        assert args.unit == args.lm_conf.unit
        assert args.vocab == args.lm_conf.vocab

    # Model setting
    model = Speech2Text(args, save_path, train_set.idx2token[0])

    if not args.resume:
        # Save conf file as a yaml file
        save_config(vars(args), os.path.join(save_path, 'conf.yml'))
        if args.external_lm:
            save_config(args.lm_conf, os.path.join(save_path, 'conf_lm.yml'))

        # Save nlsyms, dictionary, and wp_model
        if args.nlsyms:
            shutil.copy(args.nlsyms, os.path.join(save_path, 'nlsyms.txt'))
        for sub in ['', '_sub1', '_sub2']:
            if getattr(args, 'dict' + sub):
                shutil.copy(getattr(args, 'dict' + sub), os.path.join(save_path, 'dict' + sub + '.txt'))
            if getattr(args, 'unit' + sub) == 'wp':
                shutil.copy(getattr(args, 'wp_model' + sub), os.path.join(save_path, 'wp' + sub + '.model'))

        for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
            logger.info('%s: %s' % (k, str(v)))

        # Count total parameters
        for n in sorted(list(model.num_params_dict.keys())):
            n_params = model.num_params_dict[n]
            logger.info("%s %d" % (n, n_params))
        logger.info("Total %.2f M parameters" % (model.total_parameters / 1000000))
        logger.info('torch version: %s' % str(torch.__version__))
        logger.info(model)

        # Initialize with pre-trained model's parameters
        if args.asr_init:
            # Load ASR model (full model)
            conf_init = load_config(os.path.join(os.path.dirname(args.asr_init), 'conf.yml'))
            for k, v in conf_init.items():
                setattr(args_init, k, v)
            model_init = Speech2Text(args_init)
            load_checkpoint(args.asr_init, model_init)

            # Overwrite parameters
            param_dict = dict(model_init.named_parameters())
            for n, p in model.named_parameters():
                if n in param_dict.keys() and p.size() == param_dict[n].size():
                    if args.asr_init_enc_only and 'enc' not in n:
                        continue
                    p.data = param_dict[n].data
                    logger.info('Overwrite %s' % n)

    # Set optimizer
    if args.resume:
        resume_epoch = int(args.resume.split('-')[-1])
        optimizer = set_optimizer(model, 'sgd' if resume_epoch > args.convert_to_sgd_epoch else args.optimizer,
                                  args.lr, args.weight_decay)
    else:
        resume_epoch = 0
        optimizer = set_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    # Wrap optimizer by learning rate scheduler
    is_transformer = 'former' in args.enc_type or 'former' in args.dec_type
    scheduler = LRScheduler(optimizer, args.lr,
                            decay_type=args.lr_decay_type,
                            decay_start_epoch=args.lr_decay_start_epoch,
                            decay_rate=args.lr_decay_rate,
                            decay_patient_n_epochs=args.lr_decay_patient_n_epochs,
                            early_stop_patient_n_epochs=args.early_stop_patient_n_epochs,
                            lower_better=args.metric not in ['accuracy', 'bleu'],
                            warmup_start_lr=args.warmup_start_lr,
                            warmup_n_steps=args.warmup_n_steps,
                            peak_lr=0.05 / (getattr(args, 'transformer_enc_d_model', 0) **
                                            0.5) if 'conformer' in args.enc_type else 1e6,
                            model_size=getattr(args, 'transformer_enc_d_model',
                                               getattr(args, 'transformer_dec_d_model', 0)),
                            factor=args.lr_factor,
                            noam=args.optimizer == 'noam',
                            save_checkpoints_topk=10 if is_transformer else 1)

    if args.resume:
        # Restore the last saved model
        load_checkpoint(args.resume, model, scheduler)

        # Resume between convert_to_sgd_epoch -1 and convert_to_sgd_epoch
        if resume_epoch == args.convert_to_sgd_epoch:
            scheduler.convert_to_sgd(model, args.lr, args.weight_decay,
                                     decay_type='always', decay_rate=0.5)

    # Load teacher ASR model
    teacher = None
    if args.teacher:
        assert os.path.isfile(args.teacher), 'There is no checkpoint.'
        conf_teacher = load_config(os.path.join(os.path.dirname(args.teacher), 'conf.yml'))
        for k, v in conf_teacher.items():
            setattr(args_teacher, k, v)
        # Setting for knowledge distillation
        args_teacher.ss_prob = 0
        args.lsm_prob = 0
        teacher = Speech2Text(args_teacher)
        load_checkpoint(args.teacher, teacher)

    # Load teacher LM
    teacher_lm = None
    if args.teacher_lm:
        assert os.path.isfile(args.teacher_lm), 'There is no checkpoint.'
        conf_lm = load_config(os.path.join(os.path.dirname(args.teacher_lm), 'conf.yml'))
        args_lm = argparse.Namespace()
        for k, v in conf_lm.items():
            setattr(args_lm, k, v)
        teacher_lm = build_lm(args_lm)
        load_checkpoint(args.teacher_lm, teacher_lm)

    # GPU setting
    use_apex = args.train_dtype in ["O0", "O1", "O2", "O3"]
    amp = None
    if args.n_gpus >= 1:
        model.cudnn_setting(deterministic=not (is_transformer or args.cudnn_benchmark),
                            benchmark=not is_transformer and args.cudnn_benchmark)
        model.cuda()

        # Mixed precision training setting
        if use_apex:
            if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
                scaler = torch.cuda.amp.GradScaler()
            else:
                scaler = None
                from apex import amp
                model, scheduler.optimizer = amp.initialize(model, scheduler.optimizer,
                                                            opt_level=args.train_dtype)
                from neural_sp.models.seq2seq.decoders.ctc import CTC
                amp.register_float_function(CTC, "loss_fn")
                # NOTE: see https://github.com/espnet/espnet/pull/1779
                amp.init()
                if args.resume:
                    load_checkpoint(args.resume, amp=amp)
        model = CustomDataParallel(model, device_ids=list(range(0, args.n_gpus)))

        if teacher is not None:
            teacher.cuda()
        if teacher_lm is not None:
            teacher_lm.cuda()
    else:
        model = CPUWrapperASR(model)

    # Set process name
    logger.info('PID: %s' % os.getpid())
    logger.info('USERNAME: %s' % os.uname()[1])
    logger.info('#GPU: %d' % torch.cuda.device_count())
    setproctitle(args.job_name if args.job_name else dir_name)

    # Set reporter
    reporter = Reporter(save_path)

    if args.mtl_per_batch:
        # NOTE: from easier to harder tasks
        tasks = []
        if 1 - args.bwd_weight - args.ctc_weight - args.sub1_weight - args.sub2_weight > 0:
            tasks += ['ys']
        if args.bwd_weight > 0:
            tasks = ['ys.bwd'] + tasks
        if args.ctc_weight > 0:
            tasks = ['ys.ctc'] + tasks
        if args.mbr_ce_weight > 0:
            tasks = ['ys.mbr'] + tasks
        for sub in ['sub1', 'sub2']:
            if getattr(args, 'train_set_' + sub):
                if getattr(args, sub + '_weight') - getattr(args, 'ctc_weight_' + sub) > 0:
                    tasks = ['ys_' + sub] + tasks
                if getattr(args, 'ctc_weight_' + sub) > 0:
                    tasks = ['ys_' + sub + '.ctc'] + tasks
    else:
        tasks = ['all']

    if getattr(args, 'ss_start_epoch', 0) <= resume_epoch:
        model.module.trigger_scheduled_sampling()
    if getattr(args, 'mocha_quantity_loss_start_epoch', 0) <= resume_epoch:
        model.module.trigger_quantity_loss()

    start_time_train = time.time()
    start_time_epoch = time.time()
    start_time_step = time.time()
    accum_n_steps = 0
    n_steps = scheduler.n_steps * accum_grad_n_steps
    epoch_detail_prev = 0
    for ep in range(resume_epoch, args.n_epochs):
        pbar_epoch = tqdm(total=len(train_set))
        session_prev = None
        for batch_train, is_new_epoch in train_set:
            # Compute loss in the training set
            if args.discourse_aware and batch_train['sessions'][0] != session_prev:
                model.module.reset_session()
            session_prev = batch_train['sessions'][0]
            accum_n_steps += 1

            # Change mini-batch depending on task
            if accum_n_steps == 1:
                loss_train = 0  # average over gradient accumulation
            for task in tasks:
                if use_apex and scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss, observation = model(batch_train, task=task,
                                                  teacher=teacher, teacher_lm=teacher_lm)
                else:
                    loss, observation = model(batch_train, task=task,
                                              teacher=teacher, teacher_lm=teacher_lm)
                loss = loss / accum_grad_n_steps
                reporter.add(observation)
                if use_apex:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        with amp.scale_loss(loss, scheduler.optimizer) as scaled_loss:
                            scaled_loss.backward()
                else:
                    loss.backward()
                loss.detach()  # Truncate the graph
                if accum_n_steps >= accum_grad_n_steps or is_new_epoch:
                    if args.clip_grad_norm > 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            model.module.parameters(), args.clip_grad_norm)
                        reporter.add_tensorboard_scalar('total_norm', total_norm)
                    if use_apex and scaler is not None:
                        scaler.step(scheduler.optimizer)
                        scaler.update()
                        scheduler.step(skip_optimizer=True)  # update lr only
                    else:
                        scheduler.step()
                    scheduler.zero_grad()
                    accum_n_steps = 0
                    # NOTE: parameters are forcibly updated at the end of every epoch
                loss_train += loss.item()
                del loss

            pbar_epoch.update(len(batch_train['utt_ids']))
            reporter.add_tensorboard_scalar('learning_rate', scheduler.lr)
            # NOTE: loss/acc/ppl are already added in the model
            reporter.step()
            n_steps += 1
            # NOTE: n_steps is different from the step counter in Noam Optimizer

            if n_steps % args.print_step == 0:
                # Compute loss in the dev set
                batch_dev = iter(dev_set).next(batch_size=1 if 'transducer' in args.dec_type else None)[0]
                # Change mini-batch depending on task
                for task in tasks:
                    loss, observation = model(batch_dev, task=task, is_eval=True)
                    reporter.add(observation, is_eval=True)
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
                logger.info("step:%d(ep:%.2f) loss:%.3f(%.3f)/lr:%.7f/bs:%d/xlen:%d/ylen:%d (%.2f min)" %
                            (n_steps, scheduler.n_epochs + train_set.epoch_detail,
                             loss_train, loss_dev,
                             scheduler.lr, len(batch_train['utt_ids']),
                             xlen, ylen, duration_step / 60))
                start_time_step = time.time()

            # Save fugures of loss and accuracy
            if n_steps % (args.print_step * 10) == 0:
                reporter.snapshot()
                model.module.plot_attention()
                model.module.plot_ctc()

            # Ealuate model every 0.1 epoch during MBR training
            if args.mbr_training:
                if int(train_set.epoch_detail * 10) != int(epoch_detail_prev * 10):
                    sub_epoch = int(train_set.epoch_detail * 10) / 10
                    # dev
                    metric_dev = evaluate([model.module], dev_set, recog_params, args,
                                          sub_epoch, logger)
                    reporter.epoch(metric_dev, name=args.metric)  # plot
                    # Save model
                    scheduler.save_checkpoint(
                        model, save_path, remove_old=False, amp=amp,
                        epoch_detail=sub_epoch)
                    # test
                    for eval_set in eval_sets:
                        evaluate([model.module], eval_set, recog_params, args,
                                 sub_epoch, logger)
                epoch_detail_prev = train_set.epoch_detail

            if is_new_epoch:
                break

        # Save checkpoint and evaluate model per epoch
        duration_epoch = time.time() - start_time_epoch
        logger.info('========== EPOCH:%d (%.2f min) ==========' %
                    (scheduler.n_epochs + 1, duration_epoch / 60))

        if scheduler.n_epochs + 1 < args.eval_start_epoch:
            scheduler.epoch()  # lr decay
            reporter.epoch()  # plot

            # Save model
            scheduler.save_checkpoint(
                model, save_path, remove_old=not is_transformer and args.remove_old_checkpoints, amp=amp)
        else:
            start_time_eval = time.time()
            # dev
            metric_dev = evaluate([model.module], dev_set, recog_params, args,
                                  scheduler.n_epochs + 1, logger)
            scheduler.epoch(metric_dev)  # lr decay
            reporter.epoch(metric_dev, name=args.metric)  # plot

            if scheduler.is_topk or is_transformer:
                # Save model
                scheduler.save_checkpoint(
                    model, save_path, remove_old=not is_transformer and args.remove_old_checkpoints, amp=amp)

                # test
                if scheduler.is_topk:
                    for eval_set in eval_sets:
                        evaluate([model.module], eval_set, recog_params, args,
                                 scheduler.n_epochs, logger)

            duration_eval = time.time() - start_time_eval
            logger.info('Evaluation time: %.2f min' % (duration_eval / 60))

            # Early stopping
            if scheduler.is_early_stop:
                break

            # Convert to fine-tuning stage
            if scheduler.n_epochs == args.convert_to_sgd_epoch:
                scheduler.convert_to_sgd(model, args.lr, args.weight_decay,
                                         decay_type='always', decay_rate=0.5)

        if scheduler.n_epochs >= args.n_epochs:
            break
        if getattr(args, 'ss_start_epoch', 0) == (ep + 1):
            model.module.trigger_scheduled_sampling()
        if getattr(args, 'mocha_quantity_loss_start_epoch', 0) == (ep + 1):
            model.module.trigger_quantity_loss()

        start_time_step = time.time()
        start_time_epoch = time.time()

    duration_train = time.time() - start_time_train
    logger.info('Total time: %.2f hour' % (duration_train / 3600))

    reporter.tf_writer.close()
    pbar_epoch.close()

    return save_path


def evaluate(models, dataloader, recog_params, args, epoch, logger):

    if args.metric == 'edit_distance':
        if args.unit in ['word', 'word_char']:
            metric = eval_word(models, dataloader, recog_params, epoch=epoch)[0]
            logger.info('WER (%s, ep:%d): %.2f %%' % (dataloader.set, epoch, metric))

        elif args.unit == 'wp':
            metric, cer = eval_wordpiece(models, dataloader, recog_params, epoch=epoch)
            logger.info('WER (%s, ep:%d): %.2f %%' % (dataloader.set, epoch, metric))
            logger.info('CER (%s, ep:%d): %.2f %%' % (dataloader.set, epoch, cer))

        elif 'char' in args.unit:
            wer, cer = eval_char(models, dataloader, recog_params, epoch=epoch)
            logger.info('WER (%s, ep:%d): %.2f %%' % (dataloader.set, epoch, wer))
            logger.info('CER (%s, ep:%d): %.2f %%' % (dataloader.set, epoch, cer))
            if dataloader.corpus in ['aishell1']:
                metric = cer
            else:
                metric = wer

        elif 'phone' in args.unit:
            metric = eval_phone(models, dataloader, recog_params, epoch=epoch)
            logger.info('PER (%s, ep:%d): %.2f %%' % (dataloader.set, epoch, metric))

    elif args.metric == 'ppl':
        metric = eval_ppl(models, dataloader, batch_size=args.batch_size)[0]
        logger.info('PPL (%s, ep:%d): %.3f' % (dataloader.set, epoch, metric))

    elif args.metric == 'loss':
        metric = eval_ppl(models, dataloader, batch_size=args.batch_size)[1]
        logger.info('Loss (%s, ep:%d): %.5f' % (dataloader.set, epoch, metric))

    elif args.metric == 'accuracy':
        metric = eval_accuracy(models, dataloader, batch_size=args.batch_size)
        logger.info('Accuracy (%s, ep:%d): %.3f' % (dataloader.set, epoch, metric))

    elif args.metric == 'bleu':
        metric = eval_wordpiece_bleu(models, dataloader, recog_params, epoch=epoch)
        logger.info('BLEU (%s, ep:%d): %.3f' % (dataloader.set, epoch, metric))

    else:
        raise NotImplementedError(args.metric)

    return metric


if __name__ == '__main__':
    # Setting for profiling
    pr = cProfile.Profile()
    save_path = pr.runcall(main)
    pr.dump_stats(os.path.join(save_path, 'train.profile'))
