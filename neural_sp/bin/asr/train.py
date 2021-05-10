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
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
from neural_sp.datasets.asr.build import build_dataloader
from neural_sp.evaluators.accuracy import eval_accuracy
from neural_sp.evaluators.character import eval_char
from neural_sp.evaluators.phone import eval_phone
from neural_sp.evaluators.ppl import eval_ppl
from neural_sp.evaluators.word import eval_word
from neural_sp.evaluators.wordpiece import eval_wordpiece
from neural_sp.evaluators.wordpiece_bleu import eval_wordpiece_bleu
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

logger = logging.getLogger(__name__)


def main(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args_init = copy.deepcopy(args)
    args_teacher = copy.deepcopy(args)

    # Load a conf file
    if args.resume:
        conf = load_config(os.path.join(os.path.dirname(args.resume), 'conf.yml'))
        for k, v in conf.items():
            if k != 'resume':
                setattr(args, k, v)

    args = compute_subsampling_factor(args)
    resume_epoch = int(args.resume.split('-')[-1]) if args.resume else 0

    # Load dataset
    train_set = build_dataloader(args=args,
                                 tsv_path=args.train_set,
                                 tsv_path_sub1=args.train_set_sub1,
                                 tsv_path_sub2=args.train_set_sub2,
                                 batch_size=args.batch_size,
                                 batch_size_type=args.batch_size_type,
                                 max_n_frames=args.max_n_frames,
                                 resume_epoch=resume_epoch,
                                 sort_by=args.sort_by,
                                 short2long=args.sort_short2long,
                                 sort_stop_epoch=args.sort_stop_epoch,
                                 num_workers=args.workers,
                                 pin_memory=args.pin_memory,
                                 distributed=args.distributed,
                                 word_alignment_dir=args.train_word_alignment,
                                 ctc_alignment_dir=args.train_ctc_alignment)
    dev_set = build_dataloader(args=args,
                               tsv_path=args.dev_set,
                               tsv_path_sub1=args.dev_set_sub1,
                               tsv_path_sub2=args.dev_set_sub2,
                               batch_size=1 if 'transducer' in args.dec_type else args.batch_size,
                               batch_size_type='seq' if 'transducer' in args.dec_type else args.batch_size_type,
                               max_n_frames=1600,
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
        args.save_path = os.path.dirname(args.resume)
        dir_name = os.path.basename(args.save_path)
    else:
        dir_name = set_asr_model_name(args)
        if args.mbr_training:
            assert args.asr_init
            args.save_path = mkdir_join(os.path.dirname(args.asr_init), dir_name)
        else:
            args.save_path = mkdir_join(args.model_save_dir, '_'.join(
                os.path.basename(args.train_set).split('.')[:-1]), dir_name)
        if args.local_rank > 0:
            time.sleep(1)
        args.save_path = set_save_path(args.save_path)  # avoid overwriting

    # Set logger
    set_logger(os.path.join(args.save_path, 'train.log'), args.stdout, args.local_rank)

    # Load a LM conf file for LM fusion & LM initialization
    if not args.resume and args.external_lm:
        lm_conf = load_config(os.path.join(os.path.dirname(args.external_lm), 'conf.yml'))
        args.lm_conf = argparse.Namespace()
        for k, v in lm_conf.items():
            setattr(args.lm_conf, k, v)
        assert args.unit == args.lm_conf.unit
        assert args.vocab == args.lm_conf.vocab

    # Model setting
    model = Speech2Text(args, args.save_path, train_set.idx2token[0])

    if not args.resume:
        # Save nlsyms, dictionary, and wp_model
        if args.nlsyms:
            shutil.copy(args.nlsyms, os.path.join(args.save_path, 'nlsyms.txt'))
        for sub in ['', '_sub1', '_sub2']:
            if args.get('dict' + sub):
                shutil.copy(args.get('dict' + sub), os.path.join(args.save_path, 'dict' + sub + '.txt'))
            if args.get('unit' + sub) == 'wp':
                shutil.copy(args.get('wp_model' + sub), os.path.join(args.save_path, 'wp' + sub + '.model'))

        for k, v in sorted(args.items(), key=lambda x: x[0]):
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
    optimizer = set_optimizer(model, 'sgd' if resume_epoch > args.convert_to_sgd_epoch else args.optimizer,
                              args.lr, args.weight_decay)

    # Wrap optimizer by learning rate scheduler
    is_transformer = 'former' in args.enc_type or 'former' in args.dec_type or 'former' in args.dec_type_sub1
    scheduler = LRScheduler(optimizer, args.lr,
                            decay_type=args.lr_decay_type,
                            decay_start_epoch=args.lr_decay_start_epoch,
                            decay_rate=args.lr_decay_rate,
                            decay_patient_n_epochs=args.lr_decay_patient_n_epochs,
                            early_stop_patient_n_epochs=args.early_stop_patient_n_epochs,
                            lower_better=args.metric not in ['accuracy', 'bleu'],
                            warmup_start_lr=args.warmup_start_lr,
                            warmup_n_steps=args.warmup_n_steps,
                            peak_lr=0.05 / (args.get('transformer_enc_d_model', 0) **
                                            0.5) if 'conformer' in args.enc_type else 1e6,
                            model_size=args.get('transformer_enc_d_model', args.get('transformer_dec_d_model', 0)),
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
    args.use_apex = args.train_dtype in ["O0", "O1", "O2", "O3"]
    amp, scaler = None, None
    if args.n_gpus >= 1:
        model.cudnn_setting(deterministic=((not is_transformer) and (not args.cudnn_benchmark)) or args.cudnn_deterministic,
                            benchmark=(not is_transformer) and args.cudnn_benchmark)

        # Mixed precision training setting
        if args.use_apex:
            if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
                scaler = torch.cuda.amp.GradScaler()
            else:
                from apex import amp
                model, scheduler.optimizer = amp.initialize(model, scheduler.optimizer,
                                                            opt_level=args.train_dtype)
                from neural_sp.models.seq2seq.decoders.ctc import CTC
                amp.register_float_function(CTC, "loss_fn")
                # NOTE: see https://github.com/espnet/espnet/pull/1779
                amp.init()
                if args.resume:
                    load_checkpoint(args.resume, amp=amp)

        n = torch.cuda.device_count() // args.local_world_size
        device_ids = list(range(args.local_rank * n, (args.local_rank + 1) * n))

        torch.cuda.set_device(device_ids[0])
        model.cuda(device_ids[0])
        if args.distributed:
            model = DDP(model, device_ids=device_ids)
        else:
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
    reporter = Reporter(args, model, args.local_rank)
    args.wandb_id = reporter.wandb_id
    if args.resume:
        n_steps = scheduler.n_steps * args.accum_grad_n_steps
        reporter.resume(n_steps, resume_epoch)

    # Save conf file as a yaml file
    if args.local_rank == 0:
        save_config(args, os.path.join(args.save_path, 'conf.yml'))
        if args.external_lm:
            save_config(args.lm_conf, os.path.join(args.save_path, 'conf_lm.yml'))
        # NOTE: save after reporter for wandb ID

    # Define tasks
    if args.mtl_per_batch:
        # NOTE: from easier to harder tasks
        tasks = []
        if args.total_weight - args.bwd_weight - args.ctc_weight - args.sub1_weight - args.sub2_weight > 0:
            tasks += ['ys']
        if args.bwd_weight > 0:
            tasks = ['ys.bwd'] + tasks
        if args.ctc_weight > 0:
            tasks = ['ys.ctc'] + tasks
        if args.mbr_ce_weight > 0:
            tasks = ['ys.mbr'] + tasks
        for sub in ['sub1', 'sub2']:
            if args.get('train_set_' + sub) is not None:
                if args.get(sub + '_weight', 0) - args.get('ctc_weight_' + sub, 0) > 0:
                    tasks = ['ys_' + sub] + tasks
                if args.get('ctc_weight_' + sub, 0) > 0:
                    tasks = ['ys_' + sub + '.ctc'] + tasks
    else:
        tasks = ['all']

    if args.get('ss_start_epoch', 0) <= resume_epoch:
        model.module.trigger_scheduled_sampling()
    if args.get('mocha_quantity_loss_start_epoch', 0) <= resume_epoch:
        model.module.trigger_quantity_loss()

    start_time_train = time.time()
    for ep in range(resume_epoch, args.n_epochs):
        train_one_epoch(model, train_set, dev_set, eval_sets,
                        scheduler, reporter, logger, args, amp, scaler,
                        tasks, teacher, teacher_lm)

        # Save checkpoint and validate model per epoch
        if reporter.n_epochs + 1 < args.eval_start_epoch:
            scheduler.epoch()  # lr decay
            reporter.epoch()  # plot

            # Save model
            if args.local_rank == 0:
                scheduler.save_checkpoint(
                    model, args.save_path, amp=amp,
                    remove_old=(not is_transformer) and args.remove_old_checkpoints)
        else:
            start_time_eval = time.time()
            # dev
            metric_dev = validate([model.module], dev_set, args, reporter.n_epochs + 1, logger)
            scheduler.epoch(metric_dev)  # lr decay
            reporter.epoch(metric_dev, name=args.metric)  # plot
            reporter.add_scalar('dev/' + args.metric, metric_dev)

            if scheduler.is_topk or is_transformer:
                # Save model
                if args.local_rank == 0:
                    scheduler.save_checkpoint(
                        model, args.save_path, amp=amp,
                        remove_old=(not is_transformer) and args.remove_old_checkpoints)

                # test
                if scheduler.is_topk:
                    for eval_set in eval_sets:
                        validate([model.module], eval_set, args, reporter.n_epochs, logger)

            logger.info('Evaluation time: %.2f min' % ((time.time() - start_time_eval) / 60))

            # Early stopping
            if scheduler.is_early_stop:
                break

            # Convert to fine-tuning stage
            if reporter.n_epochs == args.convert_to_sgd_epoch:
                scheduler.convert_to_sgd(model, args.lr, args.weight_decay,
                                         decay_type='always', decay_rate=0.5)

        if reporter.n_epochs >= args.n_epochs:
            break
        if args.get('ss_start_epoch', 0) == (ep + 1):
            model.module.trigger_scheduled_sampling()
        if args.get('mocha_quantity_loss_start_epoch', 0) == (ep + 1):
            model.module.trigger_quantity_loss()

    logger.info('Total time: %.2f hour' % ((time.time() - start_time_train) / 3600))
    reporter.close()

    return args.save_path


def train_one_epoch(model, train_set, dev_set, eval_sets,
                    scheduler, reporter, logger, args, amp, scaler,
                    tasks, teacher, teacher_lm):
    """Train model for one epoch."""
    if args.local_rank == 0:
        pbar_epoch = tqdm(total=len(train_set))
    num_replicas = args.local_world_size
    accum_grad_n_steps = max(1, args.accum_grad_n_steps // num_replicas)
    print_step = args.print_step // num_replicas

    session_prev = None
    _accum_n_steps = 0  # reset at every epoch
    epoch_detail_prev = train_set.epoch_detail
    start_time_step = time.time()
    start_time_epoch = time.time()
    n_rest = len(train_set)

    for batch_train in train_set:
        if args.discourse_aware and batch_train['sessions'][0] != session_prev:
            model.module.reset_session()
        session_prev = batch_train['sessions'][0]
        _accum_n_steps += 1
        num_samples = len(batch_train['utt_ids']) * num_replicas
        n_rest -= num_samples
        is_new_epoch = (n_rest == 0)

        # Compute loss in the training set
        reporter.add_scalar('learning_rate', scheduler.lr)
        if _accum_n_steps == 1:
            loss_train = 0  # moving average over gradient accumulation
        for i_task, task in enumerate(tasks):
            if args.use_apex and scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, observation = model(batch_train, task=task,
                                              teacher=teacher, teacher_lm=teacher_lm)
            else:
                loss, observation = model(batch_train, task=task,
                                          teacher=teacher, teacher_lm=teacher_lm)
            reporter.add_observation(observation)
            if args.distributed:
                loss *= num_replicas
            loss /= accum_grad_n_steps

            if args.use_apex:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    with amp.scale_loss(loss, scheduler.optimizer) as scaled_loss:
                        scaled_loss.backward()
            else:
                loss.backward()

            loss.detach()  # Truncate the graph
            loss_train += loss.item()
            del loss

            if (_accum_n_steps >= accum_grad_n_steps or is_new_epoch) and i_task == len(tasks) - 1:
                if args.clip_grad_norm > 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        model.module.parameters(), args.clip_grad_norm)
                    reporter.add_scalar('total_norm.' + task, total_norm)
                if args.use_apex and scaler is not None:
                    scaler.step(scheduler.optimizer)
                    scaler.update()
                    scheduler.step(skip_optimizer=True)  # update lr only
                else:
                    scheduler.step()
                scheduler.zero_grad(set_to_none=True)
                _accum_n_steps = 0
                reporter.add_scalar('train/total_loss', loss_train)
                # NOTE: parameters are forcibly updated at the end of every epoch

        if args.local_rank == 0:
            pbar_epoch.update(num_samples)

        if reporter.n_steps > 0 and reporter.n_steps % print_step == 0:
            # Compute loss in the dev set
            batch_dev = next(iter(dev_set))
            loss, observation = model(batch_dev, task='all', is_eval=True)
            reporter.add_observation(observation, is_eval=True)
            loss_dev = loss.item()
            del loss
            reporter.add_scalar('dev/total_loss', loss_dev)
            reporter.step(is_eval=True)

            if args.input_type == 'speech':
                xlen = max(len(x) for x in batch_train['xs'])
                ylen = max(len(y) for y in batch_train['ys'])
            elif args.input_type == 'text':
                xlen = max(len(x) for x in batch_train['ys'])
                ylen = max(len(y) for y in batch_train['ys_sub1'])
            logger.info("rank:%d, step:%d(ep:%.2f) loss:%.3f(%.3f)/lr:%.7f/bs:%d/xlen:%d/ylen:%d (%.2f min)" %
                        (args.local_rank, reporter.n_steps, reporter.n_epochs + train_set.epoch_detail,
                         loss_train, loss_dev, scheduler.lr, num_samples,
                         xlen, ylen, (time.time() - start_time_step) / 60))
            start_time_step = time.time()

        reporter.step()

        # Save figures of loss and accuracy
        if args.local_rank == 0 and reporter.n_steps > 0 and reporter.n_steps % (print_step * 10) == 0:
            reporter.snapshot()
            model.module.plot_attention()
            model.module.plot_ctc()

        # Ealuate model every 0.1 epoch during MBR training
        if args.mbr_training:
            if int(train_set.epoch_detail * 10) != int(epoch_detail_prev * 10):
                sub_epoch = int(train_set.epoch_detail * 10) / 10
                # dev
                metric_dev = validate([model.module], dev_set, args, sub_epoch, logger)
                reporter.epoch(metric_dev, name=args.metric)  # plot
                # Save model
                if args.local_rank == 0:
                    scheduler.save_checkpoint(
                        model, args.save_path, remove_old=False, amp=amp,
                        epoch_detail=sub_epoch)
                # test
                for eval_set in eval_sets:
                    validate([model.module], eval_set, args, sub_epoch, logger)
            epoch_detail_prev = train_set.epoch_detail

    train_set.reset(is_new_epoch=True)
    logger.info('========== EPOCH:%d (%.2f min) ==========' %
                (reporter.n_epochs + 1, (time.time() - start_time_epoch) / 60))
    if args.local_rank == 0:
        pbar_epoch.close()


def validate(models, dataloader, args, epoch, logger):
    """Validate performance per epoch."""
    if args.metric == 'edit_distance':
        if args.unit in ['word', 'word_char']:
            metric = eval_word(models, dataloader, args, epoch, args.local_rank)[0]
            logger.info('WER (%s, ep:%d): %.2f %%' % (dataloader.set, epoch, metric))

        elif args.unit == 'wp':
            metric, cer = eval_wordpiece(models, dataloader, args, epoch, args.local_rank)
            logger.info('WER (%s, ep:%d): %.2f %%' % (dataloader.set, epoch, metric))
            logger.info('CER (%s, ep:%d): %.2f %%' % (dataloader.set, epoch, cer))

        elif 'char' in args.unit:
            wer, cer = eval_char(models, dataloader, args, epoch, args.local_rank)
            logger.info('WER (%s, ep:%d): %.2f %%' % (dataloader.set, epoch, wer))
            logger.info('CER (%s, ep:%d): %.2f %%' % (dataloader.set, epoch, cer))
            if dataloader.corpus in ['aishell1']:
                metric = cer
            else:
                metric = wer

        elif 'phone' in args.unit:
            metric = eval_phone(models, dataloader, args, epoch, args.local_rank)
            logger.info('PER (%s, ep:%d): %.2f %%' % (dataloader.set, epoch, metric))

    elif args.metric == 'ppl':
        metric = eval_ppl(models, dataloader, args.batch_size)[0]
        logger.info('PPL (%s, ep:%d): %.3f' % (dataloader.set, epoch, metric))

    elif args.metric == 'loss':
        metric = eval_ppl(models, dataloader, args.batch_size)[1]
        logger.info('Loss (%s, ep:%d): %.5f' % (dataloader.set, epoch, metric))

    elif args.metric == 'accuracy':
        metric = eval_accuracy(models, dataloader, args.batch_size)
        logger.info('Accuracy (%s, ep:%d): %.3f' % (dataloader.set, epoch, metric))

    elif args.metric == 'bleu':
        metric = eval_wordpiece_bleu(models, dataloader, args, epoch, args.local_rank)
        logger.info('BLEU (%s, ep:%d): %.3f' % (dataloader.set, epoch, metric))

    else:
        raise NotImplementedError(args.metric)

    return metric


def spmd_main(args):
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend=args.dist_backend)
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    main(args)

    # Tear down the process group
    dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_args_train(sys.argv[1:])
    args.distributed = args.n_gpus > 1 and args.local_world_size > 1

    if args.distributed:
        spmd_main(args)
    else:
        main(args)
