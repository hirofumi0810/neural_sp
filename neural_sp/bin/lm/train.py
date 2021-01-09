#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Train LM."""

import cProfile
import logging
import os
from setproctitle import setproctitle
import shutil
import sys
import time
import torch
from tqdm import tqdm

from neural_sp.bin.args_lm import parse_args_train
from neural_sp.bin.model_name import set_lm_name
from neural_sp.bin.train_utils import (
    load_checkpoint,
    load_config,
    save_config,
    set_logger,
    set_save_path
)
from neural_sp.datasets.lm import Dataset
from neural_sp.evaluators.ppl import eval_ppl
from neural_sp.models.data_parallel import CustomDataParallel
from neural_sp.models.data_parallel import CPUWrapperLM
from neural_sp.models.lm.build import build_lm
from neural_sp.trainers.lr_scheduler import LRScheduler
from neural_sp.trainers.optimizer import set_optimizer
from neural_sp.trainers.reporter import Reporter
from neural_sp.utils import mkdir_join

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

logger = logging.getLogger(__name__)


def main():

    args = parse_args_train(sys.argv[1:])

    # Load a conf file
    if args.resume:
        conf = load_config(os.path.join(os.path.dirname(args.resume), 'conf.yml'))
        for k, v in conf.items():
            if k != 'resume':
                setattr(args, k, v)

    # for multi-GPUs
    if args.n_gpus > 1:
        batch_size = args.batch_size * args.n_gpus
        accum_grad_n_steps = max(1, args.accum_grad_n_steps // args.n_gpus)
    else:
        batch_size = args.batch_size
        accum_grad_n_steps = args.accum_grad_n_steps

    # Load dataset
    train_set = Dataset(corpus=args.corpus,
                        tsv_path=args.train_set,
                        dict_path=args.dict,
                        nlsyms=args.nlsyms,
                        unit=args.unit,
                        wp_model=args.wp_model,
                        batch_size=batch_size,
                        n_epochs=args.n_epochs,
                        min_n_tokens=args.min_n_tokens,
                        bptt=args.bptt,
                        shuffle=args.shuffle,
                        backward=args.backward,
                        serialize=args.serialize)
    dev_set = Dataset(corpus=args.corpus,
                      tsv_path=args.dev_set,
                      dict_path=args.dict,
                      nlsyms=args.nlsyms,
                      unit=args.unit,
                      wp_model=args.wp_model,
                      batch_size=batch_size,
                      bptt=args.bptt,
                      backward=args.backward,
                      serialize=args.serialize)
    eval_sets = [Dataset(corpus=args.corpus,
                         tsv_path=s,
                         dict_path=args.dict,
                         nlsyms=args.nlsyms,
                         unit=args.unit,
                         wp_model=args.wp_model,
                         batch_size=1,
                         bptt=args.bptt,
                         backward=args.backward,
                         serialize=args.serialize) for s in args.eval_sets]

    args.vocab = train_set.vocab

    # Set save path
    if args.resume:
        save_path = os.path.dirname(args.resume)
        dir_name = os.path.basename(save_path)
    else:
        dir_name = set_lm_name(args)
        save_path = mkdir_join(args.model_save_dir, '_'.join(
            os.path.basename(args.train_set).split('.')[:-1]), dir_name)
        save_path = set_save_path(save_path)  # avoid overwriting

    # Set logger
    set_logger(os.path.join(save_path, 'train.log'), stdout=args.stdout)

    # Model setting
    model = build_lm(args, save_path)

    if not args.resume:
        # Save conf file as a yaml file
        save_config(vars(args), os.path.join(save_path, 'conf.yml'))

        # Save nlsyms, dictionary, and wp_model
        if args.nlsyms:
            shutil.copy(args.nlsyms, os.path.join(save_path, 'nlsyms.txt'))
        shutil.copy(args.dict, os.path.join(save_path, 'dict.txt'))
        if args.unit == 'wp':
            shutil.copy(args.wp_model, os.path.join(save_path, 'wp.model'))

        for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
            logger.info('%s: %s' % (k, str(v)))

        # Count total parameters
        for n in sorted(list(model.num_params_dict.keys())):
            n_params = model.num_params_dict[n]
            logger.info("%s %d" % (n, n_params))
        logger.info("Total %.2f M parameters" % (model.total_parameters / 1000000))
        logger.info('torch version: %s' % str(torch.__version__))
        logger.info(model)

    # Set optimizer
    if args.resume:
        resume_epoch = int(args.resume.split('-')[-1])
        optimizer = set_optimizer(model, 'sgd' if resume_epoch > args.convert_to_sgd_epoch else args.optimizer,
                                  args.lr, args.weight_decay)
    else:
        resume_epoch = 0
        optimizer = set_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    # Wrap optimizer by learning rate scheduler
    is_transformer = args.lm_type in ['transformer', 'transformer_xl']
    scheduler = LRScheduler(optimizer, args.lr,
                            decay_type=args.lr_decay_type,
                            decay_start_epoch=args.lr_decay_start_epoch,
                            decay_rate=args.lr_decay_rate,
                            decay_patient_n_epochs=args.lr_decay_patient_n_epochs,
                            early_stop_patient_n_epochs=args.early_stop_patient_n_epochs,
                            warmup_start_lr=args.warmup_start_lr,
                            warmup_n_steps=args.warmup_n_steps,
                            model_size=getattr(args, 'transformer_d_model', 0),
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
                amp.init()
                if args.resume:
                    load_checkpoint(args.resume, amp=amp)
        model = CustomDataParallel(model, device_ids=list(range(0, args.n_gpus)))
    else:
        model = CPUWrapperLM(model)

    # Set process name
    logger.info('PID: %s' % os.getpid())
    logger.info('USERNAME: %s' % os.uname()[1])
    logger.info('#GPU: %d' % torch.cuda.device_count())
    setproctitle(args.job_name if args.job_name else dir_name)

    # Set reporter
    reporter = Reporter(save_path)

    hidden = None
    start_time_train = time.time()
    start_time_epoch = time.time()
    start_time_step = time.time()
    accum_n_steps = 0
    n_steps = scheduler.n_steps * accum_grad_n_steps
    for ep in range(resume_epoch, args.n_epochs):
        pbar_epoch = tqdm(total=len(train_set))

        for ys_train, is_new_epoch in train_set:
            # Compute loss in the training set
            accum_n_steps += 1

            if accum_n_steps == 1:
                loss_train = 0  # moving average over gradient accumulation
            if use_apex and scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, hidden, observation = model(ys_train, state=hidden)
            else:
                loss, hidden, observation = model(ys_train, state=hidden)
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
            hidden = model.module.repackage_state(hidden)

            pbar_epoch.update(ys_train.shape[0] * (ys_train.shape[1] - 1))
            reporter.add_tensorboard_scalar('learning_rate', scheduler.lr)
            # NOTE: loss/acc/ppl are already added in the model
            reporter.step()
            n_steps += 1
            # NOTE: n_steps is different from the step counter in Noam Optimizer

            if n_steps % args.print_step == 0:
                # Compute loss in the dev set
                ys_dev = iter(dev_set).next(bptt=args.bptt)[0]
                loss, _, observation = model(ys_dev, state=None, is_eval=True)
                reporter.add(observation, is_eval=True)
                loss_dev = loss.item()
                del loss
                reporter.step(is_eval=True)

                duration_step = time.time() - start_time_step
                logger.info("step:%d(ep:%.2f) loss:%.3f(%.3f)/lr:%.5f/bs:%d (%.2f min)" %
                            (n_steps, scheduler.n_epochs + train_set.epoch_detail,
                             loss_train, loss_dev,
                             scheduler.lr, ys_train.shape[0], duration_step / 60))
                start_time_step = time.time()

            # Save figures of loss and accuracy
            if n_steps % (args.print_step * 10) == 0:
                reporter.snapshot()
                model.module.plot_attention()

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
            model.module.reset_length(args.bptt)
            ppl_dev, _ = eval_ppl([model.module], dev_set,
                                  batch_size=1, bptt=args.bptt)
            model.module.reset_length(args.bptt)
            scheduler.epoch(ppl_dev)  # lr decay
            reporter.epoch(ppl_dev, name='perplexity')  # plot
            logger.info('PPL (%s, ep:%d): %.2f' %
                        (dev_set.set, scheduler.n_epochs, ppl_dev))

            if scheduler.is_topk or is_transformer:
                # Save model
                scheduler.save_checkpoint(
                    model, save_path, remove_old=not is_transformer and args.remove_old_checkpoints, amp=amp)

                # test
                ppl_test_avg = 0.
                for eval_set in eval_sets:
                    model.module.reset_length(args.bptt)
                    ppl_test, _ = eval_ppl([model.module], eval_set,
                                           batch_size=1, bptt=args.bptt)
                    model.module.reset_length(args.bptt)
                    logger.info('PPL (%s, ep:%d): %.2f' %
                                (eval_set.set, scheduler.n_epochs, ppl_test))
                    ppl_test_avg += ppl_test
                if len(eval_sets) > 0:
                    logger.info('PPL (avg., ep:%d): %.2f' %
                                (scheduler.n_epochs, ppl_test_avg / len(eval_sets)))

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

        start_time_step = time.time()
        start_time_epoch = time.time()

    duration_train = time.time() - start_time_train
    logger.info('Total time: %.2f hour' % (duration_train / 3600))

    reporter.tf_writer.close()
    pbar_epoch.close()

    return save_path


if __name__ == '__main__':
    # Setting for profiling
    pr = cProfile.Profile()
    save_path = pr.runcall(main)
    pr.dump_stats(os.path.join(save_path, 'train.profile'))
