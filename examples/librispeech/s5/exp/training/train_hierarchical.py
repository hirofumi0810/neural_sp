#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the hierarchical ASR model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
import sys
import time
from setproctitle import setproctitle
import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm
import cProfile

torch.manual_seed(1623)
torch.cuda.manual_seed_all(1623)

sys.path.append(os.path.abspath('../../../'))
from models.load_model import load
from examples.librispeech.s5.exp.dataset.load_dataset_hierarchical import Dataset
from examples.librispeech.s5.exp.metrics.character import eval_char
from examples.librispeech.s5.exp.metrics.word import eval_word
from utils.training.learning_rate_controller import Controller
from utils.training.reporter_hierarchical import Reporter
from utils.training.updater_hierarchical import Updater
from utils.training.logging import set_logger
from utils.directory import mkdir_join
from utils.config import load_config, save_config

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1,
                    help='the index of GPU (negative value indicates CPU)')
parser.add_argument('--config_path', type=str, default=None,
                    help='path to the configuration file')
parser.add_argument('--data_save_path', type=str,
                    help='path to saved data')
parser.add_argument('--model_save_path', type=str, default=None,
                    help='path to save the model')
parser.add_argument('--saved_model_path', type=str, default=None,
                    help='path to the saved model to retrain')

MAX_DECODE_LEN_WORD = 200
MAX_DECODE_LEN_CHAR = 600


def main():

    args = parser.parse_args()

    # Load a ASR config file
    if args.model_save_path is not None:
        config = load_config(args.config_path)
    # NOTE: Restart from the last checkpoint
    elif args.saved_model_path is not None:
        config = load_config(os.path.join(args.saved_model_path, 'config.yml'))
    else:
        raise ValueError("Set model_save_path or saved_model_path.")
    config['data_size'] = str(config['data_size'])

    # Load dataset
    train_data = Dataset(
        data_save_path=args.data_save_path,
        input_freq=config['input_freq'],
        use_delta=config['use_delta'],
        use_double_delta=config['use_double_delta'],
        data_type='train', data_size=config['data_size'],
        label_type=config['label_type'],
        label_type_sub=config['label_type_sub'],
        batch_size=config['batch_size'], max_epoch=config['num_epoch'],
        max_frame_num=config['max_frame_num'],
        min_frame_num=config['min_frame_num'],
        sort_utt=True, sort_stop_epoch=config['sort_stop_epoch'],
        tool=config['tool'], dynamic_batching=config['dynamic_batching'])
    dev_data = Dataset(
        data_save_path=args.data_save_path,
        input_freq=config['input_freq'],
        use_delta=config['use_delta'],
        use_double_delta=config['use_double_delta'],
        data_type='dev_clean', data_size=config['data_size'],
        label_type=config['label_type'],
        label_type_sub=config['label_type_sub'],
        batch_size=config['batch_size'],
        shuffle=True, tool=config['tool'])
    test_data = Dataset(
        data_save_path=args.data_save_path,
        input_freq=config['input_freq'],
        use_delta=config['use_delta'],
        use_double_delta=config['use_double_delta'],
        data_type='test_clean', data_size=config['data_size'],
        label_type=config['label_type'],
        label_type_sub=config['label_type_sub'],
        batch_size=1, tool=config['tool'])
    config['num_classes'] = train_data.num_classes
    config['num_classes_sub'] = train_data.num_classes_sub

    # TODO: add mult-level cold fusion

    # Model setting
    model = load(model_type=config['model_type'],
                 config=config,
                 backend=config['backend'])

    if args.model_save_path is not None:
        # Set save path
        save_path = mkdir_join(
            args.model_save_path, config['backend'],
            config['model_type'],
            config['label_type'] + '_' + config['label_type_sub'],
            config['data_size'], model.name)
        model.set_save_path(save_path)

        # Save the config file
        save_config(config_path=args.config_path, save_path=model.save_path)

        # Setting for logging
        logger = set_logger(model.save_path)

        for k, v in sorted(config.items(), key=lambda x: x[0]):
            logger.info('%s: %s' % (k, str(v)))

        if os.path.isdir(config['pretrained_model_path']):
            # NOTE: Start training from the pre-trained model
            model.load_checkpoint(
                save_path=config['pretrained_model_path'], epoch=-1,
                load_pretrained_model=True)

        # Count total parameters
        for name in sorted(list(model.num_params_dict.keys())):
            num_params = model.num_params_dict[name]
            logger.info("%s %d" % (name, num_params))
        logger.info("Total %.2f M parameters" %
                    (model.total_parameters / 1000000))

        # Define optimizer
        model.set_optimizer(
            optimizer=config['optimizer'],
            learning_rate_init=float(config['learning_rate']),
            weight_decay=float(config['weight_decay']),
            clip_grad_norm=config['clip_grad_norm'],
            lr_schedule=False,
            factor=config['decay_rate'],
            patience_epoch=config['decay_patient_epoch'])

        epoch, step = 1, 0
        learning_rate = float(config['learning_rate'])
        metric_dev_best = 100

    # NOTE: Restart from the last checkpoint
    elif args.saved_model_path is not None:
        # Set save path
        model.save_path = args.saved_model_path

        # Setting for logging
        logger = set_logger(model.save_path)

        # Define optimizer
        model.set_optimizer(
            optimizer=config['optimizer'],
            learning_rate_init=float(config['learning_rate']),  # on-the-fly
            weight_decay=float(config['weight_decay']),
            clip_grad_norm=config['clip_grad_norm'],
            lr_schedule=False,
            factor=config['decay_rate'],
            patience_epoch=config['decay_patient_epoch'])

        # Restore the last saved model
        epoch, step, learning_rate, metric_dev_best = model.load_checkpoint(
            save_path=args.saved_model_path, epoch=-1, restart=True)

    else:
        raise ValueError("Set model_save_path or saved_model_path.")

    train_data.epoch = epoch - 1

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    logger.info('PID: %s' % os.getpid())
    logger.info('USERNAME: %s' % os.uname()[1])

    # Set process name
    setproctitle('libri_' + config['backend'] + '_' + config['model_type'] + '_' +
                 config['label_type'] + '_' + config['label_type_sub'] + '_' + config['data_size'])

    # Set learning rate controller
    lr_controller = Controller(
        learning_rate_init=learning_rate,
        backend=config['backend'],
        decay_type=config['decay_type'],
        decay_start_epoch=config['decay_start_epoch'],
        decay_rate=config['decay_rate'],
        decay_patient_epoch=config['decay_patient_epoch'],
        lower_better=True,
        best_value=metric_dev_best)

    # Set reporter
    reporter = Reporter(model.save_path)

    # Set the updater
    updater = Updater(config['clip_grad_norm'], config['backend'])

    # Setting for tensorboard
    if config['backend'] == 'pytorch':
        tf_writer = SummaryWriter(model.save_path)

    start_time_train = time.time()
    start_time_epoch = time.time()
    start_time_step = time.time()
    not_improved_epoch = 0
    loss_train_mean, loss_main_train_mean, loss_sub_train_mean = 0., 0., 0.
    acc_main_train_mean, acc_sub_train_mean = 0., 0.
    pbar_epoch = tqdm(total=len(train_data))
    while True:
        # Compute loss in the training set (including parameter update)
        batch_train, is_new_epoch = train_data.next()
        model, loss_train, loss_main_train, loss_sub_train, acc_main_train, acc_sub_train = updater(
            model, batch_train)
        loss_train_mean += loss_train
        loss_main_train_mean += loss_main_train
        loss_sub_train_mean += loss_sub_train
        acc_main_train_mean += acc_main_train
        acc_sub_train_mean += acc_sub_train
        pbar_epoch.update(len(batch_train['xs']))

        if (step + 1) % config['print_step'] == 0:
            # Compute loss in the dev set
            batch_dev = dev_data.next()[0]
            model, loss_dev, loss_main_dev, loss_sub_dev, acc_main_dev, acc_sub_dev = updater(
                model, batch_dev, is_eval=True)

            loss_train_mean /= config['print_step']
            loss_main_train_mean /= config['print_step']
            loss_sub_train_mean /= config['print_step']
            acc_main_train_mean /= config['print_step']
            acc_sub_train_mean /= config['print_step']
            reporter.step(step, loss_train_mean, loss_main_train_mean, loss_sub_train_mean,
                          loss_dev, loss_main_dev, loss_sub_dev,
                          acc_main_train_mean, acc_sub_train_mean,
                          acc_main_dev, acc_sub_dev)

            # Logging by tensorboard
            if config['backend'] == 'pytorch':
                tf_writer.add_scalar('train/loss', loss_train_mean, step + 1)
                tf_writer.add_scalar(
                    'train/loss_main', loss_main_train_mean, step + 1)
                tf_writer.add_scalar(
                    'train/loss_sub', loss_sub_train_mean, step + 1)
                tf_writer.add_scalar('dev/loss', loss_dev, step + 1)
                tf_writer.add_scalar('dev/loss_main', loss_main_dev, step + 1)
                tf_writer.add_scalar('dev/loss_sub', loss_sub_dev, step + 1)
                for name, param in model.named_parameters():
                    name = name.replace('.', '/')
                    tf_writer.add_histogram(
                        name, param.data.cpu().numpy(), step + 1)
                    if param.grad is not None:
                        tf_writer.add_histogram(
                            name + '/grad', param.grad.data.cpu().numpy(), step + 1)

            duration_step = time.time() - start_time_step
            logger.info("...Step:%d(epoch:%.2f) loss:%.2f/%.2f/%.2f(%.2f/%.2f/%.2f)/acc:%.2f/%.2f(%.2f/%.2f)/lr:%.5f/batch:%d/x_lens:%d (%.2f min)" %
                        (step + 1, train_data.epoch_detail,
                         loss_train_mean, loss_main_train_mean, loss_sub_train_mean,
                         loss_dev, loss_main_dev, loss_sub_dev,
                         acc_main_train_mean, acc_sub_train_mean, acc_main_dev, acc_sub_dev,
                         learning_rate, train_data.current_batch_size,
                         max(len(x) for x in batch_train['xs']),
                         duration_step / 60))
            start_time_step = time.time()
            loss_train_mean, loss_main_train_mean, loss_sub_train_mean = 0., 0., 0.
            acc_main_train_mean, acc_sub_train_mean = 0., 0.
        step += 1

        # Save checkpoint and evaluate model per epoch
        if is_new_epoch:
            duration_epoch = time.time() - start_time_epoch
            logger.info('===== EPOCH:%d (%.2f min) =====' %
                        (epoch, duration_epoch / 60))

            # Save fugures of loss and accuracy
            reporter.epoch()

            if epoch < config['eval_start_epoch']:
                # Save the model
                model.save_checkpoint(model.save_path, epoch, step,
                                      learning_rate, metric_dev_best)
            else:
                start_time_eval = time.time()
                # dev
                if model.main_loss_weight > 0:
                    metric_dev, _ = eval_word(
                        models=[model],
                        dataset=dev_data,
                        eval_batch_size=1,
                        beam_width=1,
                        max_decode_len=MAX_DECODE_LEN_WORD,
                        max_decode_len_sub=MAX_DECODE_LEN_CHAR)
                    logger.info('  WER (%s, main): %.3f %%' %
                                (dev_data.data_type, metric_dev))
                else:
                    wer_dev_sub, metric_dev, _ = eval_char(
                        models=[model],
                        dataset=dev_data,
                        eval_batch_size=1,
                        beam_width=1,
                        max_decode_len=MAX_DECODE_LEN_CHAR)
                    logger.info('  WER / CER (%s, sub): %.3f / %.3f %%' %
                                (dev_data.data_type, wer_dev_sub, metric_dev))

                if metric_dev < metric_dev_best:
                    metric_dev_best = metric_dev
                    not_improved_epoch = 0
                    logger.info('||||| Best Score |||||')

                    # Save the model
                    model.save_checkpoint(model.save_path, epoch, step,
                                          learning_rate, metric_dev_best)

                    # test
                    if model.main_loss_weight > 0:
                        wer_test, _ = eval_word(
                            models=[model],
                            dataset=test_data,
                            eval_batch_size=1,
                            beam_width=1,
                            max_decode_len=MAX_DECODE_LEN_WORD,
                            max_decode_len_sub=MAX_DECODE_LEN_CHAR)
                        logger.info('  WER (%s, main): %.3f %%' %
                                    (test_data.data_type, wer_test))
                    else:
                        wer_test_sub, cer_test_sub, _ = eval_char(
                            models=[model],
                            dataset=test_data,
                            eval_batch_size=1,
                            beam_width=1,
                            max_decode_len=MAX_DECODE_LEN_CHAR)
                        logger.info('  WER / CER (%s, sub): %.3f / %.3f %%' %
                                    (test_data.data_type, wer_test_sub, cer_test_sub))
                else:
                    not_improved_epoch += 1

                duration_eval = time.time() - start_time_eval
                logger.info('Evaluation time: %.2f min' % (duration_eval / 60))

                # Early stopping
                if not_improved_epoch == config['not_improved_patient_epoch']:
                    break

                # Update learning rate
                model.optimizer, learning_rate = lr_controller.decay_lr(
                    optimizer=model.optimizer,
                    learning_rate=learning_rate,
                    epoch=epoch,
                    value=metric_dev)

                if epoch == config['convert_to_sgd_epoch']:
                    # For nested attention
                    if 'fix_second_decoder' in config.keys() and config['fix_second_decoder'] and model.model_type == 'nested_attention':
                        logger.info('========== Fix second decoder ==========')
                        model.fix_second_decoder()

                    # Convert to fine-tuning stage
                    model.set_optimizer(
                        'sgd',
                        learning_rate_init=learning_rate,
                        weight_decay=float(config['weight_decay']),
                        clip_grad_norm=config['clip_grad_norm'],
                        lr_schedule=False,
                        factor=config['decay_rate'],
                        patience_epoch=config['decay_patient_epoch'])
                    logger.info('========== Convert to SGD ==========')

                    # Inject Gaussian noise to all parameters
                    if float(config['weight_noise_std']) > 0:
                        model.weight_noise_injection = True

            pbar_epoch = tqdm(total=len(train_data))

            if epoch == config['num_epoch']:
                break

            start_time_step = time.time()
            start_time_epoch = time.time()
            epoch += 1

    duration_train = time.time() - start_time_train
    logger.info('Total time: %.2f hour' % (duration_train / 3600))

    if config['backend'] == 'pytorch':
        tf_writer.close()
    pbar_epoch.close()

    # Training was finished correctly
    with open(os.path.join(model.save_path, 'COMPLETE'), 'w') as f:
        f.write('')

    return model.save_path


if __name__ == '__main__':
    # Setting for profiling
    pr = cProfile.Profile()
    save_path = pr.runcall(main)
    pr.dump_stats(os.path.join(save_path, 'train.profile'))
