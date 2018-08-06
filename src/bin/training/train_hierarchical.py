#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the hierarchical ASR model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cProfile
import os
from setproctitle import setproctitle
import sys
from tensorboardX import SummaryWriter
import time
import torch
from tqdm import tqdm

torch.manual_seed(1623)
torch.cuda.manual_seed_all(1623)

sys.path.append(os.path.abspath('../../../'))
from src.models.load_model import load
from src.models.pytorch_v3.data_parallel import CustomDataParallel
from src.dataset.loader_hierarchical import Dataset as Dataset_asr
from src.dataset.loader_hierarchical_p2w import Dataset as Dataset_p2w
from src.metrics.phone import eval_phone
from src.metrics.character import eval_char
from src.metrics.word import eval_word
from src.bin.training.learning_rate_controller import Controller
from src.bin.training.reporter_hierarchical import Reporter
from src.bin.training.updater_hierarchical import Updater
from src.utils.logging import set_logger
from src.utils.directory import mkdir_join
from src.utils.config import load_config, save_config

parser = argparse.ArgumentParser()
parser.add_argument('--ngpus', type=int, default=0,
                    help='the number of GPUs (negative value indicates CPU)')
parser.add_argument('--corpus', type=str,
                    help='the name of corpus')
parser.add_argument('--train_set', type=str,
                    help='training set')
parser.add_argument('--dev_set', type=str,
                    help='development set')
parser.add_argument('--eval_sets', type=str, nargs='+',
                    help='evaluation sets')
parser.add_argument('--config_path', type=str, default=None,
                    help='path to the configuration file')
parser.add_argument('--data_save_path', type=str,
                    help='path to saved data')
parser.add_argument('--model_save_path', type=str, default=None,
                    help='path to save the model')
parser.add_argument('--saved_model_path', type=str, default=None,
                    help='path to the saved model to retrain')
args = parser.parse_args()


def main():

    # Load a ASR config file
    if args.model_save_path is not None:
        config = load_config(args.config_path)
    # NOTE: Restart from the last checkpoint
    elif args.saved_model_path is not None:
        config = load_config(os.path.join(args.saved_model_path, 'config.yml'))
    else:
        raise ValueError("Set model_save_path or saved_model_path.")

    if 'data_size' not in config.keys():
        config['data_size'] = ''

    # Automatically reduce batch size in multi-GPU setting
    if args.ngpus > 1:
        config['batch_size'] -= 20
        config['print_step'] //= args.ngpus
        if 'scheduled_sampling_max_step' in config.keys():
            config['scheduled_sampling_max_step'] //= args.ngpus

    # Load dataset
    if config['input_type'] == 'speech':
        train_set = Dataset_asr(
            corpus=args.corpus,
            data_save_path=args.data_save_path,
            model_type=config['model_type'],
            input_freq=config['input_freq'],
            use_delta=config['use_delta'],
            use_double_delta=config['use_double_delta'],
            data_type=args.train_set,
            data_size=config['data_size'] if 'data_size' in config.keys() else '',
            vocab=config['vocab'],
            label_type=config['label_type'],
            label_type_sub=config['label_type_sub'],
            batch_size=config['batch_size'] * args.ngpus,
            max_epoch=config['n_epochs'],
            max_n_frames=config['max_n_frames'] if 'max_n_frames' in config.keys() else 10000,
            min_n_frames=config['min_n_frames'] if 'min_n_frames' in config.keys() else 0,
            sort_utt=True, sort_stop_epoch=config['sort_stop_epoch'],
            tool=config['tool'], dynamic_batching=config['dynamic_batching'],
            use_ctc=config['model_type'] == 'hierarchical_ctc',
            subsampling_factor=2 ** sum(config['subsample_list']),
            use_ctc_sub=config['model_type'] == 'hierarchical_ctc' or (
                config['model_type'] == 'hierarchical_attention' and config['ctc_weight_sub'] > 0),
            subsampling_factor_sub=2 ** sum(config['subsample_list'][:config['enc_n_layers_sub'] - 1]))
        dev_set = Dataset_asr(
            corpus=args.corpus,
            data_save_path=args.data_save_path,
            model_type=config['model_type'],
            input_freq=config['input_freq'],
            use_delta=config['use_delta'],
            use_double_delta=config['use_double_delta'],
            data_type=args.dev_set,
            data_size=config['data_size'] if 'data_size' in config.keys() else '',
            vocab=config['vocab'],
            label_type=config['label_type'],
            label_type_sub=config['label_type_sub'],
            batch_size=config['batch_size'] * args.ngpus,
            shuffle=True, tool=config['tool'],
            use_ctc=config['model_type'] == 'hierarchical_ctc',
            subsampling_factor=2 ** sum(config['subsample_list']),
            use_ctc_sub=config['model_type'] == 'hierarchical_ctc' or (
                config['model_type'] == 'hierarchical_attention' and config['ctc_weight_sub'] > 0),
            subsampling_factor_sub=2 ** sum(config['subsample_list'][:config['enc_n_layers_sub'] - 1]))
        eval_sets = []
        for data_type in args.eval_sets:
            eval_sets += [Dataset_asr(
                corpus=args.corpus,
                data_save_path=args.data_save_path,
                model_type=config['model_type'],
                input_freq=config['input_freq'],
                use_delta=config['use_delta'],
                use_double_delta=config['use_double_delta'],
                data_type=data_type,
                data_size=config['data_size'] if 'data_size' in config.keys() else '',
                vocab=config['vocab'],
                label_type=config['label_type'],
                label_type_sub=config['label_type_sub'],
                batch_size=1, tool=config['tool'])]
    elif config['input_type'] == 'text':
        train_set = Dataset_p2w(
            corpus=args.corpus,
            data_save_path=args.data_save_path,
            model_type=config['model_type'],
            data_type=args.train_set,
            data_size=config['data_size'] if 'data_size' in config.keys() else '',
            vocab=config['vocab'],
            label_type_in=config['label_type_in'],
            label_type=config['label_type'],
            label_type_sub=config['label_type_sub'],
            batch_size=config['batch_size'] * args.ngpus,
            max_epoch=config['n_epochs'],
            max_n_frames=config['max_n_frames'] if 'max_n_frames' in config.keys() else 10000,
            min_n_frames=config['min_n_frames'] if 'min_n_frames' in config.keys() else 0,
            sort_utt=True, sort_stop_epoch=config['sort_stop_epoch'],
            tool=config['tool'], dynamic_batching=config['dynamic_batching'],
            use_ctc=config['model_type'] == 'ctc' or (
                config['model_type'] == 'attention' and config['ctc_weight'] > 0),
            subsampling_factor=2 ** sum(config['subsample_list']))
        dev_set = Dataset_p2w(
            corpus=args.corpus,
            data_save_path=args.data_save_path,
            model_type=config['model_type'],
            data_type=args.dev_set,
            data_size=config['data_size'] if 'data_size' in config.keys() else '',
            vocab=config['vocab'],
            label_type_in=config['label_type_in'],
            label_type=config['label_type'],
            label_type_sub=config['label_type_sub'],
            batch_size=config['batch_size'] * args.ngpus,
            shuffle=True, tool=config['tool'],
            use_ctc=config['model_type'] == 'ctc' or (
                config['model_type'] == 'attention' and config['ctc_weight'] > 0),
            subsampling_factor=2 ** sum(config['subsample_list']))
        eval_sets = []
        for data_type in args.eval_sets:
            if 'phone' in config['label_type_in']:
                continue
            eval_sets += [Dataset_p2w(
                corpus=args.corpus,
                data_save_path=args.data_save_path,
                model_type=config['model_type'],
                data_type=data_type,
                data_size=config['data_size'] if 'data_size' in config.keys() else '',
                vocab=config['vocab'],
                label_type_in=config['label_type_in'],
                label_type=config['label_type'],
                label_type_sub=config['label_type_sub'],
                batch_size=1, tool=config['tool'])]
        config['num_classes_input'] = train_set.num_classes_in

    config['n_classes'] = train_set.n_classes
    config['n_classes_sub'] = train_set.n_classes_sub

    # Load a RNNLM config file for cold fusion in the main task
    if config['rnnlm_path_cf']:
        if args.model_save_path is not None:
            config['lm_config_cf'] = load_config(
                os.path.join(config['rnnlm_path_cf'], 'config.yml'), is_eval=True)
        elif args.saved_model_path is not None:
            config = load_config(os.path.join(
                args.saved_model_path, 'config_rnnlm_cf.yml'))
        assert config['label_type'] == config['lm_config_cf']['label_type']
        config['lm_config_cf']['n_classes'] = train_set.n_classes
    else:
        config['lm_config_cf'] = None

    # Load a RNNLM config file for cold fusion in the sub task
    if config['rnnlm_path_sub_cf']:
        if args.model_save_path is not None:
            config['rnnlm_config_sub'] = load_config(
                os.path.join(config['rnnlm_path_sub_cf'], 'config.yml'), is_eval=True)
        elif args.saved_model_path is not None:
            config = load_config(os.path.join(
                args.saved_model_path, 'config_rnnlm_sub_cf.yml'))
        assert config['label_type_sub'] == config['rnnlm_config_sub']['label_type']
        config['rnnlm_config_sub']['n_classes'] = train_set.n_classes_sub
    else:
        config['rnnlm_config_sub'] = None

    # Model setting
    model = load(model_type=config['model_type'],
                 config=config,
                 backend=config['backend'])

    if args.model_save_path is not None:
        # Load pre-trained RNNLM in the main task
        if config['rnnlm_path_cf']:
            rnnlm_cf = load(model_type=config['lm_config_cf']['model_type'],
                            config=config['lm_config_cf'],
                            backend=config['lm_config_cf']['backend'])
            rnnlm_cf.load_checkpoint(save_path=config['rnnlm_path_cf'], epoch=-1)
            rnnlm_cf.flatten_parameters()

            # Fix RNNLM parameters
            for param in rnnlm_cf.parameters():
                param.requires_grad = False

            # Set pre-trained parameters
            if config['lm_config_cf']['backward']:
                model.rnnlm_0_bwd = rnnlm_cf
            else:
                model.rnnlm_0_fwd = rnnlm_cf

        # Load pre-trained RNNLM in the sub task
        if config['rnnlm_path_sub_cf']:
            rnnlm_sub = load(model_type=config['rnnlm_config_sub']['model_type'],
                             config=config['rnnlm_config_sub'],
                             backend=config['rnnlm_config_sub']['backend'])
            rnnlm_sub.load_checkpoint(
                save_path=config['rnnlm_path_sub_cf'], epoch=-1)
            rnnlm_sub.flatten_parameters()

            # Fix RNNLM parameters
            if config['rnnlm_weight_sub'] == 0:
                for param in rnnlm_sub.parameters():
                    param.requires_grad = False

            # Set pre-trained parameters
            if config['rnnlm_config_sub']['backward']:
                model.rnnlm_1_bwd = rnnlm_sub
            else:
                model.rnnlm_1_fwd = rnnlm_sub

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
        logger = set_logger(os.path.join(model.save_path, 'train.log'), key='training')

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
        logger.info("Total %.2f M parameters" % (model.total_parameters / 1000000))

        # Set optimizer
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
        logger = set_logger(os.path.join(model.save_path, 'train.log'), key='training')

        # Set optimizer
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

        if epoch >= config['convert_to_sgd_epoch']:
            model.set_optimizer(
                optimizer='sgd',
                learning_rate_init=float(config['learning_rate']),  # on-the-fly
                weight_decay=float(config['weight_decay']),
                clip_grad_norm=config['clip_grad_norm'],
                lr_schedule=False,
                factor=config['decay_rate'],
                patience_epoch=config['decay_patient_epoch'])

    else:
        raise ValueError("Set model_save_path or saved_model_path.")

    train_set.epoch = epoch - 1

    # GPU setting
    if args.ngpus >= 1:
        model = CustomDataParallel(
            model, device_ids=list(range(0, args.ngpus, 1)),
            benchmark=True)
        model.cuda()

    logger.info('PID: %s' % os.getpid())
    logger.info('USERNAME: %s' % os.uname()[1])

    # Set process name
    title = args.corpus + '_' + config['backend'] + '_' + config['model_type'] + \
        '_' + config['label_type'] + '_' + config['label_type_sub']
    if config['data_size'] != '':
        title += '_' + config['data_size']
    setproctitle(title)

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
    reporter = Reporter(model.module.save_path, max_loss=500)

    # Set the updater
    updater = Updater(config['clip_grad_norm'], config['backend'])

    # Setting for tensorboard
    if config['backend'] == 'pytorch':
        tf_writer = SummaryWriter(model.module.save_path)

    # For nested attention
    if model.module.model_type == 'nested_attention' and epoch >= config['fix_second_decoder_epoch']:
        logger.info('========== Fix second decoder ==========')
        model.module.fix_second_decoder()

    start_time_train = time.time()
    start_time_epoch = time.time()
    start_time_step = time.time()
    not_improved_epoch = 0
    loss_train_mean, loss_main_train_mean, loss_sub_train_mean = 0., 0., 0.
    acc_main_train_mean, acc_sub_train_mean = 0., 0.
    pbar_epoch = tqdm(total=len(train_set))
    while True:
        # Compute loss in the training set (including parameter update)
        batch_train, is_new_epoch = train_set.next()
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
            batch_dev = dev_set.next()[0]
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
                tf_writer.add_scalar('train/loss_main', loss_main_train_mean, step + 1)
                tf_writer.add_scalar('train/loss_sub', loss_sub_train_mean, step + 1)
                tf_writer.add_scalar('dev/loss', loss_dev, step + 1)
                tf_writer.add_scalar('dev/loss_main', loss_main_dev, step + 1)
                tf_writer.add_scalar('dev/loss_sub', loss_sub_dev, step + 1)
                for name, param in model.module.named_parameters():
                    name = name.replace('.', '/')
                    if param.grad is not None:
                        tf_writer.add_histogram(
                            name, param.data.cpu().numpy(), step + 1)
                        tf_writer.add_histogram(
                            name + '/grad', param.grad.data.cpu().numpy(), step + 1)

            duration_step = time.time() - start_time_step
            logger.info("...Step:%d(epoch:%.2f) loss:%.2f/%.2f/%.2f(%.2f/%.2f/%.2f)/acc:%.2f/%.2f(%.2f/%.2f)/lr:%.5f/batch:%d/x_lens:%d (%.2f min)" %
                        (step + 1, train_set.epoch_detail,
                         loss_train_mean, loss_main_train_mean, loss_sub_train_mean,
                         loss_dev, loss_main_dev, loss_sub_dev,
                         acc_main_train_mean, acc_sub_train_mean, acc_main_dev, acc_sub_dev,
                         learning_rate, train_set.current_batch_size,
                         max(len(x) for x in batch_train['xs']),
                         duration_step / 60))
            start_time_step = time.time()
            loss_train_mean, loss_main_train_mean, loss_sub_train_mean = 0., 0., 0.
            acc_main_train_mean, acc_sub_train_mean = 0., 0.
        step += args.ngpus

        # Save checkpoint and evaluate model per epoch
        if is_new_epoch:
            duration_epoch = time.time() - start_time_epoch
            logger.info('===== EPOCH:%d (%.2f min) =====' % (epoch, duration_epoch / 60))

            # Save fugures of loss and accuracy
            reporter.epoch()

            if epoch < config['eval_start_epoch']:
                # Save the model
                model.module.save_checkpoint(
                    model.module.save_path, epoch, step,
                    learning_rate, metric_dev_best)
            else:
                start_time_eval = time.time()
                # dev
                if model.module.main_loss_weight > 0:
                    if 'word' in config['label_type']:
                        metric_dev, _ = eval_word(
                            models=[model.module], dataset=dev_set)
                        logger.info('  WER (%s, main): %.3f %%' %
                                    (dev_set.data_type, metric_dev))
                    elif 'character' in config['label_type']:
                        wer_dev, metric_dev, _ = eval_char(
                            models=[model.module], dataset=dev_set)
                        logger.info('  WER / CER (%s): %.3f / %.3f %%' %
                                    (dev_set.data_type, wer_dev, metric_dev))
                    elif 'phone' in config['label_type_sub']:
                        metric_dev, _ = eval_phone(
                            models=[model.module], dataset=dev_set)
                        logger.info('  PER (%s): %.3f %%' %
                                    (dev_set.data_type, metric_dev))
                    else:
                        raise ValueError(config['label_type_sub'])

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
                        if model.module.main_loss_weight > 0:
                            if 'word' in config['label_type']:
                                wer_test, _ = eval_word(
                                    models=[model.module], dataset=eval_set)
                                logger.info('  WER (%s, main): %.3f %%' %
                                            (eval_set.data_type, wer_test))
                            elif 'character' in config['label_type']:
                                wer_test, cer_test, _ = eval_char(
                                    models=[model.module], dataset=eval_set)
                                logger.info('  WER / CER (%s): %.3f / %.3f %%' %
                                            (eval_set.data_type, wer_test, cer_test))
                        else:
                            if 'character' in config['label_type_sub']:
                                wer_test_sub, cer_test_sub, _ = eval_char(
                                    models=[model.module], dataset=eval_set)
                                logger.info('  WER / CER (%s, sub): %.3f / %.3f %%' %
                                            (eval_set.data_type, wer_test_sub, cer_test_sub))
                            elif 'phone' in config['label_type_sub']:
                                per_test, _ = eval_phone(
                                    models=[model.module], dataset=eval_set)
                                logger.info('  PER (%s): %.3f %%' %
                                            (eval_set.data_type, per_test))
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
                if not_improved_epoch == config['not_improved_patient_epoch']:
                    break

                if epoch == config['convert_to_sgd_epoch']:
                    # Convert to fine-tuning stage
                    model.module.set_optimizer(
                        'sgd',
                        learning_rate_init=learning_rate,
                        weight_decay=float(config['weight_decay']),
                        clip_grad_norm=config['clip_grad_norm'],
                        lr_schedule=False,
                        factor=config['decay_rate'],
                        patience_epoch=config['decay_patient_epoch'])
                    logger.info('========== Convert to SGD ==========')

                # For nested attention
                if model.module.model_type == 'nested_attention' and epoch >= config['fix_second_decoder_epoch']:
                    logger.info('========== Fix second decoder ==========')
                    model.module.fix_second_decoder()

            pbar_epoch = tqdm(total=len(train_set))

            if epoch == config['n_epochs']:
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
    with open(os.path.join(model.module.save_path, 'COMPLETE'), 'w') as f:
        f.write('')

    return model.module.save_path


if __name__ == '__main__':
    # Setting for profiling
    pr = cProfile.Profile()
    save_path = pr.runcall(main)
    pr.dump_stats(os.path.join(save_path, 'train.profile'))
