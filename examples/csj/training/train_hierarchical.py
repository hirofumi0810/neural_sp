#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the model (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import time
from setproctitle import setproctitle
import yaml
import shutil
import argparse
from tensorboardX import SummaryWriter

sys.path.append(abspath('../../../'))
from models.pytorch.load_model import load
from examples.csj.data.load_dataset_hierarchical import Dataset
from examples.csj.metrics.cer import do_eval_cer
from examples.csj.metrics.wer import do_eval_wer
from utils.training.learning_rate_controller import Controller
from utils.training.plot import plot_loss
from utils.training.training_loop import train_hierarchical_step
from utils.directory import mkdir_join, mkdir
from utils.io.variable import np2var, var2np

MAX_DECODE_LEN_POS = 60
MAX_DECODE_LEN_WORD = 60
MAX_DECODE_LEN_CHAR = 100


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str,
                    help='path to the configuration file')
parser.add_argument('--model_save_path', type=str,
                    help='path to save the model')


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    with open(args.config_path, "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Get voabulary number (excluding blank, <SOS>, <EOS> classes)
    with open('../metrics/vocab_num.yml', "r") as f:
        vocab_num = yaml.load(f)
        params['num_classes'] = vocab_num[params['data_size']
                                          ][params['label_type']]
        params['num_classes_sub'] = vocab_num[params['data_size']
                                              ][params['label_type_sub']]

    # Model setting
    model = load(model_type=params['model_type'], params=params)

    # Set process name
    setproctitle('csj_' + params['model_type'] + '_' +
                 params['label_type'] + '_' + params['label_type_sub'] + '_' + params['data_size'])

    # Set save path
    save_path = mkdir_join(
        args.model_save_path, params['model_type'],
        params['label_type'] + '_' + params['label_type_sub'],
        params['data_size'], model.name)
    model.set_save_path(save_path)

    # Save config file
    shutil.copyfile(args.config_path, join(model.save_path, 'config.yml'))

    sys.stdout = open(join(model.save_path, 'train.log'), 'w')
    # TODO(hirofumi): change to logger

    # Load dataset
    vocab_file_path = '../metrics/vocab_files/' + \
        params['label_type'] + '_' + params['data_size'] + '.txt'
    vocab_file_path_sub = '../metrics/vocab_files/' + \
        params['label_type_sub'] + '_' + params['data_size'] + '.txt'
    train_data = Dataset(
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='train', data_size=params['data_size'],
        label_type=params['label_type'],
        label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=params['batch_size'], max_epoch=params['num_epoch'],
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, sort_stop_epoch=params['sort_stop_epoch'],
        save_format=params['save_format'], num_enque=None)
    dev_data = Dataset(
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='dev', data_size=params['data_size'],
        label_type=params['label_type'],
        label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=True, save_format=params['save_format'])
    eval1_data = Dataset(
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='eval1', data_size=params['data_size'],
        label_type=params['label_type'],
        label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, save_format=params['save_format'])
    eval2_data = Dataset(
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='eval2', data_size=params['data_size'],
        label_type=params['label_type'],
        label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, save_format=params['save_format'])
    eval3_data = Dataset(
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='eval3', data_size=params['data_size'],
        label_type=params['label_type'],
        label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, save_format=params['save_format'])

    # Count total parameters
    for name in sorted(list(model.num_params_dict.keys())):
        num_params = model.num_params_dict[name]
        print("%s %d" % (name, num_params))
    print("Total %.3f M parameters" % (model.total_parameters / 1000000))

    # Define optimizer
    optimizer, _ = model.set_optimizer(
        params['optimizer'],
        learning_rate_init=float(params['learning_rate']),
        weight_decay=float(params['weight_decay']),
        lr_schedule=False,
        factor=params['decay_rate'],
        patience_epoch=params['decay_patient_epoch'])

    # Define learning rate controller
    lr_controller = Controller(
        learning_rate_init=params['learning_rate'],
        decay_start_epoch=params['decay_start_epoch'],
        decay_rate=params['decay_rate'],
        decay_patient_epoch=params['decay_patient_epoch'],
        lower_better=True)

    # Initialize parameters
    model.init_weights()

    # GPU setting
    model.set_cuda(deterministic=False)

    # Setting for tensorboard
    tf_writer = SummaryWriter(model.save_path)

    # Train model
    csv_steps, csv_loss_train, csv_loss_dev = [], [], []
    start_time_train = time.time()
    start_time_epoch = time.time()
    start_time_step = time.time()
    wer_dev_best = 1
    not_improved_epoch = 0
    learning_rate = float(params['learning_rate'])
    loss_val_train_mean, loss_main_val_train_mean, loss_sub_val_train_mean = 0., 0., 0.
    for step, (batch, is_new_epoch) in enumerate(train_data):

        # Compute loss in the training set (including parameter update)
        model, optimizer, loss_val_train, loss_main_val_train, loss_sub_val_train = train_hierarchical_step(
            model, optimizer, batch, params['clip_grad_norm'])
        loss_val_train_mean += loss_val_train
        loss_main_val_train_mean += loss_main_val_train
        loss_sub_val_train_mean += loss_sub_val_train

        # Inject Gaussian noise to all parameters
        if float(params['weight_noise_std']) > 0 and learning_rate < float(params['learning_rate']):
            model.weight_noise_injection = True

        if (step + 1) % params['print_step'] == 0:

            # Compute loss in the dev set
            inputs, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub, _ = dev_data.next()[
                0]
            loss_dev, loss_main_dev, loss_sub_dev = model(
                inputs, labels, labels_sub, inputs_seq_len,
                labels_seq_len, labels_seq_len_sub, is_eval=True)

            loss_val_train_mean /= params['print_step']
            loss_main_val_train_mean /= params['print_step']
            loss_sub_val_train_mean /= params['print_step']
            loss_val_dev = loss_dev.data[0]
            loss_main_val_dev = loss_main_dev.data[0]
            loss_sub_val_dev = loss_sub_dev.data[0]
            csv_steps.append(step)
            csv_loss_train.append(loss_val_train_mean)
            csv_loss_dev.append(loss_val_dev)

            # Logging by tensorboard
            tf_writer.add_scalar('train/loss', loss_val_train_mean, step + 1)
            tf_writer.add_scalar(
                'train/loss_main', loss_main_val_train, step + 1)
            tf_writer.add_scalar(
                'train/loss_sub', loss_sub_val_train, step + 1)
            tf_writer.add_scalar('dev/loss', loss_val_dev, step + 1)
            tf_writer.add_scalar(
                'dev/loss_main', loss_main_val_dev, step + 1)
            tf_writer.add_scalar(
                'dev/loss_sub', loss_sub_val_dev, step + 1)
            for name, param in model.named_parameters():
                name = name.replace('.', '/')
                tf_writer.add_histogram(name, var2np(param.clone()), step + 1)
                if param.grad is not None:
                    tf_writer.add_histogram(
                        name + '/grad', var2np(param.grad.clone()), step + 1)
                # TODO: fix this

            duration_step = time.time() - start_time_step
            print("Step %d (epoch: %.3f): loss = %.3f/%.3f (%.3f/%.3f) / lr = %.5f (%.3f min)" %
                  (step + 1, train_data.epoch_detail,
                   loss_main_val_train, loss_sub_val_train,
                   loss_main_val_dev, loss_sub_val_dev,
                   learning_rate, duration_step / 60))
            sys.stdout.flush()
            start_time_step = time.time()
            loss_val_train_mean, loss_main_val_train_mean, loss_sub_val_train_mean = 0., 0., 0.

        # Save checkpoint and evaluate model per epoch
        if is_new_epoch:
            duration_epoch = time.time() - start_time_epoch
            print('-----EPOCH:%d (%.3f min)-----' %
                  (train_data.epoch, duration_epoch / 60))

            # Save fugure of loss
            plot_loss(csv_loss_train, csv_loss_dev, csv_steps,
                      save_path=model.save_path)

            if train_data.epoch < params['eval_start_epoch']:
                # Save the model
                saved_path = model.save_checkpoint(
                    model.save_path, epoch=train_data.epoch)
                print("=> Saved checkpoint (epoch:%d): %s" %
                      (train_data.epoch, saved_path))
            else:
                start_time_eval = time.time()
                print('=== Dev Data Evaluation ===')
                if param['label_type'] == 'pos':
                    wer_dev_epoch = do_eval_wer(
                        model=model,
                        model_type=params['model_type'],
                        dataset=dev_data,
                        label_type=params['label_type_sub'],
                        data_size=params['data_size'],
                        beam_width=1,
                        max_decode_len=MAX_DECODE_LEN_WORD,
                        eval_batch_size=1,
                        is_pos=True)
                else:
                    wer_dev_epoch = do_eval_wer(
                        model=model,
                        model_type=params['model_type'],
                        dataset=dev_data,
                        label_type=params['label_type'],
                        data_size=params['data_size'],
                        beam_width=1,
                        max_decode_len=MAX_DECODE_LEN_WORD,
                        eval_batch_size=1)
                print('  WER: %f %%' % (wer_dev_epoch * 100))

                if wer_dev_epoch < wer_dev_best:
                    wer_dev_best = wer_dev_epoch
                    not_improved_epoch = 0
                    print('■■■ ↑Best Score (WER)↑ ■■■')

                    # Save the model
                    saved_path = model.save_checkpoint(
                        model.save_path, epoch=train_data.epoch)
                    print("=> Saved checkpoint (epoch:%d): %s" %
                          (train_data.epoch, saved_path))

                    print('=== Test Data Evaluation ===')
                    if param['label_type'] == 'pos':
                        wer_eval1 = do_eval_wer(
                            model=model,
                            model_type=params['model_type'],
                            dataset=eval1_data,
                            label_type=params['label_type_sub'],
                            data_size=params['data_size'],
                            beam_width=args.beam_width,
                            max_decode_len=MAX_DECODE_LEN_WORD,
                            eval_batch_size=1,
                            is_pos=True)
                        wer_eval2 = do_eval_wer(
                            model=model,
                            model_type=params['model_type'],
                            dataset=eval2_data,
                            label_type=params['label_type_sub'],
                            data_size=params['data_size'],
                            beam_width=args.beam_width,
                            max_decode_len=MAX_DECODE_LEN_WORD,
                            eval_batch_size=1,
                            is_pos=True)
                        wer_eval3 = do_eval_wer(
                            model=model,
                            model_type=params['model_type'],
                            dataset=eval3_data,
                            label_type=params['label_type_sub'],
                            data_size=params['data_size'],
                            beam_width=args.beam_width,
                            max_decode_len=MAX_DECODE_LEN_WORD,
                            eval_batch_size=1,
                            is_pos=True)
                    else:
                        wer_eval1 = do_eval_wer(
                            model=model,
                            model_type=params['model_type'],
                            dataset=eval1_data,
                            label_type=params['label_type'],
                            data_size=params['data_size'],
                            beam_width=args.beam_width,
                            max_decode_len=MAX_DECODE_LEN_WORD,
                            eval_batch_size=1)
                        wer_eval2 = do_eval_wer(
                            model=model,
                            model_type=params['model_type'],
                            dataset=eval2_data,
                            label_type=params['label_type'],
                            data_size=params['data_size'],
                            beam_width=args.beam_width,
                            max_decode_len=MAX_DECODE_LEN_WORD,
                            eval_batch_size=1)
                        wer_eval3 = do_eval_wer(
                            model=model,
                            model_type=params['model_type'],
                            dataset=eval3_data,
                            label_type=params['label_type'],
                            data_size=params['data_size'],
                            beam_width=args.beam_width,
                            max_decode_len=MAX_DECODE_LEN_WORD,
                            eval_batch_size=1)
                    print('  WER (eval1, main): %f %%' % (wer_eval1 * 100))
                    print('  WER (eval2, main): %f %%' % (wer_eval2 * 100))
                    print('  WER (eval3, main): %f %%' % (wer_eval3 * 100))
                    print('  WER (mean, main): %f %%' %
                          ((wer_eval1 + wer_eval2 + wer_eval3) * 100 / 3))
                else:
                    not_improved_epoch += 1

                duration_eval = time.time() - start_time_eval
                print('Evaluation time: %.3f min' % (duration_eval / 60))

                # Early stopping
                if not_improved_epoch == params['not_improved_patient_epoch']:
                    break

                # Update learning rate
                optimizer, learning_rate = lr_controller.decay_lr(
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    epoch=train_data.epoch,
                    value=wer_dev_epoch)

            start_time_step = time.time()
            start_time_epoch = time.time()

    duration_train = time.time() - start_time_train
    print('Total time: %.3f hour' % (duration_train / 3600))

    # Training was finished correctly
    with open(join(model.save_path, 'complete.txt'), 'w') as f:
        f.write('')

    tf_writer.close()


if __name__ == '__main__':
    main()
