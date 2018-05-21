#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.timit.s5.exp.dataset.load_dataset import Dataset
from examples.timit.s5.exp.metrics.phone import eval_phone
from utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str,
                    help='path to saved data')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--beam_width', type=int, default=1,
                    help='the size of beam')
parser.add_argument('--length_penalty', type=float, default=0,
                    help='length penalty in beam search decoding')
parser.add_argument('--coverage_penalty', type=float, default=0,
                    help='coverage penalty in beam search decoding')

MAX_DECODE_LEN_PHONE = 71
MIN_DECODE_LEN_PHONE = 13
# NOTE*
# dev: 13-71
# test: 13-69


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    dev_data = Dataset(
        data_save_path=args.data_save_path,
        backend=params['backend'],
        input_freq=params['input_freq'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        data_type='dev',
        label_type=params['label_type'],
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, tool=params['tool'])
    test_data = Dataset(
        data_save_path=args.data_save_path,
        backend=params['backend'],
        input_freq=params['input_freq'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        data_type='test',
        label_type=params['label_type'],
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, tool=params['tool'])

    params['num_classes'] = test_data.num_classes

    # Load model
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    print('beam width: %d' % args.beam_width)

    # dev
    per_dev, df_dev = eval_phone(
        model=model,
        dataset=dev_data,
        map_file_path='./conf/phones.60-48-39.map',
        eval_batch_size=args.eval_batch_size,
        beam_width=args.beam_width,
        max_decode_len=MAX_DECODE_LEN_PHONE,
        min_decode_len=MIN_DECODE_LEN_PHONE,
        length_penalty=args.length_penalty,
        coverage_penalty=args.coverage_penalty,
        progressbar=True)
    print('  PER (dev): %.3f %%' % (per_dev * 100))
    print(df_dev)

    # test
    per_test, df_test = eval_phone(
        model=model,
        dataset=test_data,
        map_file_path='./conf/phones.60-48-39.map',
        eval_batch_size=args.eval_batch_size,
        beam_width=args.beam_width,
        max_decode_len=MAX_DECODE_LEN_PHONE,
        min_decode_len=MIN_DECODE_LEN_PHONE,
        length_penalty=args.length_penalty,
        coverage_penalty=args.coverage_penalty,
        progressbar=True)
    print('  PER (test): %.3f %%' % (per_test * 100))
    print(df_test)

    with open(join(args.model_path, 'RESULTS'), 'w') as f:
        f.write('beam width: %d\n' % args.beam_width)
        f.write('  PER (dev): %.3f %%' % (per_test * 100))
        f.write('  PER (test): %.3f %%' % (per_test * 100))


if __name__ == '__main__':
    main()
