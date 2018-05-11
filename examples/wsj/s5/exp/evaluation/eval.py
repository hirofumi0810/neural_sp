#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained model (WSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.wsj.s5.exp.dataset.load_dataset import Dataset
from examples.wsj.s5.exp.metrics.wer import do_eval_wer
from utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str,
                    help='path to saved data')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--beam_width', type=int, default=1,
                    help='the size of beam')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--length_penalty', type=float,
                    help='length penalty in beam search decodding')

MAX_DECODE_LEN_CHAR = 300


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    test_data = Dataset(
        data_save_path=args.data_save_path,
        backend=params['backend'],
        input_freq=params['input_freq'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        # data_type='test_dev93',
        data_type='test_eval92',
        data_size=params['data_size'],
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

    wer_eval92, cer_eval92, df_wer_cer = do_eval_wer(
        models=[model],
        dataset=test_data,
        beam_width=args.beam_width,
        max_decode_len=MAX_DECODE_LEN_CHAR,
        eval_batch_size=args.eval_batch_size,
        progressbar=True)
    print('  WER / CER (eval92): %.3f / %.3f %%' %
          ((wer_eval92 * 100), (cer_eval92 * 100)))
    print(df_wer_cer)


if __name__ == '__main__':
    main()
