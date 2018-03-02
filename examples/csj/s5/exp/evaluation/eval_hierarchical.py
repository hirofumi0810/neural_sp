#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained model (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.csj.s5.exp.dataset.load_dataset_hierarchical import Dataset
from examples.csj.s5.exp.metrics.cer import do_eval_cer
from examples.csj.s5.exp.metrics.wer import do_eval_wer
from utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--beam_width', type=int, default=1,
                    help='beam_width (int, optional): beam width for beam search.' +
                    ' 1 disables beam search, which mean greedy decoding.')
parser.add_argument('--max_decode_len', type=int, default=60,
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')
parser.add_argument('--max_decode_len_sub', type=int, default=100,
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')
parser.add_argument('--data_save_path', type=str, help='path to saved data')


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    eval1_data = Dataset(
        data_save_path=args.data_save_path,
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        data_type='eval1', data_size=params['data_size'],
        label_type=params['label_type'], label_type_sub=params['label_type_sub'],
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, tool=params['tool'])
    eval2_data = Dataset(
        data_save_path=args.data_save_path,
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        data_type='eval2', data_size=params['data_size'],
        label_type=params['label_type'], label_type_sub=params['label_type_sub'],
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, tool=params['tool'])
    eval3_data = Dataset(
        data_save_path=args.data_save_path,
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        data_type='eval3', data_size=params['data_size'],
        label_type=params['label_type'], label_type_sub=params['label_type_sub'],
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, tool=params['tool'])
    params['num_classes'] = eval1_data.num_classes
    params['num_classes_sub'] = eval1_data.num_classes_sub

    # Load model
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    wer_eval1, df_wer_eval1 = do_eval_wer(
        model=model,
        dataset=eval1_data,
        beam_width=args.beam_width,
        max_decode_len=args.max_decode_len,
        eval_batch_size=args.eval_batch_size,
        progressbar=True)
    print('  WER (eval1, main): %f %%' % (wer_eval1 * 100))
    print(df_wer_eval1)
    wer_eval2, df_wer_eval2 = do_eval_wer(
        model=model,
        dataset=eval2_data,
        beam_width=args.beam_width,
        max_decode_len=args.max_decode_len,
        eval_batch_size=args.eval_batch_size,
        progressbar=True)
    print('  WER (eval2, main): %f %%' % (wer_eval2 * 100))
    print(df_wer_eval2)
    wer_eval3, df_wer_eval3 = do_eval_wer(
        model=model,
        dataset=eval3_data,
        beam_width=args.beam_width,
        max_decode_len=args.max_decode_len,
        eval_batch_size=args.eval_batch_size,
        progressbar=True)
    print('  WER (eval3, main): %f %%' % (wer_eval3 * 100))
    print(df_wer_eval3)

    print('  WER (mean, main): %f %%' %
          ((wer_eval1 + wer_eval2 + wer_eval3) * 100 / 3))

    cer_eval1, wer_eval1_sub, df_cer_eval1 = do_eval_cer(
        model=model,
        dataset=eval1_data,
        beam_width=args.beam_width,
        max_decode_len=args.max_decode_len_sub,
        eval_batch_size=args.eval_batch_size,
        progressbar=True)
    print('  CER (eval1, sub): %f %%' % (cer_eval1 * 100))
    if params['label_type_sub'] == 'kanji_divide':
        print('  WER (eval1, sub): %f %%' % (wer_eval1_sub * 100))
    print(df_cer_eval1)
    cer_eval2, wer_eval2_sub, df_cer_eval2 = do_eval_cer(
        model=model,
        dataset=eval2_data,
        beam_width=args.beam_width,
        max_decode_len=args.max_decode_len_sub,
        eval_batch_size=args.eval_batch_size,
        progressbar=True)
    print('  CER (eval2, sub): %f %%' % (cer_eval2 * 100))
    if params['label_type_sub'] == 'kanji_divide':
        print('  WER (eval2, sub): %f %%' % (wer_eval2_sub * 100))
    print(df_cer_eval2)
    cer_eval3, wer_eval3_sub, df_cer_eval3 = do_eval_cer(
        model=model,
        dataset=eval3_data,
        beam_width=args.beam_width,
        max_decode_len=args.max_decode_len_sub,
        eval_batch_size=args.eval_batch_size,
        progressbar=True)
    print('  CER (eval3, sub): %f %%' % (cer_eval3 * 100))
    if params['label_type_sub'] == 'kanji_divide':
        print('  WER (eval3, sub): %f %%' % (wer_eval3_sub * 100))
    print(df_cer_eval3)

    print('  CER (mean, sub): %f %%' %
          ((cer_eval1 + cer_eval2 + cer_eval3) * 100 / 3))
    if params['label_type_sub'] == 'kanji_divide':
        print('  WER (mean, sub): %f %%' %
              ((wer_eval1_sub + wer_eval2_sub + wer_eval3_sub) * 100 / 3))


if __name__ == '__main__':
    main()
