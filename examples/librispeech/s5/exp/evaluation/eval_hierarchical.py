#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained hierarchical model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.librispeech.data.load_dataset_hierarchical import Dataset
from examples.librispeech.metrics.cer import do_eval_cer
from examples.librispeech.metrics.wer import do_eval_wer
from utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--beam_width', type=int, default=1,
                    help='beam_width (int, optional): beam width for beam search.' +
                    ' 1 disables beam search, which mean greedy decoding.')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--max_decode_len', type=int, default=100,
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')
parser.add_argument('--max_decode_len_sub', type=int, default=600,
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    vocab_file_path = '../metrics/vocab_files/' + \
        params['label_type'] + '_' + params['data_size'] + '.txt'
    vocab_file_path_sub = '../metrics/vocab_files/' + \
        params['label_type_sub'] + '_' + params['data_size'] + '.txt'
    test_clean_data = Dataset(
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        data_type='test_clean', data_size=params['data_size'],
        label_type=params['label_type'], label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, save_format=params['save_format'])
    test_other_data = Dataset(
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        data_type='test_other', data_size=params['data_size'],
        label_type=params['label_type'], label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, save_format=params['save_format'])
    params['num_classes'] = test_clean_data.num_classes
    params['num_classes_sub'] = test_clean_data.num_classes_sub

    # Load model
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    wer_test_clean = do_eval_wer(
        model=model,
        dataset=test_clean_data,
        beam_width=args.beam_width,
        max_decode_len=args.max_decode_len,
        eval_batch_size=args.eval_batch_size,
        progressbar=True)
    print('  WER (clean, main): %f %%' % (wer_test_clean * 100))
    wer_test_other = do_eval_wer(
        model=model,
        dataset=test_other_data,
        beam_width=args.beam_width,
        max_decode_len=args.max_decode_len,
        eval_batch_size=args.eval_batch_size,
        progressbar=True)
    print('  WER (other, main): %f %%' % (wer_test_other * 100))
    print('  WER (mean, main): %f %%' %
          ((wer_test_clean + wer_test_other) * 100 / 2))

    cer_test_clean, _ = do_eval_cer(
        model=model,
        dataset=test_clean_data,
        beam_width=args.beam_width,
        max_decode_len=args.max_decode_len_sub,
        eval_batch_size=args.eval_batch_size,
        progressbar=True)
    print('  CER (clean, sub): %f %%' % (cer_test_clean * 100))
    cer_test_other, _ = do_eval_cer(
        model=model,
        dataset=test_other_data,
        beam_width=args.beam_width,
        max_decode_len=args.max_decode_len_sub,
        eval_batch_size=args.eval_batch_size,
        progressbar=True)
    print('  CER (other, sub): %f %%' % (cer_test_other * 100))
    print('  CER (mean, sub): %f %%' %
          ((cer_test_clean + cer_test_other) * 100 / 2))


if __name__ == '__main__':
    main()
