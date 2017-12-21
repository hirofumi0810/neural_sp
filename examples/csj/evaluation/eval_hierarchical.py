#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained model (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import yaml
import argparse

sys.path.append(abspath('../../../'))
from models.pytorch.load_model import load
from examples.csj.data.load_dataset_hierarchical import Dataset
from examples.csj.metrics.cer import do_eval_cer
from examples.csj.metrics.wer import do_eval_wer

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
parser.add_argument('--max_decode_length', type=int, default=60,
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')
parser.add_argument('--max_decode_length_sub', type=int, default=100,
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')


def main():

    args = parser.parse_args()

    # Load config file
    with open(join(args.model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Get voabulary number (excluding blank, <SOS>, <EOS> classes)
    with open('../metrics/vocab_num.yml', "r") as f:
        vocab_num = yaml.load(f)
        params['num_classes'] = vocab_num[params['data_size']
                                          ][params['label_type']]
        params['num_classes_sub'] = vocab_num[params['data_size']
                                              ][params['label_type_sub']]

    # Load model
    model = load(model_type=params['model_type'], params=params)

    # GPU setting
    model.set_cuda(deterministic=False)

    # Restore the saved model
    checkpoint = model.load_checkpoint(
        save_path=args.model_path, epoch=args.epoch)
    model.load_state_dict(checkpoint['state_dict'])

    # ***Change to evaluation mode***
    model.eval()

    # Load dataset
    vocab_file_path = '../metrics/vocab_files/' + \
        params['label_type'] + '_' + params['data_size'] + '.txt'
    vocab_file_path_sub = '../metrics/vocab_files/' + \
        params['label_type_sub'] + '_' + params['data_size'] + '.txt'
    eval1_data = Dataset(
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='eval1', data_size=params['data_size'],
        label_type=params['label_type'], label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, save_format=params['save_format'])
    eval2_data = Dataset(
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='eval2', data_size=params['data_size'],
        label_type=params['label_type'], label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, save_format=params['save_format'])
    eval3_data = Dataset(
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='eval3', data_size=params['data_size'],
        label_type=params['label_type'], label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, save_format=params['save_format'])

    print('=== Test Data Evaluation ===')
    if params['label_type'] == 'pos':
        wer_eval1 = do_eval_wer(
            model=model,
            model_type=params['model_type'],
            dataset=eval1_data,
            label_type=params['label_type_sub'],
            data_size=params['data_size'],
            beam_width=args.beam_width,
            max_decode_length=args.max_decode_length,
            eval_batch_size=args.eval_batch_size,
            progressbar=True,
            is_pos=True)
        print('  WER (eval1, main): %f %%' % (wer_eval1 * 100))
        wer_eval2 = do_eval_wer(
            model=model,
            model_type=params['model_type'],
            dataset=eval2_data,
            label_type=params['label_type_sub'],
            data_size=params['data_size'],
            beam_width=args.beam_width,
            max_decode_length=args.max_decode_length,
            eval_batch_size=args.eval_batch_size,
            progressbar=True,
            is_pos=True)
        print('  WER (eval2, main): %f %%' % (wer_eval2 * 100))
        wer_eval3 = do_eval_wer(
            model=model,
            model_type=params['model_type'],
            dataset=eval3_data,
            label_type=params['label_type_sub'],
            data_size=params['data_size'],
            beam_width=args.beam_width,
            max_decode_length=args.max_decode_length,
            eval_batch_size=args.eval_batch_size,
            progressbar=True,
            is_pos=True)
        print('  WER (eval3, main): %f %%' % (wer_eval3 * 100))
        print('  WER (mean, main): %f %%' %
              ((wer_eval1 + wer_eval2 + wer_eval3) * 100 / 3))
    else:
        wer_eval1 = do_eval_wer(
            model=model,
            model_type=params['model_type'],
            dataset=eval1_data,
            label_type=params['label_type'],
            data_size=params['data_size'],
            beam_width=args.beam_width,
            max_decode_length=args.max_decode_length,
            eval_batch_size=args.eval_batch_size,
            progressbar=True)
        print('  WER (eval1, main): %f %%' % (wer_eval1 * 100))
        wer_eval2 = do_eval_wer(
            model=model,
            model_type=params['model_type'],
            dataset=eval2_data,
            label_type=params['label_type'],
            data_size=params['data_size'],
            beam_width=args.beam_width,
            max_decode_length=args.max_decode_length,
            eval_batch_size=args.eval_batch_size,
            progressbar=True)
        print('  WER (eval2, main): %f %%' % (wer_eval2 * 100))
        wer_eval3 = do_eval_wer(
            model=model,
            model_type=params['model_type'],
            dataset=eval3_data,
            label_type=params['label_type'],
            data_size=params['data_size'],
            beam_width=args.beam_width,
            max_decode_length=args.max_decode_length,
            eval_batch_size=args.eval_batch_size,
            progressbar=True)
        print('  WER (eval3, main): %f %%' % (wer_eval3 * 100))
        print('  WER (mean, main): %f %%' %
              ((wer_eval1 + wer_eval2 + wer_eval3) * 100 / 3))

        cer_eval1 = do_eval_cer(
            model=model,
            model_type=params['model_type'],
            dataset=eval1_data,
            label_type=params['label_type_sub'],
            data_size=params['data_size'],
            beam_width=args.beam_width,
            max_decode_length=args.max_decode_length,
            eval_batch_size=args.eval_batch_size,
            progressbar=True)
        print('  CER (eval1, sub): %f %%' % (cer_eval1 * 100))
        cer_eval2 = do_eval_cer(
            model=model,
            model_type=params['model_type'],
            dataset=eval2_data,
            label_type=params['label_type_sub'],
            data_size=params['data_size'],
            beam_width=args.beam_width,
            max_decode_length=args.max_decode_length,
            eval_batch_size=args.eval_batch_size,
            progressbar=True)
        print('  CER (eval2, sub): %f %%' % (cer_eval2 * 100))
        cer_eval3 = do_eval_cer(
            model=model,
            model_type=params['model_type'],
            dataset=eval3_data,
            label_type=params['label_type_sub'],
            data_size=params['data_size'],
            beam_width=args.beam_width,
            max_decode_length=args.max_decode_length,
            eval_batch_size=args.eval_batch_size,
            progressbar=True)
        print('  CER (eval3, sub): %f %%' % (cer_eval3 * 100))
        print('  CER (mean, sub): %f %%' %
              ((cer_eval1 + cer_eval2 + cer_eval3) * 100 / 3))


if __name__ == '__main__':
    main()
