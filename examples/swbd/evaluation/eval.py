#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained model (Switchboard corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import yaml
import argparse

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.swbd.data.load_dataset import Dataset
from examples.swbd.metrics.cer import do_eval_cer
from examples.swbd.metrics.wer import do_eval_wer

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
parser.add_argument('--max_decode_len', type=int, default=300,  # or 100
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

    # Load model
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    # Restore the saved model
    checkpoint = model.load_checkpoint(
        save_path=args.model_path, epoch=args.epoch)
    model.load_state_dict(checkpoint['state_dict'])

    # Load dataset
    vocab_file_path = '../metrics/vocab_files/' + \
        params['label_type'] + '_' + params['data_size'] + '.txt'
    eval2000_swbd_data = Dataset(
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='eval2000_swbd', data_size=params['data_size'],
        label_type=params['label_type'], vocab_file_path=vocab_file_path,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, save_format=params['save_format'])
    eval2000_ch_data = Dataset(
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='eval2000_ch', data_size=params['data_size'],
        label_type=params['label_type'], vocab_file_path=vocab_file_path,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, save_format=params['save_format'])

    if 'word' in params['label_type']:
        wer_eval2000_swbd = do_eval_wer(
            model=model,
            model_type=params['model_type'],
            dataset=eval2000_swbd_data,
            label_type=params['label_type'],
            beam_width=args.beam_width,
            max_decode_len=args.max_decode_len,
            eval_batch_size=args.eval_batch_size,
            progressbar=True)
        print('  WER (SWB): %f %%' % (wer_eval2000_swbd * 100))
        wer_eval2000_ch = do_eval_wer(
            model=model,
            model_type=params['model_type'],
            dataset=eval2000_ch_data,
            label_type=params['label_type'],
            beam_width=args.beam_width,
            max_decode_len=args.max_decode_len,
            eval_batch_size=args.eval_batch_size,
            progressbar=True)
        print('  WER (CHE): %f %%' % (wer_eval2000_ch * 100))
        print('  WER (mean): %f %%' %
              ((wer_eval2000_swbd + wer_eval2000_ch) * 100 / 2))
    else:
        cer_eval2000_swbd, wer_eval2000_swbd = do_eval_cer(
            model=model,
            model_type=params['model_type'],
            dataset=eval2000_swbd_data,
            label_type=params['label_type'],
            beam_width=args.beam_width,
            max_decode_len=args.max_decode_len,
            eval_batch_size=args.eval_batch_size,
            progressbar=True)
        print('  CER (SWB): %f %%' % (cer_eval2000_swbd * 100))
        print('  WER (SWB): %f %%' % (wer_eval2000_swbd * 100))
        cer_eval2000_ch, wer_eval2000_ch = do_eval_cer(
            model=model,
            model_type=params['model_type'],
            dataset=eval2000_ch_data,
            label_type=params['label_type'],
            beam_width=args.beam_width,
            max_decode_len=args.max_decode_len,
            eval_batch_size=args.eval_batch_size,
            progressbar=True)
        print('  CER (CHE): %f %%' % (cer_eval2000_ch * 100))
        print('  WER (CHE): %f %%' % (wer_eval2000_ch * 100))
        print('  CER (mean): %f %%' %
              ((cer_eval2000_swbd + cer_eval2000_ch) * 100 / 2))
        print('  WER (mean): %f %%' %
              ((wer_eval2000_swbd + wer_eval2000_ch) * 100 / 2))


if __name__ == '__main__':
    main()
