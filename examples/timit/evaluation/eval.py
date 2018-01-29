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
from examples.timit.data.load_dataset import Dataset
from examples.timit.metrics.per import do_eval_per
from utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--beam_width', type=int, default=10,
                    help='beam_width (int, optional): beam width for beam search.' +
                    ' 1 disables beam search, which mean greedy decoding.')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--max_decode_len', type=int, default=40,
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    vocab_file_path = '../metrics/vocab_files/phone39.txt'
    test_data = Dataset(
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='test',
        label_type='phone39',
        vocab_file_path=vocab_file_path,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, save_format=params['save_format'])
    params['num_classes'] = test_data.num_classes

    # Load model
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    per_test = do_eval_per(
        model=model,
        model_type=params['model_type'],
        dataset=test_data,
        label_type=params['label_type'],
        beam_width=args.beam_width,
        max_decode_len=args.max_decode_len,
        eval_batch_size=args.eval_batch_size,
        progressbar=True)
    print('  PER (test): %f %%' % (per_test * 100))


if __name__ == '__main__':
    main()
