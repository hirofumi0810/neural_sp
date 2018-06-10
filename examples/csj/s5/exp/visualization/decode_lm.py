#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate texts by the RNNLM (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.csj.s5.exp.dataset.load_dataset_lm import Dataset
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

MAX_DECODE_LEN_WORD = 100
MIN_DECODE_LEN_WORD = 1
MAX_DECODE_LEN_CHAR = 200
MIN_DECODE_LEN_CHAR = 1


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    dataset = Dataset(
        data_save_path=args.data_save_path,
        data_type='eval1',
        # data_type='eval2',
        # data_type='eval3',
        data_size=params['data_size'],
        label_type=params['label_type'],
        batch_size=args.eval_batch_size,
        sort_utt=False, reverse=False, tool=params['tool'])
    params['num_classes'] = dataset.num_classes

    # Load model
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    # sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    ######################################################################

    if dataset.label_type == 'word':
        map_fn = dataset.idx2word
        max_decode_len = MAX_DECODE_LEN_WORD
    else:
        map_fn = dataset.idx2char
        max_decode_len = MAX_DECODE_LEN_CHAR

    for batch, is_new_epoch in dataset:

        if dataset.is_test:
            ys = []
            for b in range(len(batch['ys'])):
                if dataset.label_type == 'word':
                    indices = dataset.word2idx(batch['ys'][b])
                else:
                    indices = dataset.char2idx(batch['ys'][b])
                ys += [indices]
                # NOTE: transcript is seperated by space('_')
        else:
            ys = batch['ys']

        # Decode
        best_hyps = model.decode([y[0] for y in ys],
                                 max_decode_len=max_decode_len)

        for b in range(len(batch['ys'])):
            # Reference
            if dataset.is_test:
                str_ref = batch['ys'][b]
                # NOTE: transcript is seperated by space('_')
            else:
                str_ref = map_fn(batch['ys'][b])

            # Hypothesis
            str_hyp = map_fn(best_hyps[b])

            print('----- wav: %s -----' % batch['input_names'][b])
            print('Ref: %s' % str_ref.replace('_', ' '))
            print('Hyp: %s' % str_hyp.replace('_', ' '))

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
