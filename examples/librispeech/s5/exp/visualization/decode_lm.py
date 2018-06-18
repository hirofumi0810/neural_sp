#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate texts by the RNNLM (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.librispeech.s5.exp.dataset.load_dataset_lm import Dataset
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

MAX_DECODE_LEN_WORD = 200
MIN_DECODE_LEN_WORD = 1
MAX_DECODE_LEN_CHAR = 600
MIN_DECODE_LEN_CHAR = 1


def main():

    args = parser.parse_args()

    # Load a config file
    config = load_config(join(args.model_path, 'config.yml'), is_eval=True)
    config['data_size'] = str(config['data_size'])

    # Load dataset
    dataset = Dataset(data_save_path=args.data_save_path,
                      data_type='test_clean',
                      # data_type='test_other',
                      data_size=config['data_size'],
                      label_type=config['label_type'],
                      batch_size=args.eval_batch_size,
                      sort_utt=False, reverse=False, tool=config['tool'],
                      vocab=config['vocab'])
    config['num_classes'] = dataset.num_classes

    # Load model
    model = load(model_type=config['model_type'],
                 config=config,
                 backend=config['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # NOTE: after load the rnn config are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()
    # https://github.com/pytorch/examples/blob/master/word_language_model/main.py

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    sys.stdout = open(join(args.model_path, 'decode.txt'), 'w')

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
        best_hyps = model.decode([y[:5] for y in ys],
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
