#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate texts by the RNNLM."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from src.models.load_model import load
from src.dataset.loader_lm import Dataset
from src.utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str,
                    help='the name of corpus')
parser.add_argument('--data_type', type=str,
                    help='the type of data (ex. train, dev etc.)')
parser.add_argument('--data_save_path', type=str,
                    help='path to saved data')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
args = parser.parse_args()


def main():

    # Load a config file
    config = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    dataset = Dataset(
        corpus=args.corpus,
        data_save_path=args.data_save_path,
        model_type=config['model_type'],
        data_size=config['data_size'] if 'data_size' in config.keys() else '',
        data_type=args.data_type,
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
    model.flatten_parameters()
    # https://github.com/pytorch/examples/blob/master/word_language_model/main.py

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    sys.stdout = open(join(args.model_path, 'decode.txt'), 'w')

    if dataset.label_type == 'word':
        map_fn = dataset.idx2word
    else:
        map_fn = dataset.idx2char

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
                                 max_decode_len=100)

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
