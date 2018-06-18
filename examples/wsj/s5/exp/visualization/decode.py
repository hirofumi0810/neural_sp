#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate texts by the ASR model (WSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.wsj.s5.exp.dataset.load_dataset import Dataset
from utils.config import load_config
from utils.evaluation.edit_distance import wer_align
from utils.evaluation.normalization import normalize

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
                    help='length penalty')
parser.add_argument('--coverage_penalty', type=float, default=0,
                    help='coverage penalty')

MAX_DECODE_LEN_WORD = 32
MIN_DECODE_LEN_WORD = 2
MIN_DECODE_LEN_RATIO_WORD = 0
MAX_DECODE_LEN_CHAR = 199
MIN_DECODE_LEN_CHAR = 10
MIN_DECODE_LEN_RATIO_CHAR = 0.2


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    config = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    dataset = Dataset(data_save_path=args.data_save_path,
                      input_freq=config['input_freq'],
                      use_delta=config['use_delta'],
                      use_double_delta=config['use_double_delta'],
                      data_type='test_eval92',
                      data_size=config['data_size'],
                      label_type=config['label_type'],
                      batch_size=args.eval_batch_size,
                      sort_utt=False, reverse=False, tool=config['tool'])
    config['num_classes'] = dataset.num_classes
    config['num_classes_sub'] = dataset.num_classes

    # Load model
    model = load(model_type=config['model_type'],
                 config=config,
                 backend=config['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    sys.stdout = open(join(args.model_path, 'decode.txt'), 'w')

    if dataset.label_type == 'word':
        map_fn = dataset.idx2word
        max_decode_len = MAX_DECODE_LEN_WORD
        min_decode_len = MIN_DECODE_LEN_WORD
        min_decode_len_ratio = MIN_DECODE_LEN_RATIO_WORD
    else:
        map_fn = dataset.idx2char
        max_decode_len = MAX_DECODE_LEN_CHAR
        min_decode_len = MIN_DECODE_LEN_CHAR
        min_decode_len_ratio = MIN_DECODE_LEN_RATIO_CHAR

    for batch, is_new_epoch in dataset:
        # Decode
        if model.model_type == 'nested_attention':
            best_hyps, _, best_hyps_sub, _, perm_idx = model.decode(
                batch['xs'],
                beam_width=args.beam_width,
                max_decode_len=max_decode_len,
                max_decode_len_sub=max_decode_len,
                length_penalty=args.length_penalty,
                coverage_penalty=args.coverage_penalty)
        else:
            best_hyps, _, perm_idx = model.decode(
                batch['xs'],
                beam_width=args.beam_width,
                max_decode_len=max_decode_len,
                min_decode_len=min_decode_len,
                min_decode_len_ratio=min_decode_len_ratio,
                length_penalty=args.length_penalty,
                coverage_penalty=args.coverage_penalty)

        ys = [batch['ys'][i] for i in perm_idx]

        for b in range(len(batch['xs'])):
            # Reference
            if dataset.is_test:
                str_ref = ys[b]
                # NOTE: transcript is seperated by space('_')
            else:
                str_ref = map_fn(ys[b])

            # Hypothesis
            str_hyp = map_fn(best_hyps[b])

            print('----- wav: %s -----' % batch['input_names'][b])

            str_hyp = normalize(str_hyp)

            wer, _, _, _ = wer_align(ref=str_ref.split('_'),
                                     hyp=str_hyp.split('_'),
                                     normalize=True)
            print('\nWER: %.3f %%' % wer)

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
