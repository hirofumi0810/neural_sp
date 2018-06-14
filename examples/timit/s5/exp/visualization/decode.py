#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate texts by the ASR model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse
import re

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.timit.s5.exp.dataset.load_dataset import Dataset
from utils.config import load_config
from utils.evaluation.edit_distance import compute_wer

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
                    help='length penalty in the beam search decoding')
parser.add_argument('--coverage_penalty', type=float, default=0,
                    help='coverage penalty in the beam search decoding')

MAX_DECODE_LEN_PHONE = 71
MIN_DECODE_LEN_PHONE = 13


def main():

    args = parser.parse_args()

    # Load a config file
    config = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    dataset = Dataset(data_save_path=args.data_save_path,
                      input_freq=config['input_freq'],
                      use_delta=config['use_delta'],
                      use_double_delta=config['use_double_delta'],
                      data_type='test',
                      label_type=config['label_type'],
                      batch_size=args.eval_batch_size,
                      sort_utt=True, reverse=True, tool=config['tool'])
    config['num_classes'] = dataset.num_classes

    # Load model
    model = load(model_type=config['model_type'],
                 config=config,
                 backend=config['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    # sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    for batch, is_new_epoch in dataset:
        # Decode
        best_hyps, _, perm_idx = model.decode(
            batch['xs'],
            beam_width=args.beam_width,
            max_decode_len=MAX_DECODE_LEN_PHONE,
            min_decode_len=MIN_DECODE_LEN_PHONE,
            length_penalty=args.length_penalty,
            coverage_penalty=args.coverage_penalty)

        if model.model_type == 'attention' and model.ctc_loss_weight > 0:
            best_hyps_ctc, perm_idx = model.decode_ctc(
                batch['xs'], beam_width=args.beam_width)

        ys = [batch['ys'][i] for i in perm_idx]

        for b in range(len(batch['xs'])):
            # Reference
            if dataset.is_test:
                str_ref = ys[b]
                # NOTE: transcript is seperated by space('_')
            else:
                str_ref = dataset.idx2phone(ys[b])

            # Hypothesis
            str_hyp = dataset.idx2phone(best_hyps[b])

            print('----- wav: %s -----' % batch['input_names'][b])
            print('Ref      : %s' % str_ref)
            print('Hyp      : %s' % str_hyp)
            if model.model_type == 'attention' and model.ctc_loss_weight > 0:
                str_hyp_ctc = dataset.idx2phone(best_hyps_ctc[b])
                print('Hyp (CTC): %s' % str_hyp_ctc)

            # Compute PER
            per, _, _, _ = compute_wer(ref=str_ref.split('_'),
                                       hyp=re.sub(r'(.*)_>(.*)', r'\1',
                                                  str_hyp).split('_'),
                                       normalize=True)
            print('PER: %.3f %%' % (per * 100))
            if model.model_type == 'attention' and model.ctc_loss_weight > 0:
                per_ctc, _, _, _ = compute_wer(ref=str_ref.split('_'),
                                               hyp=str_hyp_ctc.split('_'),
                                               normalize=True)
                print('PER (CTC): %.3f %%' % (per_ctc * 100))

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
