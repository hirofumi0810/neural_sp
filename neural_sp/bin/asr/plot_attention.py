#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Plot attention weights of the attention model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from distutils.util import strtobool
import os
import shutil
import sys

from neural_sp.bin.asr.plot_utils import plot_attention_weights
from neural_sp.datasets.loader_asr import Dataset
from neural_sp.models.seq2seq.seq2seq import Seq2seq
from neural_sp.models.rnnlm.rnnlm import RNNLM
from neural_sp.utils.config import load_config
from neural_sp.utils.general import mkdir_join
from neural_sp.utils.general import set_logger

from neural_sp.evaluators.edit_distance import wer_align

parser = argparse.ArgumentParser()
# general
parser.add_argument('--model', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
# dataset
parser.add_argument('--corpus', type=str,
                    help='the name of corpus')
parser.add_argument('--eval_sets', type=str, nargs='+',
                    help='path to csv files for the evaluation sets')
# decoding paramter
parser.add_argument('--batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--beam_width', type=int, default=1,
                    help='the size of beam')
parser.add_argument('--max_len_ratio', type=float, default=1,
                    help='')
parser.add_argument('--min_len_ratio', type=float, default=0.0,
                    help='')
parser.add_argument('--length_penalty', type=float, default=0.0,
                    help='length penalty')
parser.add_argument('--coverage_penalty', type=float, default=0.0,
                    help='coverage penalty')
parser.add_argument('--coverage_threshold', type=float, default=0.0,
                    help='coverage threshold')
parser.add_argument('--rnnlm_weight', type=float, default=0.0,
                    help='the weight of RNNLM score')
parser.add_argument('--rnnlm', type=str, default=None, nargs='?',
                    help='path to the RMMLM')
parser.add_argument('--resolving_unk', type=strtobool, default=False,
                    help='')
args = parser.parse_args()


def main():

    # Load a config file
    config = load_config(os.path.join(args.model, 'config.yml'))

    decode_params = vars(args)

    # Merge config with args
    for k, v in config.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # Setting for logging
    logger = set_logger(os.path.join(args.model, 'decode.log'), key='decoding')

    for i, set in enumerate(args.eval_sets):
        # Load dataset
        eval_set = Dataset(csv_path=set,
                           dict_path=os.path.join(args.model, 'dict.txt'),
                           dict_path_sub=os.path.join(args.model, 'dict_sub.txt') if os.path.isfile(
                               os.path.join(args.model, 'dict_sub.txt')) else None,
                           label_type=args.label_type,
                           batch_size=args.batch_size,
                           max_epoch=args.num_epochs,
                           max_num_frames=args.max_num_frames,
                           min_num_frames=args.min_num_frames,
                           is_test=False)

        if i == 0:
            args.num_classes = eval_set.num_classes
            args.input_dim = eval_set.input_dim
            args.num_classes_sub = eval_set.num_classes_sub

            # TODO(hirofumi): For cold fusion
            args.rnnlm_cf = None
            args.rnnlm_init = None

            # Load the ASR model
            model = Seq2seq(args)

            # Restore the saved parameters
            epoch, _, _, _ = model.load_checkpoint(args.model, epoch=args.epoch)

            model.save_path = args.model

            # TODO(hirofumi): For shallow fusion

            # GPU setting
            model.set_cuda(deterministic=False, benchmark=True)

            logger.info('beam width: %d' % args.beam_width)
            logger.info('length penalty: %.3f' % args.length_penalty)
            logger.info('coverage penalty: %.3f' % args.coverage_penalty)
            logger.info('coverage threshold: %.3f' % args.coverage_threshold)
            logger.info('epoch: %d' % (epoch - 1))

        save_path = mkdir_join(args.model, 'att_weights')

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        while True:
            batch, is_new_epoch = eval_set.next(decode_params['batch_size'])
            best_hyps, aw, perm_idx = model.decode(batch['xs'], decode_params,
                                                   exclude_eos=False)
            ys = [batch['ys'][i] for i in perm_idx]

            for b in range(len(batch['xs'])):
                if args.label_type in ['word', 'wordpiece']:
                    token_list = eval_set.idx2word(best_hyps[b], return_list=True)
                elif args.label_type == 'char':
                    token_list = eval_set.idx2char(best_hyps[b], return_list=True)
                elif args.label_type == 'phone':
                    token_list = eval_set.idx2phone(best_hyps[b], return_list=True)
                else:
                    raise NotImplementedError()
                token_list = [unicode(t, 'utf-8') for t in token_list]
                speaker = '_'.join(batch['utt_ids'][b].replace('-', '_').split('_')[:-2])

                # error check
                assert len(batch['xs'][b]) <= 2000

                plot_attention_weights(aw[b][:len(token_list)],
                                       label_list=token_list,
                                       spectrogram=batch['xs'][b][:,
                                                                  :eval_set.input_dim] if args.input_type == 'speech' else None,
                                       save_path=mkdir_join(save_path, speaker, batch['utt_ids'][b] + '.png'),
                                       figsize=(20, 8))

                # Reference
                if eval_set.is_test:
                    text_ref = ys[b]
                else:
                    if args.label_type in ['word', 'wordpiece']:
                        text_ref = eval_set.idx2word(ys[b])
                    if args.label_type in ['word', 'wordpiece']:
                        token_list = eval_set.idx2word(ys[b])
                    elif args.label_type == 'char':
                        token_list = eval_set.idx2char(ys[b])
                    elif args.label_type == 'phone':
                        token_list = eval_set.idx2phone(ys[b])

                # Hypothesis
                text_hyp = ' '.join(token_list)

                sys.stdout = open(os.path.join(save_path, speaker, batch['utt_ids'][b] + '.txt'), 'w')
                ler = wer_align(ref=text_ref.split(' '),
                                hyp=text_hyp.encode('utf-8').split(' '),
                                normalize=True,
                                double_byte=False)[0]  # TODO(hirofumi): add corpus to args
                print('\nLER: %.3f %%\n\n' % ler)

            if is_new_epoch:
                break


if __name__ == '__main__':
    main()
