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

from neural_sp.bin.asr.plot_utils import plot_attention_weights
from neural_sp.datasets.loader_asr import Dataset
from neural_sp.models.rnnlm.rnnlm import RNNLM
from neural_sp.models.seq2seq.seq2seq import Seq2seq
from neural_sp.utils.config import load_config
from neural_sp.utils.general import mkdir_join
from neural_sp.utils.general import set_logger

parser = argparse.ArgumentParser()
# general
parser.add_argument('--model', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--plot_dir', type=str,
                    help='directory to save figures')
# dataset
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
    logger = set_logger(os.path.join(args.plot_dir, 'plot.log'), key='decoding')

    for i, set in enumerate(args.eval_sets):
        # Load dataset
        eval_set = Dataset(csv_path=set,
                           dict_path=os.path.join(args.model, 'dict.txt'),
                           dict_path_sub=os.path.join(args.model, 'dict_sub.txt') if os.path.isfile(
                               os.path.join(args.model, 'dict_sub.txt')) else None,
                           wp_model=os.path.join(args.model, 'wp.model'),
                           label_type=args.label_type,
                           batch_size=args.batch_size,
                           max_num_frames=args.max_num_frames,
                           min_num_frames=args.min_num_frames,
                           is_test=True)

        if i == 0:
            args.num_classes = eval_set.num_classes
            args.input_dim = eval_set.input_dim
            args.num_classes_sub = eval_set.num_classes_sub

            # TODO(hirofumi): For cold fusion
            args.rnnlm_cold_fusion = None
            args.rnnlm_init = None

            # Load the ASR model
            model = Seq2seq(args)
            epoch, _, _, _ = model.load_checkpoint(args.model, epoch=args.epoch)

            model.save_path = args.model

            # For shallow fusion
            if args.rnnlm_cold_fusion is None and args.rnnlm is not None and args.rnnlm_weight > 0:
                # Load a RNNLM config file
                config_rnnlm = load_config(os.path.join(args.rnnlm, 'config.yml'))

                # Merge config with args
                args_rnnlm = argparse.Namespace()
                for k, v in config_rnnlm.items():
                    setattr(args_rnnlm, k, v)

                assert args.label_type == args_rnnlm.label_type
                args_rnnlm.num_classes = eval_set.num_classes

                # Load the pre-trianed RNNLM
                rnnlm = RNNLM(args_rnnlm)
                rnnlm.load_checkpoint(args.rnnlm, epoch=-1)
                if args_rnnlm.backward:
                    model.rnnlm_bwd_0 = rnnlm
                else:
                    model.rnnlm_fwd_0 = rnnlm

                logger.info('RNNLM path: %s' % args.rnnlm)
                logger.info('RNNLM weight: %.3f' % args.rnnlm_weight)
                logger.info('RNNLM backward: %s' % str(config_rnnlm['backward']))

            # GPU setting
            model.cuda()

            logger.info('beam width: %d' % args.beam_width)
            logger.info('length penalty: %.3f' % args.length_penalty)
            logger.info('coverage penalty: %.3f' % args.coverage_penalty)
            logger.info('coverage threshold: %.3f' % args.coverage_threshold)
            logger.info('epoch: %d' % (epoch - 1))

        save_path = mkdir_join(args.plot_dir, 'att_weights')

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        while True:
            batch, is_new_epoch = eval_set.next(decode_params['batch_size'])
            best_hyps, aws, perm_idx = model.decode(batch['xs'], decode_params,
                                                    exclude_eos=False)
            ys = [batch['ys'][i] for i in perm_idx]

            if model.bwd_weight > 0.5:
                # Reverse the order
                best_hyps = [hyp[::-1] for hyp in best_hyps]
                aws = [aw[::-1] for aw in aws]

            for b in range(len(batch['xs'])):
                if args.label_type == 'word':
                    token_list = eval_set.idx2word(best_hyps[b], return_list=True)
                if args.label_type == 'wp':
                    token_list = eval_set.idx2wp(best_hyps[b], return_list=True)
                elif args.label_type == 'char':
                    token_list = eval_set.idx2char(best_hyps[b], return_list=True)
                elif args.label_type == 'phone':
                    token_list = eval_set.idx2phone(best_hyps[b], return_list=True)
                else:
                    raise NotImplementedError(args.label_type)
                token_list = [unicode(t, 'utf-8') for t in token_list]
                speaker = '_'.join(batch['utt_ids'][b].replace('-', '_').split('_')[:-2])

                # error check
                assert len(batch['xs'][b]) <= 2000

                plot_attention_weights(aws[b][:len(token_list)],
                                       label_list=token_list,
                                       spectrogram=batch['xs'][b][:,
                                                                  :eval_set.input_dim] if args.input_type == 'speech' else None,
                                       save_path=mkdir_join(save_path, speaker, batch['utt_ids'][b] + '.png'),
                                       figsize=(20, 8))

                ref = ys[b]
                if model.bwd_weight > 0.5:
                    hyp = ' '.join(token_list[::-1])
                else:
                    hyp = ' '.join(token_list)
                logger.info('utt-id: %s' % batch['utt_ids'][b])
                logger.info('Ref: %s' % ref.lower())
                logger.info('Hyp: %s' % hyp)
                logger.info('-' * 50)

            if is_new_epoch:
                break


if __name__ == '__main__':
    main()
