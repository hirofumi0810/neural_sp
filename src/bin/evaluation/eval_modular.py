#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the A2P +P2W  model on the modular training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from src.models.load_model import load
from src.dataset.loader import Dataset as Dataset_a2p
from src.dataset.loader_p2w import Dataset as Dataset_p2w
# from src.metrics.character_modular import eval_char
from src.metrics.word_modular import eval_word
from src.utils.config import load_config
from src.utils.evaluation.logging import set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str,
                    help='the name of corpus')
parser.add_argument('--eval_sets', type=str, nargs='+',
                    help='evaluation sets')
parser.add_argument('--data_save_path', type=str,
                    help='path to saved data')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')

# A2P
parser.add_argument('--model_path_a2p', type=str,
                    help='path to the model to evaluate (A2P)')
parser.add_argument('--epoch_a2p', type=int, default=-1,
                    help='the epoch to restore (A2P)')
parser.add_argument('--beam_width_a2p', type=int, default=1,
                    help='the size of beam (A2P)')
parser.add_argument('--length_penalty_a2p', type=float, default=0,
                    help='length penalty (A2P)')
parser.add_argument('--coverage_penalty_a2p', type=float, default=0,
                    help='coverage penalty (A2P)')

# P2W
parser.add_argument('--model_path_p2w', type=str,
                    help='path to the model to evaluate (P2W)')
parser.add_argument('--epoch_p2w', type=int, default=-1,
                    help='the epoch to restore (P2W)')
parser.add_argument('--beam_width_p2w', type=int, default=1,
                    help='the size of beam (P2W)')
parser.add_argument('--length_penalty_p2w', type=float, default=0,
                    help='length penalty (P2w)')
parser.add_argument('--coverage_penalty_p2w', type=float, default=0,
                    help='coverage penalty (P2w)')
parser.add_argument('--rnnlm_weight', type=float, default=0,
                    help='the weight of RNNLM score')
parser.add_argument('--rnnlm_path', type=str, default=None, nargs='?',
                    help='path to the RMMLM')
args = parser.parse_args()

# corpus depending
if args.corpus == 'csj':
    MAX_DECODE_LEN_WORD = 100
    MIN_DECODE_LEN_WORD = 1
    MAX_DECODE_LEN_RATIO_WORD = 1
    MIN_DECODE_LEN_RATIO_WORD = 0

    MAX_DECODE_LEN_CHAR = 200
    MIN_DECODE_LEN_CHAR = 1
    MAX_DECODE_LEN_RATIO_CHAR = 1
    MIN_DECODE_LEN_RATIO_CHAR = 0.2

    MAX_DECODE_LEN_PHONE = 200
    MIN_DECODE_LEN_PHONE = 1
    MAX_DECODE_LEN_RATIO_PHONE = 1
    MIN_DECODE_LEN_RATIO_PHONE = 0
elif args.corpus == 'swbd':
    MAX_DECODE_LEN_WORD = 100
    MIN_DECODE_LEN_WORD = 1
    MAX_DECODE_LEN_RATIO_WORD = 1
    MIN_DECODE_LEN_RATIO_WORD = 0

    MAX_DECODE_LEN_CHAR = 300
    MIN_DECODE_LEN_CHAR = 1
    MAX_DECODE_LEN_RATIO_CHAR = 1
    MIN_DECODE_LEN_RATIO_CHAR = 0.1

    MAX_DECODE_LEN_PHONE = 300
    MIN_DECODE_LEN_PHONE = 1
    MAX_DECODE_LEN_RATIO_PHONE = 1
    MIN_DECODE_LEN_RATIO_PHONE = 0.05
elif args.corpus == 'librispeech':
    MAX_DECODE_LEN_WORD = 200
    MIN_DECODE_LEN_WORD = 1
    MAX_DECODE_LEN_RATIO_WORD = 1
    MIN_DECODE_LEN_RATIO_WORD = 0

    MAX_DECODE_LEN_CHAR = 600
    MIN_DECODE_LEN_CHAR = 1
    MAX_DECODE_LEN_RATIO_CHAR = 1
    MIN_DECODE_LEN_RATIO_CHAR = 0.2
elif args.corpus == 'wsj':
    MAX_DECODE_LEN_WORD = 32
    MIN_DECODE_LEN_WORD = 2
    MAX_DECODE_LEN_RATIO_WORD = 1
    MIN_DECODE_LEN_RATIO_WORD = 0

    MAX_DECODE_LEN_CHAR = 199
    MIN_DECODE_LEN_CHAR = 10
    MAX_DECODE_LEN_RATIO_CHAR = 1
    MIN_DECODE_LEN_RATIO_CHAR = 0.2

    MAX_DECODE_LEN_PHONE = 200
    MIN_DECODE_LEN_PHONE = 1
    MAX_DECODE_LEN_RATIO_PHONE = 1
    MIN_DECODE_LEN_RATIO_PHONE = 0
    # NOTE:
    # dev93 (char): 10-199
    # test_eval92 (char): 16-195
    # dev93 (word): 2-32
    # test_eval92 (word): 3-30
elif args.corpus == 'timit':
    MAX_DECODE_LEN_PHONE = 71
    MIN_DECODE_LEN_PHONE = 13
    MAX_DECODE_LEN_RATIO_PHONE = 1
    MIN_DECODE_LEN_RATIO_PHONE = 0
    # NOTE*
    # dev: 13-71
    # test: 13-69
else:
    raise ValueError(args.corpus)


def main():

    # Load a A2P + P2W config file
    config_a2p = load_config(
        join(args.model_path_a2p, 'config.yml'), is_eval=True)
    config_p2w = load_config(
        join(args.model_path_p2w, 'config.yml'), is_eval=True)

    # Setting for logging
    logger = set_logger(args.model_path_p2w)

    wer_mean, cer_mean = 0, 0
    for i, data_type in enumerate(args.eval_sets):
        # Load dataset
        evalset_a2p = Dataset_a2p(
            corpus=args.corpus,
            data_save_path=args.data_save_path,
            model_type=config_a2p['model_type'],
            input_freq=config_a2p['input_freq'],
            use_delta=config_a2p['use_delta'],
            use_double_delta=config_a2p['use_double_delta'],
            data_size=config_a2p['data_size'] if 'data_size' in config_a2p.keys(
            ) else '',
            vocab=config_a2p['vocab'],
            data_type=data_type,
            label_type=config_a2p['label_type'],
            batch_size=args.eval_batch_size,
            sort_utt=False, reverse=False, tool=config_a2p['tool'])
        config_a2p['num_classes'] = evalset_a2p.num_classes
        evalset_p2w = Dataset_p2w(
            corpus=args.corpus,
            data_save_path=args.data_save_path,
            model_type=config_p2w['model_type'],
            data_type=data_type,
            data_size=config_p2w['data_size'] if 'data_size' in config_p2w.keys(
            ) else '',
            vocab=config_p2w['vocab'],
            label_type_in=config_p2w['label_type_in'],
            label_type=config_p2w['label_type'],
            batch_size=args.eval_batch_size,
            sort_utt=False, reverse=False, tool=config_p2w['tool'],
            use_ctc=config_p2w['model_type'] == 'ctc' or (
                config_p2w['model_type'] == 'attention' and config_p2w['ctc_loss_weight'] > 0),
            subsampling_factor=2 ** sum(config_p2w['subsample_list']))

        if i == 0:
            config_p2w['num_classes_input'] = evalset_p2w.num_classes_in
            config_p2w['num_classes'] = evalset_p2w.num_classes
            config_p2w['num_classes_sub'] = evalset_p2w.num_classes
            assert config_a2p['num_classes'] == config_p2w['num_classes_input']

            # Load the A2P + P2W model
            model_a2p = load(model_type=config_a2p['model_type'],
                             config=config_a2p,
                             backend=config_a2p['backend'])
            model_p2w = load(model_type=config_p2w['model_type'],
                             config=config_p2w,
                             backend=config_p2w['backend'])

            # Restore the saved parameters
            epoch_a2p, _, _, _, = model_a2p.load_checkpoint(
                save_path=args.model_path_a2p, epoch=args.epoch_a2p)
            epoch_p2w, _, _, _ = model_p2w.load_checkpoint(
                save_path=args.model_path_p2w, epoch=args.epoch_p2w)

            # For shallow fusion
            if args.rnnlm_path is not None and args.rnnlm_weight > 0:
                # Load a RNNLM config file
                config_rnnlm = load_config(
                    join(args.rnnlm_path, 'config.yml'), is_eval=True)
                assert config_p2w['label_type'] == config_rnnlm['label_type']
                config_rnnlm['num_classes'] = evalset_p2w.num_classes

                # Load the pre-trianed RNNLM
                rnnlm = load(model_type=config_rnnlm['model_type'],
                             config=config_rnnlm,
                             backend=config_rnnlm['backend'])
                rnnlm.load_checkpoint(save_path=args.rnnlm_path, epoch=-1)
                rnnlm.flatten_parameters()
                if config_rnnlm['backward']:
                    model_p2w.rnnlm_0_bwd = rnnlm
                else:
                    model_p2w.rnnlm_0_fwd = rnnlm

                logger.info('RNNLM path: %s' % args.rnnlm_path)
                logger.info('RNNLM weight: %.3f' % args.rnnlm_weight)

            # GPU setting
            model_a2p.set_cuda(deterministic=False, benchmark=True)
            model_p2w.set_cuda(deterministic=False, benchmark=True)

            logger.info('model path (A2P): %s' % args.model_path_a2p)
            logger.info('beam width (A2P): %d' % args.beam_width_a2p)
            logger.info('length penalty (A2P): %.3f' % args.length_penalty_a2p)
            logger.info('coverage penalty (A2P): %.3f' %
                        args.coverage_penalty_a2p)
            logger.info('epoch (A2P): %d' % (epoch_a2p - 1))
            logger.info('model path (P2W): %s' % args.model_path_p2w)
            logger.info('beam width (P2w): %d' % args.beam_width_p2w)
            logger.info('length penalty (P2W): %.3f' % args.length_penalty_p2w)
            logger.info('coverage penalty (P2W): %.3f' %
                        args.coverage_penalty_p2w)
            logger.info('epoch (P2W): %d' % (epoch_p2w - 1))

        if config_p2w['label_type'] == 'word':
            wer, df = eval_word(
                models_a2p=[model_a2p],
                models_p2w=[model_p2w],
                dataset_a2p=evalset_a2p,
                dataset_p2w=evalset_p2w,
                eval_batch_size=args.eval_batch_size,
                beam_width_a2p=args.beam_width_a2p,  # A2P
                max_decode_len_a2p=MAX_DECODE_LEN_PHONE,
                min_decode_len_a2p=MIN_DECODE_LEN_PHONE,
                min_decode_len_ratio_a2p=MIN_DECODE_LEN_RATIO_PHONE,
                length_penalty_a2p=args.length_penalty_a2p,
                coverage_penalty_a2p=args.coverage_penalty_a2p,
                beam_width_p2w=args.beam_width_p2w,  # P2W
                max_decode_len_p2w=MAX_DECODE_LEN_WORD,
                min_decode_len_p2w=MIN_DECODE_LEN_WORD,
                min_decode_len_ratio_p2w=MIN_DECODE_LEN_RATIO_WORD,
                length_penalty_p2w=args.length_penalty_p2w,
                coverage_penalty_p2w=args.coverage_penalty_p2w,
                rnnlm_weight=args.rnnlm_weight,
                progressbar=True)
            wer_mean += wer
            logger.info('  WER (%s): %.3f %%' % (data_type, wer))
            logger.info(df)
        elif 'character' in config_p2w['label_type']:
            raise NotImplementedError
            # wer, cer, df = eval_char(
            #     models_a2p=[model_a2p],
            #     models_p2w=[model_p2w],
            #     dataset_a2p=evalset_a2p,
            #     dataset_p2w=evalset_p2w,
            #     eval_batch_size=args.eval_batch_size,
            #     beam_width_a2p=args.beam_width_a2p,  # A2P
            #     max_decode_len_a2p=MAX_DECODE_LEN_PHONE,
            #     min_decode_len_a2p=MIN_DECODE_LEN_PHONE,
            #     min_decode_len_ratio_a2p=MIN_DECODE_LEN_RATIO_PHONE,
            #     length_penalty_a2p=args.length_penalty_a2p,
            #     coverage_penalty_a2p=args.coverage_penalty_a2p,
            #     beam_width_p2w=args.beam_width_p2w,  # P2W
            #     max_decode_len_p2w=MAX_DECODE_LEN_CHAR,
            #     min_decode_len_p2w=MIN_DECODE_LEN_CHAR,
            #     min_decode_len_ratio_p2w=MIN_DECODE_LEN_RATIO_CHAR,
            #     length_penalty_p2w=args.length_penalty_p2w,
            #     coverage_penalty_p2w=args.coverage_penalty_p2w,
            #     rnnlm_weight=args.rnnlm_weight,
            #     progressbar=True)
            # wer_mean += wer
            # cer_mean += cer
            # logger.info('  WER / CER (%s): %.3f / %.3f %%' %
            #             (data_type, wer, cer))
            # logger.info(df)
        else:
            raise ValueError(config_p2w['label_type'])

    if config_p2w['label_type'] == 'word':
        logger.info('  WER (mean): %.3f %%\n' %
                    (wer_mean / len(args.eval_sets)))
    elif 'character' in config_p2w['label_type']:
        logger.info('  WER / CER (mean): %.3f / %.3f %%\n' %
                    (wer_mean / len(args.eval_sets),
                     cer_mean / len(args.eval_sets)))


if __name__ == '__main__':
    main()
