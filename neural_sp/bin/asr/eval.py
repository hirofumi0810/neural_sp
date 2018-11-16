#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


"""Evaluate the ASR model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from distutils.util import strtobool
import os
import time

from neural_sp.datasets.loader_asr import Dataset
from neural_sp.evaluators.character import eval_char
from neural_sp.evaluators.phone import eval_phone
from neural_sp.evaluators.word import eval_word
from neural_sp.evaluators.wordpiece import eval_wordpiece
from neural_sp.models.rnnlm.rnnlm import RNNLM
from neural_sp.models.rnnlm.rnnlm_seq import SeqRNNLM
from neural_sp.models.seq2seq.seq2seq import Seq2seq
from neural_sp.utils.config import load_config
from neural_sp.utils.general import set_logger

parser = argparse.ArgumentParser()
# general
parser.add_argument('--model', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--decode_dir', type=str,
                    help='directory to save decoding results')
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
    logger = set_logger(os.path.join(args.decode_dir, 'decode.log'), key='decoding')

    wer_mean, cer_mean, per_mean = 0, 0, 0
    for i, set in enumerate(args.eval_sets):
        # Load dataset
        eval_set = Dataset(csv_path=set,
                           dict_path=os.path.join(args.model, 'dict.txt'),
                           dict_path_sub=os.path.join(args.model, 'dict_sub.txt') if os.path.isfile(
                               os.path.join(args.model, 'dict_sub.txt')) else None,
                           wp_model=os.path.join(args.model, 'wp.model'),
                           label_type=args.label_type,
                           batch_size=args.batch_size,
                           is_test=True)

        if i == 0:
            args.num_classes = eval_set.num_classes
            args.input_dim = eval_set.input_dim
            args.num_classes_sub = eval_set.num_classes_sub

            # For cold fusion
            # if args.rnnlm_cold_fusion:
            #     # Load a RNNLM config file
            #     config['rnnlm_config'] = load_config(os.path.join(args.model, 'config_rnnlm.yml'))
            #
            #     assert args.label_type == config['rnnlm_config']['label_type']
            #     rnnlm_args.num_classes = eval_set.num_classes
            #     logger.info('RNNLM path: %s' % config['rnnlm'])
            #     logger.info('RNNLM weight: %.3f' % args.rnnlm_weight)
            # else:
            #     pass

            args.rnnlm_cold_fusion = None
            args.rnnlm_init = None

            # Load the ASR model
            model = Seq2seq(args)
            epoch, _, _, _ = model.load_checkpoint(args.model, epoch=args.epoch)

            model.save_path = args.model

            # For shallow fusion
            if (not args.rnnlm_cold_fusion) and args.rnnlm is not None and args.rnnlm_weight > 0:
                # Load a RNNLM config file
                config_rnnlm = load_config(os.path.join(args.rnnlm, 'config.yml'))

                # Merge config with args
                args_rnnlm = argparse.Namespace()
                for k, v in config_rnnlm.items():
                    setattr(args_rnnlm, k, v)

                assert args.label_type == args_rnnlm.label_type
                args_rnnlm.num_classes = eval_set.num_classes

                # Load the pre-trianed RNNLM
                seq_rnnlm = SeqRNNLM(args_rnnlm)
                seq_rnnlm.load_checkpoint(args.rnnlm, epoch=-1)

                # Copy parameters
                rnnlm = RNNLM(args_rnnlm)
                rnnlm.copy_from_seqrnnlm(seq_rnnlm)

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

        start_time = time.time()

        if args.label_type == 'word':
            wer, nsub, nins, ndel = eval_word([model], eval_set, decode_params,
                                              epoch=epoch - 1,
                                              decode_dir=args.decode_dir,
                                              progressbar=True)
            wer_mean += wer
            logger.info('WER (%s): %.3f %%' % (eval_set.set, wer))
            logger.info('SUB / INS / DEL : %.3f / %.3f/ %.3f' % (nsub, nins, ndel))

        elif args.label_type == 'wp':
            wer, nsub, nins, ndel = eval_wordpiece([model], eval_set, decode_params,
                                                   epoch=epoch - 1,
                                                   decode_dir=args.decode_dir,
                                                   progressbar=True)
            wer_mean += wer
            logger.info('WER (%s): %.3f %%' % (eval_set.set, wer))
            logger.info('SUB / INS / DEL : %.3f / %.3f/ %.3f' % (nsub, nins, ndel))

        elif 'char' in args.label_type:
            (wer, nsub, nins, ndel), (cer, _, _, _) = eval_char([model], eval_set, decode_params,
                                                                epoch=epoch - 1,
                                                                decode_dir=args.decode_dir,
                                                                progressbar=True)
            wer_mean += wer
            cer_mean += cer
            logger.info('WER / CER (%s): %.3f / %.3f %%' % (eval_set.set, wer, cer))
            logger.info('SUB / INS / DEL : %.3f / %.3f/ %.3f' % (nsub, nins, ndel))

        elif 'phone' in args.label_type:
            per, nsub, nins, ndel = eval_phone([model], eval_set, decode_params,
                                               epoch=epoch - 1,
                                               decode_dir=args.decode_dir,
                                               progressbar=True)
            per_mean += per
            logger.info('PER (%s): %.3f %%' % (eval_set.set, per))
            logger.info('SUB / INS / DEL : %.3f / %.3f/ %.3f' % (nsub, nins, ndel))
        else:
            raise ValueError(args.label_type)

        logger.info('Elasped time: %.2f [sec]:' % (time.time() - start_time))

    if args.label_type == 'word':
        logger.info('WER (mean): %.3f %%\n' % (wer_mean / len(args.eval_sets)))
    if args.label_type == 'wp':
        logger.info('WER (mean): %.3f %%\n' % (wer_mean / len(args.eval_sets)))
    elif 'char' in args.label_type:
        logger.info('WER / CER (mean): %.3f / %.3f %%\n' %
                    (wer_mean / len(args.eval_sets), cer_mean / len(args.eval_sets)))
    elif 'phone' in args.label_type:
        logger.info('PER (mean): %.3f %%\n' % (per_mean / len(args.eval_sets)))


if __name__ == '__main__':
    main()
