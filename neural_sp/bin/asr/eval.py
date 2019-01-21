#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


"""Evaluate the ASR model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import os
import time

from neural_sp.bin.asr.args import parse
from neural_sp.bin.train_utils import load_config
from neural_sp.bin.train_utils import set_logger
from neural_sp.datasets.loader_asr import Dataset
from neural_sp.evaluators.character import eval_char
from neural_sp.evaluators.phone import eval_phone
from neural_sp.evaluators.word import eval_word
from neural_sp.evaluators.wordpiece import eval_wordpiece
from neural_sp.models.rnnlm.rnnlm import RNNLM
from neural_sp.models.rnnlm.rnnlm_seq import SeqRNNLM
from neural_sp.models.seq2seq.seq2seq import Seq2seq


def main():

    args = parse()

    # Load a config file
    config = load_config(os.path.join(args.recog_model[0], 'config.yml'))

    # Overwrite config
    for k, v in config.items():
        if 'recog' not in k:
            setattr(args, k, v)
    decode_params = vars(args)

    # Setting for logging
    if os.path.isfile(os.path.join(args.recog_dir, 'decode.log')):
        os.remove(os.path.join(args.recog_dir, 'decode.log'))
    logger = set_logger(os.path.join(args.recog_dir, 'decode.log'), key='decoding')

    wer_mean, cer_mean, per_mean = 0, 0, 0
    for i, set in enumerate(args.recog_sets):
        # Load dataset
        dataset = Dataset(csv_path=set,
                          dict_path=os.path.join(args.recog_model[0], 'dict.txt'),
                          dict_path_sub1=os.path.join(args.recog_model[0], 'dict_sub1.txt') if os.path.isfile(
                              os.path.join(args.recog_model[0], 'dict_sub1.txt')) else None,
                          dict_path_sub2=os.path.join(args.recog_model[0], 'dict_sub2.txt') if os.path.isfile(
                              os.path.join(args.recog_model[0], 'dict_sub2.txt')) else None,
                          dict_path_sub3=os.path.join(args.recog_model[0], 'dict_sub3.txt') if os.path.isfile(
                              os.path.join(args.recog_model[0], 'dict_sub3.txt')) else None,
                          wp_model=os.path.join(args.recog_model[0], 'wp.model'),
                          wp_model_sub1=os.path.join(args.recog_model[0], 'wp_sub1.model'),
                          wp_model_sub2=os.path.join(args.recog_model[0], 'wp_sub2.model'),
                          wp_model_sub3=os.path.join(args.recog_model[0], 'wp_sub3.model'),
                          unit=args.unit,
                          unit_sub1=args.unit_sub1,
                          unit_sub2=args.unit_sub2,
                          unit_sub3=args.unit_sub3,
                          batch_size=args.recog_batch_size,
                          is_test=True)

        if i == 0:
            args.vocab = dataset.vocab
            args.vocab_sub1 = dataset.vocab_sub1
            args.vocab_sub2 = dataset.vocab_sub2
            args.vocab_sub3 = dataset.vocab_sub3
            args.input_dim = dataset.input_dim

            # For cold fusion
            # if args.rnnlm_cold_fusion:
            #     # Load a RNNLM config file
            #     config['rnnlm_config'] = load_config(os.path.join(args.recog_model[0], 'config_rnnlm.yml'))
            #
            #     assert args.unit == config['rnnlm_config']['unit']
            #     rnnlm_args.vocab = dataset.vocab
            #     logger.info('RNNLM path: %s' % config['rnnlm'])
            #     logger.info('RNNLM weight: %.3f' % args.rnnlm_weight)
            # else:
            #     pass

            args.rnnlm_cold_fusion = None
            args.rnnlm_init = None

            # Load the ASR model
            model = Seq2seq(args)
            epoch, _, _, _ = model.load_checkpoint(args.recog_model[0], epoch=args.recog_epoch)
            model.save_path = args.recog_model[0]

            # ensemble
            ensemble_models = [model]
            if len(args.recog_model) > 1:
                for recog_model_e in args.recog_model[1:]:
                    # Load a config file
                    config_e = load_config(os.path.join(recog_model_e, 'config.yml'))

                    # Overwrite config
                    args_e = copy.deepcopy(args)
                    for k, v in config_e.items():
                        if 'recog' not in k:
                            setattr(args_e, k, v)

                    model_e = Seq2seq(args_e)
                    model_e.load_checkpoint(recog_model_e, epoch=args.recog_epoch)
                    model_e.cuda()
                    ensemble_models += [model_e]

            # For shallow fusion
            if not args.rnnlm_cold_fusion:
                if args.recog_rnnlm is not None and args.recog_rnnlm_weight > 0:
                    # Load a RNNLM config file
                    config_rnnlm = load_config(os.path.join(args.recog_rnnlm, 'config.yml'))

                    # Merge config with args
                    args_rnnlm = argparse.Namespace()
                    for k, v in config_rnnlm.items():
                        setattr(args_rnnlm, k, v)

                    assert args.unit == args_rnnlm.unit
                    args_rnnlm.vocab = dataset.vocab

                    # Load the pre-trianed RNNLM
                    seq_rnnlm = SeqRNNLM(args_rnnlm)
                    seq_rnnlm.load_checkpoint(args.recog_rnnlm, epoch=-1)

                    # Copy parameters
                    rnnlm = RNNLM(args_rnnlm)
                    rnnlm.copy_from_seqrnnlm(seq_rnnlm)

                    # Register to the ASR model
                    if args_rnnlm.backward:
                        model.rnnlm_bwd = rnnlm
                    else:
                        model.rnnlm_fwd = rnnlm

                if args.recog_rnnlm_bwd is not None and args.recog_rnnlm_weight > 0 and args.recog_fwd_bwd_attention:
                    # Load a RNNLM config file
                    config_rnnlm = load_config(os.path.join(args.recog_rnnlm_bwd, 'config.yml'))

                    # Merge config with args
                    args_rnnlm_bwd = argparse.Namespace()
                    for k, v in config_rnnlm.items():
                        setattr(args_rnnlm_bwd, k, v)

                    assert args.unit == args_rnnlm_bwd.unit
                    args_rnnlm_bwd.vocab = dataset.vocab

                    # Load the pre-trianed RNNLM
                    seq_rnnlm_bwd = SeqRNNLM(args_rnnlm_bwd)
                    seq_rnnlm_bwd.load_checkpoint(args.recog_rnnlm_bwd, epoch=-1)

                    # Copy parameters
                    rnnlm_bwd = RNNLM(args_rnnlm_bwd)
                    rnnlm_bwd.copy_from_seqrnnlm(seq_rnnlm_bwd)

                    # Resister to the ASR model
                    model.rnnlm_bwd = rnnlm_bwd

            logger.info('epoch: %d' % (epoch - 1))
            logger.info('batch size: %d' % args.recog_batch_size)
            logger.info('beam width: %d' % args.recog_beam_width)
            logger.info('min length ratio: %.3f' % args.recog_min_len_ratio)
            logger.info('max length ratio: %.3f' % args.recog_max_len_ratio)
            logger.info('length penalty: %.3f' % args.recog_length_penalty)
            logger.info('coverage penalty: %.3f' % args.recog_coverage_penalty)
            logger.info('coverage threshold: %.3f' % args.recog_coverage_threshold)
            logger.info('CTC weight: %.3f' % args.recog_ctc_weight)
            logger.info('RNNLM path: %s' % args.recog_rnnlm)
            logger.info('RNNLM path (bwd): %s' % args.recog_rnnlm_bwd)
            logger.info('RNNLM weight: %.3f' % args.recog_rnnlm_weight)
            logger.info('forward-backward attention: %s' % args.recog_fwd_bwd_attention)
            logger.info('resolving UNK: %s' % args.recog_resolving_unk)
            logger.info('recog unit: %s' % args.recog_unit)
            logger.info('ensemble: %d' % (len(ensemble_models)))

            # GPU setting
            model.cuda()

        start_time = time.time()

        if args.unit in ['word', 'word_char'] and not args.recog_unit:
            wer, nsub, nins, ndel, noov_total = eval_word(
                ensemble_models, dataset, decode_params,
                epoch=epoch - 1,
                decode_dir=args.recog_dir,
                progressbar=True)
            wer_mean += wer
            logger.info('WER (%s): %.3f %%' % (dataset.set, wer))
            logger.info('SUB: %.3f / INS: %.3f / DEL: %.3f' % (nsub, nins, ndel))
            logger.info('OOV (total): %d' % (noov_total))

        elif (args.unit == 'wp' and not args.recog_unit) or args.recog_unit == 'wp':
            wer, nsub, nins, ndel = eval_wordpiece(
                ensemble_models, dataset, decode_params,
                epoch=epoch - 1,
                decode_dir=args.recog_dir,
                progressbar=True)
            wer_mean += wer
            logger.info('WER (%s): %.3f %%' % (dataset.set, wer))
            logger.info('SUB: %.3f / INS: %.3f / DEL: %.3f' % (nsub, nins, ndel))

        elif ('char' in args.unit and not args.recog_unit) or 'char' in args.recog_unit:
            (wer, nsub, nins, ndel), (cer, _, _, _) = eval_char(
                ensemble_models, dataset, decode_params,
                epoch=epoch - 1,
                decode_dir=args.recog_dir,
                progressbar=True,
                task_id=1 if args.recog_unit and 'char' in args.recog_unit else 0)
            wer_mean += wer
            cer_mean += cer
            logger.info('WER / CER (%s): %.3f / %.3f %%' % (dataset.set, wer, cer))
            logger.info('SUB: %.3f / INS: %.3f / DEL: %.3f' % (nsub, nins, ndel))

        elif 'phone' in args.unit:
            per, nsub, nins, ndel = eval_phone(
                ensemble_models, dataset, decode_params,
                epoch=epoch - 1,
                decode_dir=args.recog_dir,
                progressbar=True)
            per_mean += per
            logger.info('PER (%s): %.3f %%' % (dataset.set, per))
            logger.info('SUB: %.3f / INS: %.3f / DEL: %.3f' % (nsub, nins, ndel))

        else:
            raise ValueError(args.unit)

        logger.info('Elasped time: %.2f [sec]:' % (time.time() - start_time))

    if args.unit == 'word':
        logger.info('WER (mean): %.3f %%\n' % (wer_mean / len(args.recog_sets)))
    if args.unit == 'wp':
        logger.info('WER (mean): %.3f %%\n' % (wer_mean / len(args.recog_sets)))
    elif 'char' in args.unit:
        logger.info('WER / CER (mean): %.3f / %.3f %%\n' %
                    (wer_mean / len(args.recog_sets), cer_mean / len(args.recog_sets)))
    elif 'phone' in args.unit:
        logger.info('PER (mean): %.3f %%\n' % (per_mean / len(args.recog_sets)))


if __name__ == '__main__':
    main()
