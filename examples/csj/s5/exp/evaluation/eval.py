#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the ASR model (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.csj.s5.exp.dataset.load_dataset import Dataset
from examples.csj.s5.exp.metrics.character import eval_char
from examples.csj.s5.exp.metrics.word import eval_word
from utils.config import load_config
from utils.evaluation.logging import set_logger

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
                    help='length penalty in beam search decoding')
parser.add_argument('--coverage_penalty', type=float, default=0,
                    help='coverage penalty in beam search decoding')
parser.add_argument('--rnnlm_weight', type=float, default=0,
                    help='the weight of RNNLM score in beam search decoding')
parser.add_argument('--rnnlm_path', default=None, type=str, nargs='?',
                    help='path to the RMMLM')

MAX_DECODE_LEN_WORD = 100
MIN_DECODE_LEN_WORD = 1
MAX_DECODE_LEN_CHAR = 200
MIN_DECODE_LEN_CHAR = 1


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Setting for logging
    logger = set_logger(args.model_path)

    wer_mean, cer_mean = 0, 0
    for i, data_type in enumerate(['eval1', 'eval2', 'eval3']):
        # Load dataset
        dataset = Dataset(
            data_save_path=args.data_save_path,
            input_freq=params['input_freq'],
            use_delta=params['use_delta'],
            use_double_delta=params['use_double_delta'],
            data_type=data_type,
            data_size=params['data_size'],
            label_type=params['label_type'],
            batch_size=args.eval_batch_size,
            shuffle=False, tool=params['tool'])

        if i == 0:
            params['num_classes'] = dataset.num_classes

            # Load the ASR model
            model = load(model_type=params['model_type'],
                         params=params,
                         backend=params['backend'])

            # Restore the saved parameters
            epoch, _, _, _ = model.load_checkpoint(
                save_path=args.model_path, epoch=args.epoch)

            if args.rnnlm_path is not None and args.rnnlm_weight > 0:
                # Load a config file (.yml)
                params_rnnlm = load_config(
                    join(args.rnnlm_path, 'config.yml'), is_eval=True)

                assert params['label_type'] == params_rnnlm['label_type']
                params_rnnlm['num_classes'] = dataset.num_classes

                # Load RNLM
                rnnlm = load(model_type=params_rnnlm['model_type'],
                             params=params_rnnlm,
                             backend=params_rnnlm['backend'])

                # Restore the saved parameters
                rnnlm.load_checkpoint(save_path=args.rnnlm_path, epoch=-1)
                # NOTE: load the best model

                # NOTE: after load the rnn params are not a continuous chunk of memory
                # this makes them a continuous chunk, and will speed up forward pass
                rnnlm.rnn.flatten_parameters()
                # https://github.com/pytorch/examples/blob/master/word_language_model/main.py

                # Resister to the ASR model
                model.rnnlm_0 = rnnlm

            # GPU setting
            model.set_cuda(deterministic=False, benchmark=True)

            logger.info('beam width: %d' % args.beam_width)
            if args.rnnlm_path is not None:
                logger.info('RNNLM path: %s' % args.rnnlm_path)
            logger.info('RNNLM weight: %.3f' % args.rnnlm_weight)
            logger.info('epoch: %d' % (epoch - 1))

        if params['label_type'] == 'word':
            wer, df = eval_word(
                models=[model],
                dataset=dataset,
                eval_batch_size=args.eval_batch_size,
                beam_width=args.beam_width,
                max_decode_len=MAX_DECODE_LEN_WORD,
                min_decode_len=MIN_DECODE_LEN_WORD,
                length_penalty=args.length_penalty,
                coverage_penalty=args.coverage_penalty,
                rnnlm_weight=args.rnnlm_weight,
                progressbar=True)
            wer_mean += wer
            logger.info('  WER (%s): %.3f %%' % (data_type, (wer * 100)))
            logger.info(df)
        else:
            wer, cer, df = eval_char(
                models=[model],
                dataset=dataset,
                eval_batch_size=args.eval_batch_size,
                beam_width=args.beam_width,
                max_decode_len=MAX_DECODE_LEN_CHAR,
                min_decode_len=MIN_DECODE_LEN_CHAR,
                length_penalty=args.length_penalty,
                coverage_penalty=args.coverage_penalty,
                rnnlm_weight=args.rnnlm_weight,
                progressbar=True)
            wer_mean += wer
            cer_mean += cer
            logger.info(' WER / CER (%s, sub): %.3f / %.3f %%' %
                        (data_type, (wer * 100), (cer * 100)))
            logger.info(df)

    if params['label_type'] == 'word':
        logger.info('  WER (mean): %.3f %%' % (wer_mean * 100 / 3))
    else:
        logger.info('  WER / CER (mean): %.3f / %.3f %%' %
                    ((wer_mean * 100 / 3), (cer_mean * 100 / 3)))


if __name__ == '__main__':
    main()
