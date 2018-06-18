#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the ASR model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.librispeech.s5.exp.dataset.load_dataset import Dataset
from examples.librispeech.s5.exp.metrics.character import eval_char
from examples.librispeech.s5.exp.metrics.word import eval_word
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
                    help='length penalty')
parser.add_argument('--coverage_penalty', type=float, default=0,
                    help='coverage penalty')
parser.add_argument('--rnnlm_weight', type=float, default=0,
                    help='the weight of RNNLM score')
parser.add_argument('--rnnlm_path', default=None, type=str, nargs='?',
                    help='path to the RMMLM')

MAX_DECODE_LEN_WORD = 200
MIN_DECODE_LEN_WORD = 1
MIN_DECODE_LEN_RATIO_WORD = 0
MAX_DECODE_LEN_CHAR = 600
MIN_DECODE_LEN_CHAR = 1
MIN_DECODE_LEN_RATIO_CHAR = 0.2


def main():

    args = parser.parse_args()

    # Load a ASR config file
    config = load_config(join(args.model_path, 'config.yml'), is_eval=True)
    config['data_size'] = str(config['data_size'])

    logger = set_logger(args.model_path)

    for i, data_type in enumerate(['dev_clean', 'dev_other', 'test_clean', 'test_other']):
        # Load dataset
        dataset = Dataset(data_save_path=args.data_save_path,
                          input_freq=config['input_freq'],
                          use_delta=config['use_delta'],
                          use_double_delta=config['use_double_delta'],
                          data_type=data_type,
                          data_size=config['data_size'],
                          label_type=config['label_type'],
                          batch_size=args.eval_batch_size,
                          tool=config['tool'])

        if i == 0:
            config['num_classes'] = dataset.num_classes

            # For cold fusion
            if config['rnnlm_fusion_type'] and config['rnnlm_path']:
                # Load a RNNLM config file
                config['rnnlm_config'] = load_config(
                    join(args.model_path, 'config_rnnlm.yml'))

                assert config['label_type'] == config['rnnlm_config']['label_type']
                assert args.rnnlm_weight > 0
                config['rnnlm_config']['num_classes'] = dataset.num_classes
                logger.info('RNNLM path: %s' % config['rnnlm_path'])
                logger.info('RNNLM weight: %.3f' % args.rnnlm_weight)
            else:
                config['rnnlm_config'] = None

            # Load the ASR model
            model = load(model_type=config['model_type'],
                         config=config,
                         backend=config['backend'])

            # Restore the saved parameters
            epoch, _, _, _ = model.load_checkpoint(
                save_path=args.model_path, epoch=args.epoch)

            # For shallow fusion
            if not (config['rnnlm_fusion_type'] and config['rnnlm_path']) and args.rnnlm_path is not None and args.rnnlm_weight > 0:
                # Load a RNNLM config file
                config_rnnlm = load_config(
                    join(args.rnnlm_path, 'config.yml'), is_eval=True)
                assert config['label_type'] == config_rnnlm['label_type']
                config_rnnlm['num_classes'] = dataset.num_classes

                # Load the pre-trianed RNNLM
                rnnlm = load(model_type=config_rnnlm['model_type'],
                             config=config_rnnlm,
                             backend=config_rnnlm['backend'])
                rnnlm.load_checkpoint(save_path=args.rnnlm_path, epoch=-1)
                rnnlm.rnn.flatten_parameters()
                if config_rnnlm['backward']:
                    model.rnnlm_0_bwd = rnnlm
                else:
                    model.rnnlm_0_fwd = rnnlm
                logger.info('RNNLM path: %s' % args.rnnlm_path)
                logger.info('RNNLM weight: %.3f' % args.rnnlm_weight)

            # GPU setting
            model.set_cuda(deterministic=False, benchmark=True)

            logger.info('beam width: %d' % args.beam_width)
            logger.info('epoch: %d' % (epoch - 1))

        if config['label_type'] == 'word':
            wer, df = eval_word(models=[model],
                                dataset=dataset,
                                eval_batch_size=args.eval_batch_size,
                                beam_width=args.beam_width,
                                max_decode_len=MAX_DECODE_LEN_WORD,
                                min_decode_len=MIN_DECODE_LEN_WORD,
                                min_decode_len_ratio=MIN_DECODE_LEN_RATIO_WORD,
                                length_penalty=args.length_penalty,
                                coverage_penalty=args.coverage_penalty,
                                rnnlm_weight=args.rnnlm_weight,
                                progressbar=True)
            logger.info('  WER (%s): %.3f %%' % (data_type, wer))
            logger.info(df)
        else:
            wer, cer, df = eval_char(models=[model],
                                     dataset=dataset,
                                     eval_batch_size=args.eval_batch_size,
                                     beam_width=args.beam_width,
                                     max_decode_len=MAX_DECODE_LEN_CHAR,
                                     min_decode_len=MIN_DECODE_LEN_CHAR,
                                     min_decode_len_ratio=MIN_DECODE_LEN_RATIO_CHAR,
                                     length_penalty=args.length_penalty,
                                     coverage_penalty=args.coverage_penalty,
                                     rnnlm_weight=args.rnnlm_weight,
                                     progressbar=True)
            logger.info('  WER / CER (%s): %.3f / %.3f %%' %
                        (data_type, wer, cer))
            logger.info(df)


if __name__ == '__main__':
    main()
