#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the ASR model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.timit.s5.exp.dataset.load_dataset import Dataset
from examples.timit.s5.exp.metrics.phone import eval_phone
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
                    help='length penalty in the beam search decoding')
parser.add_argument('--coverage_penalty', type=float, default=0,
                    help='coverage penalty in the beam search decoding')

MAX_DECODE_LEN_PHONE = 71
MIN_DECODE_LEN_PHONE = 13
# NOTE*
# dev: 13-71
# test: 13-69


def main():

    args = parser.parse_args()

    # Load a config file
    config = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Setting for logging
    logger = set_logger(args.model_path)

    for i, data_type in enumerate(['dev', 'test']):
        # Load dataset
        dataset = Dataset(data_save_path=args.data_save_path,
                          input_freq=config['input_freq'],
                          use_delta=config['use_delta'],
                          use_double_delta=config['use_double_delta'],
                          data_type=data_type,
                          label_type=config['label_type'],
                          batch_size=args.eval_batch_size,
                          sort_utt=False, tool=config['tool'])

        if i == 0:
            config['num_classes'] = dataset.num_classes

            # Load model
            model = load(model_type=config['model_type'],
                         config=config,
                         backend=config['backend'])

            # Restore the saved parameters
            epoch, _, _, _ = model.load_checkpoint(
                save_path=args.model_path, epoch=args.epoch)

            # GPU setting
            model.set_cuda(deterministic=False, benchmark=True)

            logger.info('beam width: %d' % args.beam_width)
            logger.info('epoch: %d' % (epoch - 1))

        per, df = eval_phone(model=model,
                             dataset=dataset,
                             map_file_path='./conf/phones.60-48-39.map',
                             eval_batch_size=args.eval_batch_size,
                             beam_width=args.beam_width,
                             max_decode_len=MAX_DECODE_LEN_PHONE,
                             min_decode_len=MIN_DECODE_LEN_PHONE,
                             length_penalty=args.length_penalty,
                             coverage_penalty=args.coverage_penalty,
                             progressbar=True)
        logger.info('  PER (%s): %.3f %%' % (data_type, (per * 100)))
        logger.info(df)


if __name__ == '__main__':
    main()
