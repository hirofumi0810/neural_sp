#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the RNNLM."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from src.models.load_model import load
from src.dataset.loader_lm import Dataset
from src.metrics.lm import eval_ppl
from src.utils.config import load_config
from src.utils.evaluation.logging import set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str,
                    help='the name of corpus')
parser.add_argument('--eval_sets', type=str, nargs='+',
                    help='evaluation sets')
parser.add_argument('--data_save_path', type=str,
                    help='path to saved data')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
args = parser.parse_args()


def main():

    # Load a config file
    config = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Setting for logging
    logger = set_logger(args.model_path)

    ppl_mean = 0
    for i, data_type in enumerate(args.eval_sets):
        # Load dataset
        eval_set = Dataset(
            corpus=args.corpus,
            data_save_path=args.data_save_path,
            model_type=config['model_type'],
            data_size=config['data_size'] if 'data_size' in config.keys(
            ) else '',
            data_type=data_type,
            label_type=config['label_type'],
            batch_size=args.eval_batch_size,
            shuffle=False, tool=config['tool'],
            vocab=config['vocab'])

        if i == 0:
            config['num_classes'] = eval_set.num_classes

            # Load model
            model = load(model_type=config['model_type'],
                         config=config,
                         backend=config['backend'])

            # NOTE: after load the rnn config are not a continuous chunk of memory
            # this makes them a continuous chunk, and will speed up forward pass
            model.flatten_parameters()
            # https://github.com/pytorch/examples/blob/master/word_language_model/main.py

            # Restore the saved parameters
            epoch, _, _, _ = model.load_checkpoint(
                save_path=args.model_path, epoch=args.epoch)

            # GPU setting
            model.set_cuda(deterministic=False, benchmark=True)

            logger.info('epoch: %d' % (epoch - 1))

        ppl = eval_ppl(models=[model],
                       dataset=eval_set,
                       bptt=config['bptt'],
                       progressbar=True)
        ppl_mean += ppl
        logger.info('  PPL (%s): %.3f' % (data_type, ppl))

    logger.info('PPL (mean): %.3f\n' % (ppl_mean / len(args.eval_sets)))


if __name__ == '__main__':
    main()
