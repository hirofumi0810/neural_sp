#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate LM."""

import logging
import os
import sys
import time

from neural_sp.bin.args_lm import parse_args_eval
from neural_sp.bin.train_utils import (
    load_checkpoint,
    set_logger
)
from neural_sp.datasets.lm import Dataset
from neural_sp.evaluators.ppl import eval_ppl
from neural_sp.models.lm.build import build_lm

logger = logging.getLogger(__name__)


def main():

    # Load configuration
    args, dir_name = parse_args_eval(sys.argv[1:])

    # Setting for logging
    if os.path.isfile(os.path.join(args.recog_dir, 'decode.log')):
        os.remove(os.path.join(args.recog_dir, 'decode.log'))
    set_logger(os.path.join(args.recog_dir, 'decode.log'), stdout=args.recog_stdout)

    # Load the LM
    model = build_lm(args)
    load_checkpoint(args.recog_model[0], model)
    # NOTE: model averaging is not helpful for LM

    logger.info('batch size: %d' % args.recog_batch_size)
    logger.info('BPTT: %d' % (args.bptt))
    logger.info('cache size: %d' % (args.recog_n_caches))
    logger.info('cache theta: %.3f' % (args.recog_cache_theta))
    logger.info('cache lambda: %.3f' % (args.recog_cache_lambda))

    model.cache_theta = args.recog_cache_theta
    model.cache_lambda = args.recog_cache_lambda

    # GPU setting
    if args.recog_n_gpus > 0:
        model.cuda()

    ppl_avg = 0
    for s in args.recog_sets:
        # Load dataset
        dataset = Dataset(corpus=args.corpus,
                          tsv_path=s,
                          batch_size=args.recog_batch_size,
                          bptt=args.bptt,
                          backward=args.backward,
                          serialize=args.serialize,
                          is_test=True)

        start_time = time.time()

        ppl, _ = eval_ppl([model], dataset, batch_size=args.recog_batch_size, bptt=args.bptt,
                          n_caches=args.recog_n_caches, progressbar=True)
        ppl_avg += ppl
        print('PPL (%s): %.2f' % (dataset.set, ppl))
        logger.info('Elapsed time: %.2f [sec]:' % (time.time() - start_time))

    logger.info('PPL (avg.): %.2f\n' % (ppl_avg / len(args.recog_sets)))


if __name__ == '__main__':
    main()
