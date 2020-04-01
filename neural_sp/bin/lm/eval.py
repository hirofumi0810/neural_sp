#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate the LM."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time

from neural_sp.bin.args_lm import parse
from neural_sp.bin.train_utils import load_checkpoint
from neural_sp.bin.train_utils import load_config
from neural_sp.bin.train_utils import set_logger
from neural_sp.datasets.lm import Dataset
from neural_sp.evaluators.ppl import eval_ppl
from neural_sp.models.lm.build import build_lm

logger = logging.getLogger(__name__)


def main():

    args = parse()

    # Load a conf file
    dir_name = os.path.dirname(args.recog_model[0])
    conf = load_config(os.path.join(dir_name, 'conf.yml'))

    # Overwrite conf
    for k, v in conf.items():
        if 'recog' not in k:
            setattr(args, k, v)

    # Setting for logging
    if os.path.isfile(os.path.join(args.recog_dir, 'decode.log')):
        os.remove(os.path.join(args.recog_dir, 'decode.log'))
    set_logger(os.path.join(args.recog_dir, 'decode.log'), stdout=args.recog_stdout)

    ppl_avg = 0
    for i, s in enumerate(args.recog_sets):
        # Load dataset
        dataset = Dataset(corpus=args.corpus,
                          tsv_path=s,
                          dict_path=os.path.join(dir_name, 'dict.txt'),
                          wp_model=os.path.join(dir_name, 'wp.model'),
                          unit=args.unit,
                          batch_size=args.recog_batch_size,
                          bptt=args.bptt,
                          backward=args.backward,
                          serialize=args.serialize,
                          is_test=True)

        if i == 0:
            # Load the LM
            model = build_lm(args)
            load_checkpoint(model, args.recog_model[0])
            epoch = int(args.recog_model[0].split('-')[-1])
            # NOTE: model averaging is not helpful for LM

            logger.info('epoch: %d' % epoch)
            logger.info('batch size: %d' % args.recog_batch_size)
            logger.info('BPTT: %d' % (args.bptt))
            logger.info('cache size: %d' % (args.recog_n_caches))
            logger.info('cache theta: %.3f' % (args.recog_cache_theta))
            logger.info('cache lambda: %.3f' % (args.recog_cache_lambda))
            logger.info('model average (Transformer): %d' % (args.recog_n_average))
            model.cache_theta = args.recog_cache_theta
            model.cache_lambda = args.recog_cache_lambda

            # GPU setting
            if args.recog_n_gpus > 0:
                model.cuda()

        start_time = time.time()

        ppl, _ = eval_ppl([model], dataset, batch_size=1, bptt=args.bptt,
                          n_caches=args.recog_n_caches, progressbar=True)
        ppl_avg += ppl
        print('PPL (%s): %.2f' % (dataset.set, ppl))
        logger.info('Elasped time: %.2f [sec]:' % (time.time() - start_time))

    logger.info('PPL (avg.): %.2f\n' % (ppl_avg / len(args.recog_sets)))


if __name__ == '__main__':
    main()
