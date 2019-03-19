#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


"""Evaluate the RNNLM."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from neural_sp.bin.args_lm import parse
from neural_sp.bin.train_utils import load_config
from neural_sp.bin.train_utils import set_logger
from neural_sp.datasets.loader_lm import Dataset
from neural_sp.evaluators.ppl import eval_ppl
from neural_sp.models.rnnlm.rnnlm import RNNLM
from neural_sp.models.rnnlm.rnnlm_seq import SeqRNNLM


def main():

    args = parse()

    # Load a conf file
    conf = load_config(os.path.join(args.recog_model[0], 'conf.yml'))

    # Overwrite conf
    for k, v in conf.items():
        if 'recog' not in k:
            setattr(args, k, v)

    # Setting for logging
    if os.path.isfile(os.path.join(args.recog_dir, 'decode.log')):
        os.remove(os.path.join(args.recog_dir, 'decode.log'))
    logger = set_logger(os.path.join(args.recog_dir, 'decode.log'), key='decoding')

    ppl_avg = 0
    for i, s in enumerate(args.recog_sets):
        # Load dataset
        dataset = Dataset(corpus=args.corpus,
                          tsv_path=s,
                          dict_path=os.path.join(args.recog_model[0], 'dict.txt'),
                          wp_model=os.path.join(args.recog_model[0], 'wp.model'),
                          unit=args.unit,
                          batch_size=args.recog_batch_size,
                          bptt=args.bptt,
                          serialize=args.serialize,
                          is_test=True)

        if i == 0:
            # Load the RNNLM
            seq_rnnlm = SeqRNNLM(args)
            epoch = seq_rnnlm.load_checkpoint(args.recog_model[0])['epoch']
            rnnlm = seq_rnnlm

            # Copy parameters
            # rnnlm = RNNLM(args)
            # rnnlm.copy_from_seqrnnlm(seq_rnnlm)
            rnnlm.save_path = args.recog_model[0]

            logger.info('epoch: %d' % (epoch - 1))
            logger.info('batch size: %d' % args.recog_batch_size)
            # logger.info('recog unit: %s' % args.recog_unit)
            # logger.info('ensemble: %d' % (len(ensemble_models)))
            logger.info('BPTT: %d' % (args.bptt))
            logger.info('cache size: %d' % (args.recog_n_caches))
            logger.info('cache theta: %d' % (args.recog_cache_theta))
            logger.info('cache lambda: %d' % (args.recog_cache_lambda))
            rnnlm.cache_theta = args.recog_cache_theta
            rnnlm.cache_lambda = args.recog_cache_lambda

            # GPU setting
            rnnlm.cuda()

        start_time = time.time()

        # TODO(hirofumi): ensemble
        ppl = eval_ppl([rnnlm], dataset, batch_size=1, bptt=args.bptt,
                       n_caches=args.recog_n_caches, progressbar=True)
        ppl_avg += ppl
        logger.info('PPL (%s): %.3f' % (dataset.set, ppl))

        logger.info('Elasped time: %.2f [sec]:' % (time.time() - start_time))

    logger.info('PPL (avg.): %.3f\n' % (ppl_avg / len(args.recog_sets)))


if __name__ == '__main__':
    main()
