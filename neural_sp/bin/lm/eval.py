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

    # Load a config file
    config = load_config(os.path.join(args.recog_model[0], 'config.yml'))

    # Overwrite config
    for k, v in config.items():
        if 'recog' not in k:
            setattr(args, k, v)

    # Setting for logging
    if os.path.isfile(os.path.join(args.recog_dir, 'decode.log')):
        os.remove(os.path.join(args.recog_dir, 'decode.log'))
    logger = set_logger(os.path.join(args.recog_dir, 'decode.log'), key='decoding')

    ppl_mean = 0
    for i, set in enumerate(args.recog_sets):
        # Load dataset
        dataset = Dataset(csv_path=set,
                          dict_path=os.path.join(args.recog_model[0], 'dict.txt'),
                          wp_model=os.path.join(args.recog_model[0], 'wp.model'),
                          unit=args.unit,
                          batch_size=args.recog_batch_size,
                          is_test=True)

        if i == 0:
            args.vocab = dataset.vocab

            # Load the RNNLM
            # seq_rnnlm = SeqRNNLM(args)
            seq_rnnlm = SeqRNNLM(args)
            epoch, _, _, _ = seq_rnnlm.load_checkpoint(args.recog_model[0], epoch=args.recog_epoch)
            rnnlm = seq_rnnlm

            # Copy parameters
            # rnnlm = RNNLM(args)
            # rnnlm.copy_from_seqrnnlm(seq_rnnlm)
            rnnlm.save_path = args.recog_model[0]

            logger.info('epoch: %d' % (epoch - 1))
            logger.info('batch size: %d' % args.recog_batch_size)
            # logger.info('recog unit: %s' % args.recog_unit)
            # logger.info('ensemble: %d' % (len(ensemble_models)))
            # logger.info('checkpoint ensemble: %d' % (args.recog_checkpoint_ensemble))
            logger.info('cache size: %d' % (args.recog_ncaches))

            # GPU setting
            rnnlm.cuda()

        start_time = time.time()

        # TODO(hirofumi): ensemble
        ppl = eval_ppl([rnnlm], dataset, args.bptt, progressbar=True)
        ppl_mean += ppl
        logger.info('PPL (%s): %.3f' % (dataset.set, ppl))

        logger.info('Elasped time: %.2f [sec]:' % (time.time() - start_time))

    logger.info('PPL (mean): %.3f\n' % (ppl_mean / len(args.recog_sets)))


if __name__ == '__main__':
    main()
