#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Plot cache distributions of RNNLM."""

import logging
import numpy as np
import os
import shutil
import sys

from neural_sp.bin.args_lm import parse_args_eval
from neural_sp.bin.plot_utils import plot_cache_weights
from neural_sp.bin.train_utils import (
    load_checkpoint,
    set_logger
)
from neural_sp.datasets.lm import Dataset
from neural_sp.models.lm.build import build_lm
from neural_sp.utils import mkdir_join

logger = logging.getLogger(__name__)


def main():

    # Load configuration
    args, _, dir_name = parse_args_eval(sys.argv[1:])

    # Setting for logging
    if os.path.isfile(os.path.join(args.recog_dir, 'plot.log')):
        os.remove(os.path.join(args.recog_dir, 'plot.log'))
    set_logger(os.path.join(args.recog_dir, 'plot.log'), stdout=args.recog_stdout)

    # Load the LM
    model = build_lm(args, dir_name)
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

    for s in args.recog_sets:
        # Load dataset
        dataset = Dataset(corpus=args.corpus,
                          tsv_path=s,
                          batch_size=args.recog_batch_size,
                          bptt=args.bptt,
                          backward=args.backward,
                          serialize=args.serialize,
                          is_test=True)

        assert args.recog_n_caches > 0
        save_path = mkdir_join(args.recog_dir, 'cache')

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        hidden = None
        fig_count = 0
        token_count = 0
        n_tokens = args.recog_n_caches
        while True:
            ys, is_new_epoch = dataset.next()

            for t in range(ys.shape[1] - 1):
                loss, hidden = model(ys[:, t:t + 2], hidden, is_eval=True, n_caches=args.recog_n_caches)[:2]

                if len(model.cache_attn) > 0:
                    if token_count == n_tokens:
                        tokens_keys = dataset.idx2token[0](model.cache_ids[:args.recog_n_caches], return_list=True)
                        tokens_query = dataset.idx2token[0](model.cache_ids[-n_tokens:], return_list=True)

                        # Slide attention matrix
                        n_keys = len(tokens_keys)
                        n_queries = len(tokens_query)
                        cache_probs = np.zeros((n_keys, n_queries))  # `[n_keys, n_queries]`
                        mask = np.zeros((n_keys, n_queries))
                        for i, aw in enumerate(model.cache_attn[-n_tokens:]):
                            cache_probs[:(n_keys - n_queries + i + 1), i] = aw[0, -(n_keys - n_queries + i + 1):]
                            mask[(n_keys - n_queries + i + 1):, i] = 1

                        plot_cache_weights(
                            cache_probs,
                            keys=tokens_keys,
                            queries=tokens_query,
                            save_path=mkdir_join(save_path, str(fig_count) + '.png'),
                            figsize=(40, 16),
                            mask=mask)
                        token_count = 0
                        fig_count += 1
                    else:
                        token_count += 1

            if is_new_epoch:
                break


if __name__ == '__main__':
    main()
