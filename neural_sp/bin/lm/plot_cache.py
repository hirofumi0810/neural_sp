#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Plot cache distributions of RNNLM."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import shutil

from neural_sp.bin.args_lm import parse
from neural_sp.bin.plot_utils import plot_cache_weights
from neural_sp.bin.train_utils import load_config
from neural_sp.bin.train_utils import set_logger
from neural_sp.bin.train_utils import load_checkpoint
from neural_sp.datasets.loader_lm import Dataset
from neural_sp.models.lm.build import build_lm
from neural_sp.utils import mkdir_join


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
    if os.path.isfile(os.path.join(args.recog_dir, 'plot.log')):
        os.remove(os.path.join(args.recog_dir, 'plot.log'))
    logger = set_logger(os.path.join(args.recog_dir, 'plot.log'),
                        key='decoding', stdout=args.recog_stdout)

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
            model = build_lm(args, dir_name)
            model = load_checkpoint(model, args.recog_model[0])[0]
            epoch = int(args.recog_model[0].split('-')[-1])

            logger.info('epoch: %d' % (epoch - 1))
            logger.info('batch size: %d' % args.recog_batch_size)
            # logger.info('recog unit: %s' % args.recog_unit)
            # logger.info('ensemble: %d' % (len(ensemble_models)))
            logger.info('BPTT: %d' % (args.bptt))
            logger.info('cache size: %d' % (args.recog_n_caches))
            logger.info('cache theta: %.3f' % (args.recog_cache_theta))
            logger.info('cache lambda: %.3f' % (args.recog_cache_lambda))
            model.cache_theta = args.recog_cache_theta
            model.cache_lambda = args.recog_cache_lambda

            # GPU setting
            model.cuda()

        assert args.recog_n_caches > 0
        save_path = mkdir_join(args.recog_dir, 'cache')

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        hidden = None
        fig_count = 0
        toknen_count = 0
        n_tokens = args.recog_n_caches
        while True:
            ys, is_new_epoch = dataset.next()

            for t in range(ys.shape[1] - 1):
                loss, hidden = model(ys[:, t:t + 2], hidden, is_eval=True, n_caches=args.recog_n_caches)[:2]

                if len(model.cache_attn) > 0:
                    if toknen_count == n_tokens:
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
                        toknen_count = 0
                        fig_count += 1
                    else:
                        toknen_count += 1

            if is_new_epoch:
                break


if __name__ == '__main__':
    main()
