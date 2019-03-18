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
from neural_sp.bin.asr.plot_utils import plot_cache_weights
from neural_sp.bin.train_utils import load_config
from neural_sp.bin.train_utils import set_logger
from neural_sp.datasets.loader_lm import Dataset
from neural_sp.models.rnnlm.rnnlm_seq import SeqRNNLM
from neural_sp.utils.general import mkdir_join


def main():

    args = parse()

    # Load a conf file
    conf = load_config(os.path.join(args.recog_model[0], 'conf.yml'))

    # Overwrite conf
    for k, v in conf.items():
        if 'recog' not in k:
            setattr(args, k, v)

    # Setting for logging
    if os.path.isfile(os.path.join(args.recog_dir, 'plot.log')):
        os.remove(os.path.join(args.recog_dir, 'plot.log'))
    logger = set_logger(os.path.join(args.recog_dir, 'plot.log'), key='decoding')

    for i, set in enumerate(args.recog_sets):
        # Load dataset
        dataset = Dataset(tsv_path=set,
                          dict_path=os.path.join(args.recog_model[0], 'dict.txt'),
                          wp_model=os.path.join(args.recog_model[0], 'wp.model'),
                          unit=args.unit,
                          batch_size=args.recog_batch_size,
                          is_test=True)

        if i == 0:
            # Load the RNNLM
            seq_rnnlm = SeqRNNLM(args)
            epoch, _, _, _ = seq_rnnlm.load_checkpoint(args.recog_model[0])
            rnnlm = seq_rnnlm
            rnnlm.save_path = args.recog_model[0]

            logger.info('epoch: %d' % (epoch - 1))
            logger.info('batch size: %d' % args.recog_batch_size)
            # logger.info('recog unit: %s' % args.recog_unit)
            # logger.info('ensemble: %d' % (len(ensemble_models)))
            # logger.info('checkpoint ensemble: %d' % (args.recog_checkpoint_ensemble))
            logger.info('cache size: %d' % (args.recog_n_caches))
            logger.info('cache theta: %d' % (args.recog_cache_theta))
            logger.info('cache lambda: %d' % (args.recog_cache_lambda))
            rnnlm.cache_theta = args.recog_cache_theta
            rnnlm.cache_lambda = args.recog_cache_lambda

            # GPU setting
            rnnlm.cuda()

        assert args.recog_n_caches > 0
        save_path = mkdir_join(args.recog_dir, 'cache_dist')

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        if args.unit == 'word':
            id2token = dataset.id2word
        elif args.unit == 'wp':
            id2token = dataset.id2wp
        elif args.unit == 'char':
            id2token = dataset.id2char
        elif args.unit == 'phone':
            id2token = dataset.id2phone
        else:
            raise NotImplementedError(args.unit)

        hidden = None
        counter = 0
        n_tokens = 30
        while True:
            ys, is_new_epoch = dataset.next()

            for t in range(ys.shape[1] - 1):
                loss, hidden = rnnlm(ys[:, t:t + 2], hidden, is_eval=True, n_caches=args.recog_n_caches)[:2]

                if len(rnnlm.cache_attn) > 0:
                    if counter == n_tokens:
                        token_list_keys = id2token(rnnlm.cache_keys[:args.recog_n_caches], return_list=True)
                        token_list_query = id2token(rnnlm.cache_keys[-n_tokens:], return_list=True)

                        # Slide attention matrix
                        n_keys = len(token_list_keys)
                        n_queries = len(token_list_query)
                        cache_probs = np.zeros((n_keys, n_queries))  # `[n_keys, n_queries]`
                        mask = np.zeros((n_keys, n_queries))
                        for i, aw in enumerate(rnnlm.cache_attn[-n_tokens:]):
                            cache_probs[:(n_keys - n_queries + i + 1), i] = aw[0, -(n_keys - n_queries + i + 1):]
                            mask[(n_keys - n_queries + i + 1):, i] = 1

                        plot_cache_weights(
                            cache_probs,
                            keys=token_list_keys,
                            queries=token_list_query,
                            save_path=mkdir_join(save_path, ''.join(token_list_keys[:5]) + '.png'),
                            figsize=(40, 16),
                            mask=mask)
                        counter = 0
                    else:
                        counter += 1

            if is_new_epoch:
                break


if __name__ == '__main__':
    main()
