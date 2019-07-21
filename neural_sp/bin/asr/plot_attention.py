#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Plot attention weights of the attention model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import numpy as np
import os
import shutil

from neural_sp.bin.args_asr import parse
from neural_sp.bin.plot_utils import plot_attention_weights
from neural_sp.bin.plot_utils import plot_cache_weights
from neural_sp.bin.train_utils import load_config
from neural_sp.bin.train_utils import set_logger
from neural_sp.bin.train_utils import load_checkpoint
from neural_sp.datasets.loader_asr import Dataset
from neural_sp.models.lm.build import build_lm
from neural_sp.models.seq2seq.speech2text import Speech2Text
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
    recog_params = vars(args)

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
                          dict_path_sub1=os.path.join(dir_name, 'dict_sub1.txt') if os.path.isfile(
                              os.path.join(dir_name, 'dict_sub1.txt')) else False,
                          nlsyms=args.nlsyms,
                          wp_model=os.path.join(dir_name, 'wp.model'),
                          unit=args.unit,
                          unit_sub1=args.unit_sub1,
                          batch_size=args.recog_batch_size,
                          is_test=True)

        if i == 0:
            # Load the ASR model
            model = Speech2Text(args, dir_name)
            model = load_checkpoint(model, args.recog_model[0])[0]
            epoch = int(args.recog_model[0].split('-')[-1])

            # ensemble (different models)
            ensemble_models = [model]
            if len(args.recog_model) > 1:
                for recog_model_e in args.recog_model[1:]:
                    # Load the ASR model
                    conf_e = load_config(os.path.join(os.path.dirname(recog_model_e), 'conf.yml'))
                    args_e = copy.deepcopy(args)
                    for k, v in conf_e.items():
                        if 'recog' not in k:
                            setattr(args_e, k, v)
                    model_e = Speech2Text(args_e)
                    model_e = load_checkpoint(model_e, recog_model_e)[0]
                    model_e.cuda()
                    ensemble_models += [model_e]

            # Load the LM for shallow fusion
            if not args.lm_fusion:
                if args.recog_lm is not None and args.recog_lm_weight > 0:
                    conf_lm = load_config(os.path.join(os.path.dirname(args.recog_lm), 'conf.yml'))
                    args_lm = argparse.Namespace()
                    for k, v in conf_lm.items():
                        setattr(args_lm, k, v)
                    lm = build_lm(args_lm)
                    lm = load_checkpoint(lm, args.recog_lm)[0]
                    if args_lm.backward:
                        model.lm_bwd = lm
                    else:
                        model.lm_fwd = lm

                if args.recog_lm_bwd is not None and args.recog_lm_weight > 0 and \
                        (args.recog_fwd_bwd_attention or args.recog_reverse_lm_rescoring):
                    conf_lm = load_config(os.path.join(args.recog_lm_bwd, 'conf.yml'))
                    args_lm_bwd = argparse.Namespace()
                    for k, v in conf_lm.items():
                        setattr(args_lm_bwd, k, v)
                    lm_bwd = build_lm(args_lm_bwd)
                    lm_bwd = load_checkpoint(lm_bwd, args.recog_lm_bwd)[0]
                    model.lm_bwd = lm_bwd

            if not args.recog_unit:
                args.recog_unit = args.unit

            logger.info('recog unit: %s' % args.recog_unit)
            logger.info('recog metric: %s' % args.recog_metric)
            logger.info('recog oracle: %s' % args.recog_oracle)
            logger.info('epoch: %d' % (epoch - 1))
            logger.info('batch size: %d' % args.recog_batch_size)
            logger.info('beam width: %d' % args.recog_beam_width)
            logger.info('min length ratio: %.3f' % args.recog_min_len_ratio)
            logger.info('max length ratio: %.3f' % args.recog_max_len_ratio)
            logger.info('length penalty: %.3f' % args.recog_length_penalty)
            logger.info('coverage penalty: %.3f' % args.recog_coverage_penalty)
            logger.info('coverage threshold: %.3f' % args.recog_coverage_threshold)
            logger.info('CTC weight: %.3f' % args.recog_ctc_weight)
            logger.info('LM path: %s' % args.recog_lm)
            logger.info('LM path (bwd): %s' % args.recog_lm_bwd)
            logger.info('LM weight: %.3f' % args.recog_lm_weight)
            logger.info('GNMT: %s' % args.recog_gnmt_decoding)
            logger.info('forward-backward attention: %s' % args.recog_fwd_bwd_attention)
            logger.info('reverse LM rescoring: %s' % args.recog_reverse_lm_rescoring)
            logger.info('resolving UNK: %s' % args.recog_resolving_unk)
            logger.info('ensemble: %d' % (len(ensemble_models)))
            logger.info('ASR decoder state carry over: %s' % (args.recog_asr_state_carry_over))
            logger.info('LM state carry over: %s' % (args.recog_lm_state_carry_over))
            logger.info('cache size: %d' % (args.recog_n_caches))
            logger.info('cache type: %s' % (args.recog_cache_type))
            logger.info('cache word frequency threshold: %s' % (args.recog_cache_word_freq))
            logger.info('cache theta (speech): %.3f' % (args.recog_cache_theta_speech))
            logger.info('cache lambda (speech): %.3f' % (args.recog_cache_lambda_speech))
            logger.info('cache theta (lm): %.3f' % (args.recog_cache_theta_lm))
            logger.info('cache lambda (lm): %.3f' % (args.recog_cache_lambda_lm))

            # GPU setting
            model.cuda()
            # TODO(hirofumi): move this

        save_path = mkdir_join(args.recog_dir, 'att_weights')
        if args.recog_n_caches > 0:
            save_path_cache = mkdir_join(args.recog_dir, 'cache')

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)
            if args.recog_n_caches > 0:
                shutil.rmtree(save_path_cache)
                os.mkdir(save_path_cache)

        while True:
            batch, is_new_epoch = dataset.next(recog_params['recog_batch_size'])
            best_hyps_id, aws, (cache_attn_hist, cache_id_hist) = model.decode(
                batch['xs'], recog_params, dataset.idx2token[0],
                exclude_eos=False,
                refs_id=batch['ys'],
                ensemble_models=ensemble_models[1:] if len(ensemble_models) > 1 else [],
                speakers=batch['sessions'] if dataset.corpus == 'swbd' else batch['speakers'])

            if model.bwd_weight > 0.5:
                # Reverse the order
                best_hyps_id = [hyp[::-1] for hyp in best_hyps_id]
                aws = [aw[::-1] for aw in aws]

            for b in range(len(batch['xs'])):
                tokens = dataset.idx2token[0](best_hyps_id[b], return_list=True)
                spk = batch['speakers'][b]

                plot_attention_weights(
                    aws[b][:len(tokens)],
                    tokens,
                    spectrogram=batch['xs'][b][:, :dataset.input_dim] if args.input_type == 'speech' else None,
                    save_path=mkdir_join(save_path, spk, batch['utt_ids'][b] + '.png'),
                    figsize=(20, 8))

                if args.recog_n_caches > 0 and cache_id_hist is not None and cache_attn_hist is not None:
                    n_keys, n_queries = cache_attn_hist[0].shape
                    # mask = np.ones((n_keys, n_queries))
                    # for i in range(n_queries):
                    #     mask[:n_keys - i, -(i + 1)] = 0
                    mask = np.zeros((n_keys, n_queries))

                    plot_cache_weights(
                        cache_attn_hist[0],
                        keys=dataset.idx2token[0](cache_id_hist[-1], return_list=True),  # fifo
                        # keys=dataset.idx2token[0](cache_id_hist, return_list=True),  # dict
                        queries=tokens,
                        save_path=mkdir_join(save_path_cache, spk, batch['utt_ids'][b] + '.png'),
                        figsize=(40, 16),
                        mask=mask)

                if model.bwd_weight > 0.5:
                    hyp = ' '.join(tokens[::-1])
                else:
                    hyp = ' '.join(tokens)
                logger.info('utt-id: %s' % batch['utt_ids'][b])
                logger.info('Ref: %s' % batch['text'][b].lower())
                logger.info('Hyp: %s' % hyp)
                logger.info('-' * 50)

            if is_new_epoch:
                break


if __name__ == '__main__':
    main()
