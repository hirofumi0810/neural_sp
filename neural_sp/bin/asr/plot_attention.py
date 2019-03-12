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
from neural_sp.bin.asr.plot_utils import plot_attention_weights
from neural_sp.bin.asr.plot_utils import plot_cache_weights
from neural_sp.bin.train_utils import load_config
from neural_sp.bin.train_utils import set_logger
from neural_sp.datasets.loader_asr import Dataset
from neural_sp.models.rnnlm.rnnlm import RNNLM
from neural_sp.models.rnnlm.rnnlm_seq import SeqRNNLM
from neural_sp.models.seq2seq.seq2seq import Seq2seq
from neural_sp.utils.general import mkdir_join


def main():

    args = parse()

    # Load a conf file
    conf = load_config(os.path.join(args.recog_model[0], 'conf.yml'))

    # Overwrite conf
    for k, v in conf.items():
        if 'recog' not in k:
            setattr(args, k, v)
    decode_params = vars(args)

    # Setting for logging
    if os.path.isfile(os.path.join(args.recog_dir, 'plot.log')):
        os.remove(os.path.join(args.recog_dir, 'plot.log'))
    logger = set_logger(os.path.join(args.recog_dir, 'plot.log'), key='decoding')

    for i, set in enumerate(args.recog_sets):
        subsample_factor = 1
        subsample_factor_sub1 = 1
        subsample = [int(s) for s in args.subsample.split('_')]
        if args.conv_poolings:
            for p in args.conv_poolings.split('_'):
                p = int(p.split(',')[0].replace('(', ''))
                if p > 1:
                    subsample_factor *= p
        if args.train_set_sub1 is not None:
            subsample_factor_sub1 = subsample_factor * np.prod(subsample[:args.enc_nlayers_sub1 - 1])
        subsample_factor *= np.prod(subsample)

        # Load dataset
        dataset = Dataset(tsv_path=set,
                          dict_path=os.path.join(args.recog_model[0], 'dict.txt'),
                          dict_path_sub1=os.path.join(args.recog_model[0], 'dict_sub1.txt') if os.path.isfile(
                              os.path.join(args.recog_model[0], 'dict_sub1.txt')) else None,
                          wp_model=os.path.join(args.recog_model[0], 'wp.model'),
                          unit=args.unit,
                          unit_sub1=args.unit_sub1,
                          batch_size=args.recog_batch_size,
                          is_test=True)

        if i == 0:
            args.vocab = dataset.vocab
            args.vocab_sub1 = dataset.vocab_sub1
            args.vocab_sub2 = dataset.vocab_sub2
            args.vocab_sub3 = dataset.vocab_sub3
            args.input_dim = dataset.input_dim

            # TODO(hirofumi): For cold fusion
            args.rnnlm_cold_fusion = False
            args.rnnlm_init = False

            # Load the ASR model
            model = Seq2seq(args)
            epoch, _, _, _ = model.load_checkpoint(args.recog_model[0], epoch=args.recog_epoch)
            model.save_path = args.recog_model[0]

            # ensemble (different models)
            ensemble_models = [model]
            if len(args.recog_model) > 1:
                for recog_model_e in args.recog_model[1:]:
                    # Load a conf file
                    config_e = load_config(os.path.join(recog_model_e, 'conf.yml'))

                    # Overwrite conf
                    args_e = copy.deepcopy(args)
                    for k, v in config_e.items():
                        if 'recog' not in k:
                            setattr(args_e, k, v)

                    model_e = Seq2seq(args_e)
                    model_e.load_checkpoint(recog_model_e, epoch=args.recog_epoch)
                    model_e.cuda()
                    ensemble_models += [model_e]
            # checkpoint ensemble
            elif args.recog_checkpoint_ensemble > 1:
                for i_e in range(1, args.recog_checkpoint_ensemble):
                    model_e = Seq2seq(args)
                    model_e.load_checkpoint(args.recog_model[0], epoch=args.recog_epoch - i_e)
                    model_e.cuda()
                    ensemble_models += [model_e]

            # For shallow fusion
            if not args.rnnlm_cold_fusion:
                if args.recog_rnnlm is not None and args.recog_rnnlm_weight > 0:
                    # Load a RNNLM conf file
                    config_rnnlm = load_config(os.path.join(args.recog_rnnlm, 'conf.yml'))

                    # Merge conf with args
                    args_rnnlm = argparse.Namespace()
                    for k, v in config_rnnlm.items():
                        setattr(args_rnnlm, k, v)

                    assert args.unit == args_rnnlm.unit
                    args_rnnlm.vocab = dataset.vocab

                    # Load the pre-trianed RNNLM
                    seq_rnnlm = SeqRNNLM(args_rnnlm)
                    seq_rnnlm.load_checkpoint(args.recog_rnnlm, epoch=-1)

                    # Copy parameters
                    rnnlm = RNNLM(args_rnnlm)
                    rnnlm.copy_from_seqrnnlm(seq_rnnlm)

                    # Register to the ASR model
                    if args_rnnlm.backward:
                        model.rnnlm_bwd = rnnlm
                    else:
                        # model.rnnlm_fwd = rnnlm
                        model.rnnlm_fwd = seq_rnnlm

                if args.recog_rnnlm_bwd is not None and args.recog_rnnlm_weight > 0 and (args.recog_fwd_bwd_attention or args.recog_reverse_lm_rescoring):
                    # Load a RNNLM conf file
                    config_rnnlm = load_config(os.path.join(args.recog_rnnlm_bwd, 'conf.yml'))

                    # Merge conf with args
                    args_rnnlm_bwd = argparse.Namespace()
                    for k, v in config_rnnlm.items():
                        setattr(args_rnnlm_bwd, k, v)

                    assert args.unit == args_rnnlm_bwd.unit
                    args_rnnlm_bwd.vocab = dataset.vocab

                    # Load the pre-trianed RNNLM
                    seq_rnnlm_bwd = SeqRNNLM(args_rnnlm_bwd)
                    seq_rnnlm_bwd.load_checkpoint(args.recog_rnnlm_bwd, epoch=-1)

                    # Copy parameters
                    rnnlm_bwd = RNNLM(args_rnnlm_bwd)
                    rnnlm_bwd.copy_from_seqrnnlm(seq_rnnlm_bwd)

                    # Resister to the ASR model
                    model.rnnlm_bwd = rnnlm_bwd

            logger.info('epoch: %d' % (epoch - 1))
            logger.info('batch size: %d' % args.recog_batch_size)
            logger.info('beam width: %d' % args.recog_beam_width)
            logger.info('min length ratio: %.3f' % args.recog_min_len_ratio)
            logger.info('max length ratio: %.3f' % args.recog_max_len_ratio)
            logger.info('length penalty: %.3f' % args.recog_length_penalty)
            logger.info('coverage penalty: %.3f' % args.recog_coverage_penalty)
            logger.info('coverage threshold: %.3f' % args.recog_coverage_threshold)
            logger.info('CTC weight: %.3f' % args.recog_ctc_weight)
            logger.info('RNNLM path: %s' % args.recog_rnnlm)
            logger.info('RNNLM path (bwd): %s' % args.recog_rnnlm_bwd)
            logger.info('RNNLM weight: %.3f' % args.recog_rnnlm_weight)
            logger.info('GNMT: %s' % args.recog_gnmt_decoding)
            logger.info('forward-backward attention: %s' % args.recog_fwd_bwd_attention)
            logger.info('reverse LM rescoring: %s' % args.recog_reverse_lm_rescoring)
            logger.info('resolving UNK: %s' % args.recog_resolving_unk)
            logger.info('recog unit: %s' % args.recog_unit)
            logger.info('ensemble: %d' % (len(ensemble_models)))
            logger.info('checkpoint ensemble: %d' % (args.recog_checkpoint_ensemble))
            logger.info('cache size: %d' % (args.recog_ncaches))

            # GPU setting
            model.cuda()

        save_path = mkdir_join(args.recog_dir, 'att_weights')
        if args.recog_ncaches > 0:
            save_path_cache = mkdir_join(args.recog_dir, 'cache')

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

        while True:
            batch, is_new_epoch = dataset.next(decode_params['recog_batch_size'])
            best_hyps, aws, perm_id, (cache_probs_history, cache_keys_history) = model.decode(
                batch['xs'], decode_params,
                exclude_eos=False,
                id2token=id2token,
                refs=batch['ys'],
                ensemble_models=ensemble_models[1:] if len(ensemble_models) > 1 else [],
                speakers=batch['speakers'])
            ys = [batch['ys'][i] for i in perm_id]

            if model.bwd_weight > 0.5:
                # Reverse the order
                best_hyps = [hyp[::-1] for hyp in best_hyps]
                aws = [aw[::-1] for aw in aws]

            for b in range(len(batch['xs'])):
                token_list = id2token(best_hyps[b], return_list=True)
                token_list = [unicode(t, 'utf-8') for t in token_list]
                speaker = '_'.join(batch['utt_ids'][b].replace('-', '_').split('_')[:-2])

                plot_attention_weights(
                    aws[b][:len(token_list)], token_list,
                    spectrogram=batch['xs'][b][:,
                                               :dataset.input_dim] if args.input_type == 'speech' else None,
                    save_path=mkdir_join(save_path, speaker, batch['utt_ids'][b] + '.png'),
                    figsize=(20, 8))

                if args.recog_ncaches > 0 and cache_keys_history is not None:
                    plot_cache_weights(
                        cache_probs_history[0],
                        [unicode(t, 'utf-8') for t in id2token(cache_keys_history[-1], return_list=True)],
                        token_list,
                        save_path=mkdir_join(save_path_cache, speaker, batch['utt_ids'][b] + '.png'),
                        figsize=(40, 16))

                ref = ys[b]
                if model.bwd_weight > 0.5:
                    hyp = ' '.join(token_list[::-1])
                else:
                    hyp = ' '.join(token_list)
                logger.info('utt-id: %s' % batch['utt_ids'][b])
                logger.info('Ref: %s' % ref.lower())
                logger.info('Hyp: %s' % hyp)
                logger.info('-' * 50)

            if is_new_epoch:
                break


if __name__ == '__main__':
    main()
