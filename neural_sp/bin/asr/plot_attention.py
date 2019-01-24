#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Plot attention weights of the attention model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil

from neural_sp.bin.args_asr import parse
from neural_sp.bin.asr.plot_utils import plot_attention_weights
from neural_sp.bin.train_utils import load_config
from neural_sp.bin.train_utils import set_logger
from neural_sp.datasets.loader_asr import Dataset
from neural_sp.models.rnnlm.rnnlm import RNNLM
from neural_sp.models.rnnlm.rnnlm_seq import SeqRNNLM
from neural_sp.models.seq2seq.seq2seq import Seq2seq
from neural_sp.utils.general import mkdir_join


def main():

    args = parse()

    # Load a config file
    config = load_config(os.path.join(args.recog_model, 'config.yml'))

    # Overwrite config
    for k, v in config.items():
        setattr(args, k, v)
    decode_params = vars(args)

    # Setting for logging
    if os.path.isfile(os.path.join(args.plot_dir, 'plot.log')):
        os.remove(os.path.join(args.plot_dir, 'plot.log'))
    logger = set_logger(os.path.join(args.plot_dir, 'plot.log'), key='decoding')

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
        dataset = Dataset(csv_path=set,
                          dict_path=os.path.join(args.recog_model, 'dict.txt'),
                          dict_path_sub1=os.path.join(args.recog_model, 'dict_sub1.txt') if os.path.isfile(
                              os.path.join(args.recog_model, 'dict_sub1.txt')) else None,
                          wp_model=os.path.join(args.recog_model, 'wp.model'),
                          unit=args.unit,
                          unit_sub1=args.unit_sub1,
                          batch_size=args.recog_batch_size,
                          is_test=True)

        if i == 0:
            args.vocab = dataset.vocab
            args.vocab_sub1 = dataset.vocab_sub1
            args.input_dim = dataset.input_dim

            # TODO(hirofumi): For cold fusion
            args.rnnlm_cold_fusion = None
            args.rnnlm_init = None

            # Load the ASR model
            model = Seq2seq(args)
            epoch, _, _, _ = model.load_checkpoint(args.recog_model, epoch=args.recog_epoch)

            model.save_path = args.recog_model

            # For shallow fusion
            if (not args.rnnlm_cold_fusion) and args.rnnlm is not None and args.rnnlm_weight > 0:
                # Load a RNNLM config file
                config_rnnlm = load_config(os.path.join(args.rnnlm, 'config.yml'))

                # Merge config with args
                args_rnnlm = argparse.Namespace()
                for k, v in config_rnnlm.items():
                    setattr(args_rnnlm, k, v)

                assert args.unit == args_rnnlm.unit
                args_rnnlm.vocab = dataset.vocab

                # Load the pre-trianed RNNLM
                seq_rnnlm = SeqRNNLM(args_rnnlm)
                seq_rnnlm.load_checkpoint(args.rnnlm, epoch=-1)

                # Copy parameters
                rnnlm = RNNLM(args_rnnlm)
                rnnlm.copy_from_seqrnnlm(seq_rnnlm)

                if args_rnnlm.backward:
                    model.rnnlm_bwd = rnnlm
                else:
                    model.rnnlm_fwd = rnnlm

                logger.info('RNNLM path: %s' % args.rnnlm)
                logger.info('RNNLM weight: %.3f' % args.rnnlm_weight)
                logger.info('RNNLM backward: %s' % str(config_rnnlm['backward']))

            # GPU setting
            model.cuda()

            logger.info('batch size: %d' % args.recog_batch_size)
            logger.info('beam width: %d' % args.beam_width)
            logger.info('length penalty: %.3f' % args.length_penalty)
            logger.info('coverage penalty: %.3f' % args.coverage_penalty)
            logger.info('coverage threshold: %.3f' % args.coverage_threshold)
            logger.info('epoch: %d' % (epoch - 1))

        save_path = mkdir_join(args.plot_dir, 'att_weights')

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        while True:
            batch, is_new_epoch = dataset.next(decode_params['recog_batch_size'])
            best_hyps, aws, perm_idx = model.decode(batch['xs'], decode_params,
                                                    exclude_eos=False)
            ys = [batch['ys'][i] for i in perm_idx]

            if model.bwd_weight > 0.5:
                # Reverse the order
                best_hyps = [hyp[::-1] for hyp in best_hyps]
                aws = [aw[::-1] for aw in aws]

            for b in range(len(batch['xs'])):
                if args.unit == 'word':
                    token_list = dataset.idx2word(best_hyps[b], return_list=True)
                elif args.unit == 'wp':
                    token_list = dataset.idx2wp(best_hyps[b], return_list=True)
                elif args.unit == 'char':
                    token_list = dataset.idx2char(best_hyps[b], return_list=True)
                elif args.unit == 'phone':
                    token_list = dataset.idx2phone(best_hyps[b], return_list=True)
                else:
                    raise NotImplementedError(args.unit)
                token_list = [unicode(t, 'utf-8') for t in token_list]
                speaker = '_'.join(batch['utt_ids'][b].replace('-', '_').split('_')[:-2])

                plot_attention_weights(aws[b][:len(token_list)],
                                       label_list=token_list,
                                       spectrogram=batch['xs'][b][:,
                                                                  :dataset.input_dim] if args.input_type == 'speech' else None,
                                       save_path=mkdir_join(save_path, speaker, batch['utt_ids'][b] + '.png'),
                                       figsize=(20, 8))

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
