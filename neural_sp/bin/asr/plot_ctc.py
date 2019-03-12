#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Plot the CTC posteriors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import shutil

from neural_sp.bin.args_asr import parse
from neural_sp.bin.asr.plot_utils import plot_ctc_probs
from neural_sp.bin.train_utils import load_config
from neural_sp.bin.train_utils import set_logger
from neural_sp.datasets.loader_asr import Dataset
from neural_sp.models.seq2seq.seq2seq import Seq2seq
from neural_sp.utils.general import mkdir_join


def main():

    args = parse()

    # Load a conf file
    conf = load_config(os.path.join(args.recog_model, 'conf.yml'))

    # Overwrite conf
    for k, v in conf.items():
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
        dataset = Dataset(tsv_path=set,
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

            # GPU setting
            model.cuda()

            logger.info('epoch: %d' % (epoch - 1))
            logger.info('batch size: %d' % args.recog_batch_size)

        save_path = mkdir_join(args.plot_dir, 'att_weights')

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
            best_hyps, aws, perm_id, _ = model.decode(batch['xs'], decode_params,
                                                      exclude_eos=False)
            ys = [batch['ys'][i] for i in perm_id]

            # Get CTC probs
            ctc_probs, indices_topk, xlens = model.get_ctc_posteriors(
                batch['xs'], temperature=1, topk=min(100, model.vocab))
            # NOTE: ctc_probs: '[B, T, topk]'

            for b in range(len(batch['xs'])):
                token_list = id2token(best_hyps[b], return_list=True)
                token_list = [unicode(t, 'utf-8') for t in token_list]
                speaker = '_'.join(batch['utt_ids'][b].replace('-', '_').split('_')[:-2])

                plot_ctc_probs(
                    ctc_probs[b, :xlens[b]],
                    indices_topk[b],
                    nframes=xlens[b],
                    subsample_factor=subsample_factor,
                    spectrogram=batch['xs'][b][:, :dataset.input_dim],
                    save_path=mkdir_join(save_path, speaker, batch['utt_ids'][b] + '.png'),
                    figsize=(20, 8))

                ref = ys[b]
                hyp = ' '.join(token_list)
                logger.info('utt-id: %s' % batch['utt_ids'][b])
                logger.info('Ref: %s' % ref.lower())
                logger.info('Hyp: %s' % hyp)
                logger.info('-' * 50)

            if is_new_epoch:
                break


if __name__ == '__main__':
    main()
