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
from neural_sp.bin.plot_utils import plot_ctc_probs
from neural_sp.bin.train_utils import load_config
from neural_sp.bin.train_utils import set_logger
from neural_sp.bin.train_utils import load_checkpoint
from neural_sp.datasets.loader_asr import Dataset
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
    logger = set_logger(os.path.join(args.recog_dir, 'plot.log'), key='decoding')

    for i, s in enumerate(args.recog_sets):
        subsample_factor = 1
        subsample = [int(s) for s in args.subsample.split('_')]
        if args.conv_poolings:
            for p in args.conv_poolings.split('_'):
                p = int(p.split(',')[0].replace('(', ''))
                if p > 1:
                    subsample_factor *= p
        subsample_factor *= np.prod(subsample)

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

            if not args.recog_unit:
                args.recog_unit = args.unit

            logger.info('recog unit: %s' % args.recog_unit)
            logger.info('epoch: %d' % (epoch - 1))
            logger.info('batch size: %d' % args.recog_batch_size)

            # GPU setting
            model.cuda()
            # TODO(hirofumi): move this

        save_path = mkdir_join(args.plot_dir, 'ctc_probs')

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        while True:
            batch, is_new_epoch = dataset.next(recog_params['recog_batch_size'])
            best_hyps_id, _, _ = model.decode(batch['xs'], recog_params,
                                              exclude_eos=False)

            # Get CTC probs
            ctc_probs, indices_topk, xlens = model.get_ctc_probs(
                batch['xs'], temperature=1, topk=min(100, model.vocab))
            # NOTE: ctc_probs: '[B, T, topk]'

            for b in range(len(batch['xs'])):
                tokens = dataset.idx2token[0](best_hyps_id[b], return_list=True)
                spk = '_'.join(batch['utt_ids'][b].replace('-', '_').split('_')[:-2])

                plot_ctc_probs(
                    ctc_probs[b, :xlens[b]],
                    indices_topk[b],
                    nframes=xlens[b],
                    subsample_factor=subsample_factor,
                    spectrogram=batch['xs'][b][:, :dataset.input_dim],
                    save_path=mkdir_join(save_path, spk, batch['utt_ids'][b] + '.png'),
                    figsize=(20, 8))

                ref = batch['text'][b]
                hyp = ' '.join(tokens)
                logger.info('utt-id: %s' % batch['utt_ids'][b])
                logger.info('Ref: %s' % ref.lower())
                logger.info('Hyp: %s' % hyp)
                logger.info('-' * 50)

            if is_new_epoch:
                break


if __name__ == '__main__':
    main()
