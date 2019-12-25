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
import logging
import os
import shutil

from neural_sp.bin.args_asr import parse
from neural_sp.bin.plot_utils import plot_attention_weights
from neural_sp.bin.train_utils import load_checkpoint
from neural_sp.bin.train_utils import load_config
from neural_sp.bin.train_utils import set_logger
from neural_sp.datasets.asr import Dataset
from neural_sp.models.lm.build import build_lm
from neural_sp.models.seq2seq.speech2text import Speech2Text
from neural_sp.utils import mkdir_join

logger = logging.getLogger(__name__)


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
    set_logger(os.path.join(args.recog_dir, 'plot.log'), stdout=args.recog_stdout)

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
            load_checkpoint(model, args.recog_model[0])
            epoch = int(args.recog_model[0].split('-')[-1])

            # ensemble (different models)
            ensemble_models = [model]
            if len(args.recog_model) > 1:
                for recog_model_e in args.recog_model[1:]:
                    conf_e = load_config(os.path.join(os.path.dirname(recog_model_e), 'conf.yml'))
                    args_e = copy.deepcopy(args)
                    for k, v in conf_e.items():
                        if 'recog' not in k:
                            setattr(args_e, k, v)
                    model_e = Speech2Text(args_e)
                    load_checkpoint(model_e, recog_model_e)
                    model_e.cuda()
                    ensemble_models += [model_e]

            # Load the LM for shallow fusion
            if not args.lm_fusion:
                # first path
                if args.recog_lm is not None and args.recog_lm_weight > 0:
                    conf_lm = load_config(os.path.join(os.path.dirname(args.recog_lm), 'conf.yml'))
                    args_lm = argparse.Namespace()
                    for k, v in conf_lm.items():
                        setattr(args_lm, k, v)
                    lm = build_lm(args_lm)
                    load_checkpoint(lm, args.recog_lm)
                    if args_lm.backward:
                        model.lm_bwd = lm
                    else:
                        model.lm_fwd = lm
                # NOTE: only support for first path

            if not args.recog_unit:
                args.recog_unit = args.unit

            logger.info('recog unit: %s' % args.recog_unit)
            logger.info('recog oracle: %s' % args.recog_oracle)
            logger.info('epoch: %d' % epoch)
            logger.info('batch size: %d' % args.recog_batch_size)
            logger.info('beam width: %d' % args.recog_beam_width)
            logger.info('min length ratio: %.3f' % args.recog_min_len_ratio)
            logger.info('max length ratio: %.3f' % args.recog_max_len_ratio)
            logger.info('length penalty: %.3f' % args.recog_length_penalty)
            logger.info('length norm: %s' % args.recog_length_norm)
            logger.info('coverage penalty: %.3f' % args.recog_coverage_penalty)
            logger.info('coverage threshold: %.3f' % args.recog_coverage_threshold)
            logger.info('CTC weight: %.3f' % args.recog_ctc_weight)
            logger.info('fist LM path: %s' % args.recog_lm)
            logger.info('LM weight: %.3f' % args.recog_lm_weight)
            logger.info('GNMT: %s' % args.recog_gnmt_decoding)
            logger.info('forward-backward attention: %s' % args.recog_fwd_bwd_attention)
            logger.info('resolving UNK: %s' % args.recog_resolving_unk)
            logger.info('ensemble: %d' % (len(ensemble_models)))
            logger.info('ASR decoder state carry over: %s' % (args.recog_asr_state_carry_over))
            logger.info('LM state carry over: %s' % (args.recog_lm_state_carry_over))
            logger.info('model average (Transformer): %d' % (args.recog_n_average))

            # GPU setting
            model.cuda()

        save_path = mkdir_join(args.recog_dir, 'att_weights')

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        while True:
            batch, is_new_epoch = dataset.next(recog_params['recog_batch_size'])
            best_hyps_id, aws = model.decode(
                batch['xs'], recog_params, dataset.idx2token[0],
                exclude_eos=False,
                refs_id=batch['ys'],
                ensemble_models=ensemble_models[1:] if len(ensemble_models) > 1 else [],
                speakers=batch['sessions'] if dataset.corpus == 'swbd' else batch['speakers'])

            # Get CTC probs
            ctc_probs, indices_topk = None, None
            if args.ctc_weight > 0:
                ctc_probs, indices_topk, xlens = model.get_ctc_probs(
                    batch['xs'], temperature=1, topk=min(100, model.vocab))
                # NOTE: ctc_probs: '[B, T, topk]'

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
                    figsize=(20, 8),
                    ctc_probs=ctc_probs[b, :xlens[b]] if ctc_probs is not None else None,
                    ctc_indices_topk=indices_topk[b] if indices_topk is not None else None)

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
