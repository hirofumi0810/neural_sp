#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Plot attention weights of the attention model."""

import argparse
import copy
import logging
import os
import shutil
import sys

from neural_sp.bin.args_asr import parse_args_eval
from neural_sp.bin.eval_utils import average_checkpoints
from neural_sp.bin.plot_utils import plot_attention_weights
from neural_sp.bin.train_utils import (
    compute_subsampling_factor,
    load_checkpoint,
    load_config,
    set_logger
)
from neural_sp.datasets.asr import build_dataloader
from neural_sp.models.lm.build import build_lm
from neural_sp.models.seq2seq.speech2text import Speech2Text
from neural_sp.utils import mkdir_join

logger = logging.getLogger(__name__)


def main():

    # Load configuration
    args, recog_params, dir_name = parse_args_eval(sys.argv[1:])
    args = compute_subsampling_factor(args)

    # Setting for logging
    if os.path.isfile(os.path.join(args.recog_dir, 'plot.log')):
        os.remove(os.path.join(args.recog_dir, 'plot.log'))
    set_logger(os.path.join(args.recog_dir, 'plot.log'), stdout=args.recog_stdout)

    for i, s in enumerate(args.recog_sets):
        # Load dataloader
        dataloader = build_dataloader(args=args,
                                      tsv_path=s,
                                      batch_size=1,
                                      is_test=True)

        if i == 0:
            # Load the ASR model
            model = Speech2Text(args, dir_name)
            epoch = int(float(args.recog_model[0].split('-')[-1]) * 10) / 10
            if args.recog_n_average > 1:
                # Model averaging for Transformer
                model = average_checkpoints(model, args.recog_model[0],
                                            n_average=args.recog_n_average)
            else:
                load_checkpoint(args.recog_model[0], model)

            # Ensemble (different models)
            ensemble_models = [model]
            if len(args.recog_model) > 1:
                for recog_model_e in args.recog_model[1:]:
                    conf_e = load_config(os.path.join(os.path.dirname(recog_model_e), 'conf.yml'))
                    args_e = copy.deepcopy(args)
                    for k, v in conf_e.items():
                        if 'recog' not in k:
                            setattr(args_e, k, v)
                    model_e = Speech2Text(args_e)
                    load_checkpoint(recog_model_e, model_e)
                    if args.recog_n_gpus >= 1:
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
                    load_checkpoint(args.recog_lm, lm)
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
            if args.recog_n_gpus >= 1:
                model.cudnn_setting(deterministic=True, benchmark=False)
                model.cuda()

        save_path = mkdir_join(args.recog_dir, 'att_weights')

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        while True:
            batch, is_new_epoch = dataloader.next(recog_params['recog_batch_size'])
            nbest_hyps_id, aws = model.decode(
                batch['xs'], recog_params, dataloader.idx2token[0],
                exclude_eos=False,
                refs_id=batch['ys'],
                ensemble_models=ensemble_models[1:] if len(ensemble_models) > 1 else [],
                speakers=batch['sessions'] if dataloader.corpus == 'swbd' else batch['speakers'])
            best_hyps_id = [h[0] for h in nbest_hyps_id]

            # Get CTC probs
            ctc_probs, topk_ids = None, None
            if args.ctc_weight > 0:
                ctc_probs, topk_ids, xlens = model.get_ctc_probs(
                    batch['xs'], task='ys', temperature=1, topk=min(100, model.vocab))
                # NOTE: ctc_probs: '[B, T, topk]'
            ctc_probs_sub1, topk_ids_sub1 = None, None
            if args.ctc_weight_sub1 > 0:
                ctc_probs_sub1, topk_ids_sub1, xlens_sub1 = model.get_ctc_probs(
                    batch['xs'], task='ys_sub1', temperature=1, topk=min(100, model.vocab_sub1))

            if model.bwd_weight > 0.5:
                # Reverse the order
                best_hyps_id = [hyp[::-1] for hyp in best_hyps_id]
                aws = [[aw[0][:, ::-1]] for aw in aws]

            for b in range(len(batch['xs'])):
                tokens = dataloader.idx2token[0](best_hyps_id[b], return_list=True)
                spk = batch['speakers'][b]

                plot_attention_weights(
                    aws[b][0][:, :len(tokens)], tokens,
                    spectrogram=batch['xs'][b][:, :dataloader.input_dim] if args.input_type == 'speech' else None,
                    factor=args.subsample_factor,
                    ref=batch['text'][b].lower(),
                    save_path=mkdir_join(save_path, spk, batch['utt_ids'][b] + '.png'),
                    figsize=(20, 8),
                    ctc_probs=ctc_probs[b, :xlens[b]] if ctc_probs is not None else None,
                    ctc_topk_ids=topk_ids[b] if topk_ids is not None else None,
                    ctc_probs_sub1=ctc_probs_sub1[b, :xlens_sub1[b]] if ctc_probs_sub1 is not None else None,
                    ctc_topk_ids_sub1=topk_ids_sub1[b] if topk_ids_sub1 is not None else None)

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
