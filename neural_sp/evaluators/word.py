#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate the word-level model by WER."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from tqdm import tqdm

from neural_sp.evaluators.edit_distance import compute_wer
from neural_sp.evaluators.resolving_unk import resolve_unk
from neural_sp.utils.general import mkdir_join

logger = logging.getLogger("decoding").getChild('word')


def eval_word(models, dataset, decode_params, epoch,
              decode_dir=None, progressbar=False):
    """Evaluate the word-level model by WER.

    Args:
        models (list): the models to evaluate
        dataset: An instance of a `Dataset' class
        decode_params (dict):
        epoch (int):
        decode_dir (str):
        progressbar (bool): if True, visualize the progressbar
    Returns:
        wer (float): Word error rate
        n_sub (int): the number of substitution errors
        n_ins (int): the number of insertion errors
        n_del (int): the number of deletion errors
        n_oov_total (int):

    """
    # Reset data counter
    dataset.reset()

    if decode_dir is None:
        decode_dir = 'decode_' + dataset.set + '_ep' + str(epoch) + '_beam' + str(decode_params['recog_beam_width'])
        decode_dir += '_lp' + str(decode_params['recog_length_penalty'])
        decode_dir += '_cp' + str(decode_params['recog_coverage_penalty'])
        decode_dir += '_' + str(decode_params['recog_min_len_ratio']) + '_' + str(decode_params['recog_max_len_ratio'])
        decode_dir += '_rnnlm' + str(decode_params['recog_rnnlm_weight'])

        ref_trn_save_path = mkdir_join(models[0].save_path, decode_dir, 'ref.trn')
        hyp_trn_save_path = mkdir_join(models[0].save_path, decode_dir, 'hyp.trn')
    else:
        ref_trn_save_path = mkdir_join(decode_dir, 'ref.trn')
        hyp_trn_save_path = mkdir_join(decode_dir, 'hyp.trn')

    wer = 0
    n_sub, n_ins, n_del = 0, 0, 0
    n_word = 0
    n_oov_total = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))  # TODO(hirofumi): fix this

    with open(hyp_trn_save_path, 'w') as f_hyp, open(ref_trn_save_path, 'w') as f_ref:
        while True:
            batch, is_new_ep = dataset.next(decode_params['recog_batch_size'])
            best_hyps, aws, perm_ids, _ = models[0].decode(
                batch['xs'], decode_params,
                exclude_eos=True,
                ensemble_models=models[1:] if len(models) > 1 else [],
                speakers=batch['speakers'])
            ys = [batch['text'][i] for i in perm_ids]

            for b in range(len(batch['xs'])):
                ref = ys[b]
                hyp = dataset.id2word(best_hyps[b])

                n_oov_total += hyp.count('<unk>')

                # Resolving UNK
                if decode_params['recog_resolving_unk'] and '<unk>' in hyp:
                    best_hyps_sub, aw_sub, _ = models[0].decode(
                        batch['xs'][b:b + 1], decode_params, exclude_eos=True)
                    # task_index=1

                    hyp = resolve_unk(
                        hyp, best_hyps_sub[0], aws[b], aw_sub[0], dataset.id2char,
                        diff_time_resolution=2 ** sum(models[0].subsample) // 2 ** sum(models[0].subsample[:models[0].enc_nlayers_sub - 1]))
                    hyp = hyp.replace('*', '')

                # Write to trn
                speaker = '_'.join(batch['utt_ids'][b].replace('-', '_').split('_')[:-2])
                start = batch['utt_ids'][b].replace('-', '_').split('_')[-2]
                end = batch['utt_ids'][b].replace('-', '_').split('_')[-1]
                f_ref.write(ref + ' (' + speaker + '-' + start + '-' + end + ')\n')
                f_hyp.write(hyp + ' (' + speaker + '-' + start + '-' + end + ')\n')
                logger.info('utt-id: %s' % batch['utt_ids'][b])
                # logger.info('Ref: %s' % ref.lower())
                logger.info('Ref: %s' % ref)
                logger.info('Hyp: %s' % hyp)
                logger.info('-' * 150)

                # Compute WER
                wer_b, sub_b, ins_b, del_b = compute_wer(ref=ref.split(' '),
                                                         hyp=hyp.split(' '),
                                                         normalize=False)
                wer += wer_b
                n_sub += sub_b
                n_ins += ins_b
                n_del += del_b
                n_word += len(ref.split(' '))
                # logger.info('WER: %d%%' % (wer_b / len(ref.split(' '))))

                if progressbar:
                    pbar.update(1)

            if is_new_ep:
                break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    wer /= n_word
    n_sub /= n_word
    n_ins /= n_word
    n_del /= n_word

    return wer, n_sub, n_ins, n_del, n_oov_total
