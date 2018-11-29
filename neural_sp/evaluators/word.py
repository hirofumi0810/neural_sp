#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate the word-level model by WER."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import six
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
        nsub (int): the number of substitution errors
        nins (int): the number of insertion errors
        ndel (int): the number of deletion errors

    """
    # Reset data counter
    dataset.reset()

    model = models[0]
    # TODO(hirofumi): ensemble decoding

    if decode_dir is None:
        decode_dir = 'decode_' + dataset.set + '_ep' + str(epoch) + '_beam' + str(decode_params['beam_width'])
        decode_dir += '_lp' + str(decode_params['length_penalty'])
        decode_dir += '_cp' + str(decode_params['coverage_penalty'])
        decode_dir += '_' + str(decode_params['min_len_ratio']) + '_' + str(decode_params['max_len_ratio'])
        decode_dir += '_rnnlm' + str(decode_params['rnnlm_weight'])

        ref_trn_save_path = mkdir_join(model.save_path, decode_dir, 'ref.trn')
        hyp_trn_save_path = mkdir_join(model.save_path, decode_dir, 'hyp.trn')
    else:
        ref_trn_save_path = mkdir_join(decode_dir, 'ref.trn')
        hyp_trn_save_path = mkdir_join(decode_dir, 'hyp.trn')

    wer = 0
    nsub, nins, ndel = 0, 0, 0
    nword = 0
    noov_total = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))  # TODO(hirofumi): fix this

    with open(hyp_trn_save_path, 'w') as f_hyp, open(ref_trn_save_path, 'w') as f_ref:
        while True:
            batch, is_new_epoch = dataset.next(decode_params['batch_size'])
            best_hyps, aws, perm_idx = model.decode(batch['xs'], decode_params,
                                                    exclude_eos=True)
            ys = [batch['text'][i] for i in perm_idx]

            for b in six.moves.range(len(batch['xs'])):
                ref = ys[b]
                hyp = dataset.idx2word(best_hyps[b])

                noov_total += hyp.count('<unk>')

                # Resolving UNK
                if decode_params['resolving_unk'] and '<unk>' in hyp:
                    best_hyps_sub, aw_sub, _ = model.decode(
                        batch['xs'][b:b + 1], decode_params, exclude_eos=True)
                    # task_index=1

                    hyp = resolve_unk(
                        hyp, best_hyps_sub[0], aws[b], aw_sub[0], dataset.idx2char,
                        diff_time_resolution=2 ** sum(model.subsample) // 2 ** sum(model.subsample[:model.enc_nlayers_sub - 1]))
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
                logger.info('-' * 50)

                # Compute WER
                wer_b, sub_b, ins_b, del_b = compute_wer(ref=ref.split(' '),
                                                         hyp=hyp.split(' '),
                                                         normalize=False)
                wer += wer_b
                nsub += sub_b
                nins += ins_b
                ndel += del_b
                nword += len(ref.split(' '))
                # logger.info('WER: %d%%' % (wer_b / len(ref.split(' '))))

                if progressbar:
                    pbar.update(1)

            if is_new_epoch:
                break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    wer /= nword
    nsub /= nword
    nins /= nword
    ndel /= nword

    return wer, nsub, nins, ndel
