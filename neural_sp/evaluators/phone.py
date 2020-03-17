#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate a phene-level model by PER."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from tqdm import tqdm

from neural_sp.evaluators.edit_distance import compute_wer
from neural_sp.utils import mkdir_join

logger = logging.getLogger(__name__)


def eval_phone(models, dataset, recog_params, epoch,
               recog_dir=None, streaming=False, progressbar=False):
    """Evaluate a phone-level model by PER.

    Args:
        models (list): models to evaluate
        dataset (Dataset): evaluation dataset
        recog_params (dict):
        epoch (int):
        recog_dir (str):
        streaming (bool): streaming decoding for the session-level evaluation
        progressbar (bool): visualize the progressbar
    Returns:
        per (float): Phone error rate

    """
    # Reset data counter
    dataset.reset()

    if recog_dir is None:
        recog_dir = 'decode_' + dataset.set + '_ep' + str(epoch) + '_beam' + str(recog_params['recog_beam_width'])
        recog_dir += '_lp' + str(recog_params['recog_length_penalty'])
        recog_dir += '_cp' + str(recog_params['recog_coverage_penalty'])
        recog_dir += '_' + str(recog_params['recog_min_len_ratio']) + '_' + str(recog_params['recog_max_len_ratio'])

        ref_trn_save_path = mkdir_join(models[0].save_path, recog_dir, 'ref.trn')
        hyp_trn_save_path = mkdir_join(models[0].save_path, recog_dir, 'hyp.trn')
    else:
        ref_trn_save_path = mkdir_join(recog_dir, 'ref.trn')
        hyp_trn_save_path = mkdir_join(recog_dir, 'hyp.trn')

    per = 0
    n_sub, n_ins, n_del = 0, 0, 0
    n_phone = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))

    with open(hyp_trn_save_path, 'w') as f_hyp, open(ref_trn_save_path, 'w') as f_ref:
        while True:
            batch, is_new_epoch = dataset.next(recog_params['recog_batch_size'])
            if streaming or recog_params['recog_chunk_sync']:
                best_hyps_id, _ = models[0].decode_streaming(
                    batch['xs'], recog_params, dataset.idx2token[0],
                    exclude_eos=True)
            else:
                best_hyps_id, _ = models[0].decode(
                    batch['xs'], recog_params, dataset.idx2token[0],
                    exclude_eos=True,
                    refs_id=batch['ys'],
                    utt_ids=batch['utt_ids'],
                    speakers=batch['sessions' if dataset.corpus == 'swbd' else 'speakers'],
                    ensemble_models=models[1:] if len(models) > 1 else [])

            for b in range(len(batch['xs'])):
                ref = batch['text'][b]
                hyp = dataset.idx2token[0](best_hyps_id[b])

                # Write to trn
                speaker = str(batch['speakers'][b]).replace('-', '_')
                if streaming:
                    utt_id = str(batch['utt_ids'][b]) + '_0000000_0000001'
                else:
                    utt_id = str(batch['utt_ids'][b])
                f_ref.write(ref + ' (' + speaker + '-' + utt_id + ')\n')
                f_hyp.write(hyp + ' (' + speaker + '-' + utt_id + ')\n')
                logger.debug('utt-id: %s' % utt_id)
                logger.debug('Ref: %s' % ref)
                logger.debug('Hyp: %s' % hyp)
                logger.debug('-' * 150)

                if not streaming:
                    # Compute PER
                    per_b, sub_b, ins_b, del_b = compute_wer(ref=ref.split(' '),
                                                             hyp=hyp.split(' '),
                                                             normalize=False)
                    per += per_b
                    n_sub += sub_b
                    n_ins += ins_b
                    n_del += del_b
                    n_phone += len(ref.split(' '))

                if progressbar:
                    pbar.update(1)

            if is_new_epoch:
                break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    if not streaming:
        per /= n_phone
        n_sub /= n_phone
        n_ins /= n_phone
        n_del /= n_phone

    logger.debug('PER (%s): %.2f %%' % (dataset.set, per))
    logger.debug('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub, n_ins, n_del))

    return per
