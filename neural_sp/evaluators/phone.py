#! /usr/bin/env python
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
from neural_sp.utils.general import mkdir_join

logger = logging.getLogger("decoding").getChild('phone')


def eval_phone(models, dataset, decode_params, epoch,
               decode_dir=None, progressbar=False):
    """Evaluate a phone-level model by PER.

    Args:
        models (list): the models to evaluate
        dataset: An instance of a `Dataset' class
        decode_params (dict):
        epoch (int):
        decode_dir (str):
        progressbar (bool): if True, visualize the progressbar
    Returns:
        per (float): Phone error rate
        n_sub (int): the number of substitution errors
        n_ins (int): the number of insertion errors
        n_del (int): the number of deletion errors

    """
    # Reset data counter
    dataset.reset()

    if decode_dir is None:
        decode_dir = 'decode_' + dataset.set + '_ep' + str(epoch) + '_beam' + str(decode_params['recog_beam_width'])
        decode_dir += '_lp' + str(decode_params['recog_length_penalty'])
        decode_dir += '_cp' + str(decode_params['recog_coverage_penalty'])
        decode_dir += '_' + str(decode_params['recog_min_len_ratio']) + '_' + str(decode_params['recog_max_len_ratio'])

        ref_trn_save_path = mkdir_join(models[0].save_path, decode_dir, 'ref.trn')
        hyp_trn_save_path = mkdir_join(models[0].save_path, decode_dir, 'hyp.trn')
    else:
        ref_trn_save_path = mkdir_join(decode_dir, 'ref.trn')
        hyp_trn_save_path = mkdir_join(decode_dir, 'hyp.trn')

    per = 0
    n_sub, n_ins, n_del = 0, 0, 0
    n_phone = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))

    with open(hyp_trn_save_path, 'w') as f_hyp, open(ref_trn_save_path, 'w') as f_ref:
        while True:
            batch, is_new_ep = dataset.next(decode_params['recog_batch_size'])
            best_hyps, _, perm_ids, _ = models[0].decode(
                batch['xs'], decode_params,
                exclude_eos=True,
                ensemble_models=models[1:] if len(models) > 1 else [],
                speakers=batch['speakers'])
            ys = [batch['text'][i] for i in perm_ids]

            for b in range(len(batch['xs'])):
                ref = ys[b]
                hyp = dataset.id2phone(best_hyps[b])

                # Write to trn
                speaker = batch['speakers'][b]
                start = batch['utt_ids'][b].replace('-', '_').split('_')[-2]
                end = batch['utt_ids'][b].replace('-', '_').split('_')[-1]
                f_ref.write(ref + ' (' + speaker + '-' + start + '-' + end + ')\n')
                f_hyp.write(hyp + ' (' + speaker + '-' + start + '-' + end + ')\n')
                logger.info('utt-id: %s' % batch['utt_ids'][b])
                logger.info('Ref: %s' % ref)
                logger.info('Hyp: %s' % hyp)
                logger.info('-' * 150)

                # Compute PER
                per_b, sub_b, ins_b, del_b = compute_wer(ref=ref.split(' '),
                                                         hyp=hyp.split(' '),
                                                         normalize=False)
                per += per_b
                n_sub += sub_b
                n_ins += ins_b
                n_del += del_b
                n_phone += len(ref.split(' '))
                # logger.info('PER: %d%%' % (per_b / len(ref.split(' '))))

                if progressbar:
                    pbar.update(1)

            if is_new_ep:
                break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    per /= n_phone
    n_sub /= n_phone
    n_ins /= n_phone
    n_del /= n_phone

    return per, n_sub, n_ins, n_del
