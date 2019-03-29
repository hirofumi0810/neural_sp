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


def eval_phone(models, dataset, recog_params, epoch,
               recog_dir=None, progressbar=False):
    """Evaluate a phone-level model by PER.

    Args:
        models (list): the models to evaluate
        dataset: An instance of a `Dataset' class
        recog_params (dict):
        epoch (int):
        recog_dir (str):
        progressbar (bool): if True, visualize the progressbar
    Returns:
        per (float): Phone error rate
        n_sub (int): the number of substitution errors
        n_ins (int): the number of insertion errors
        n_del (int): the number of deletion errors

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
            best_hyps, _, perm_ids, _ = models[0].decode(
                batch['xs'], recog_params,
                exclude_eos=True,
                idx2token=dataset.idx2phone,
                refs=batch['ys'],
                ensemble_models=models[1:] if len(models) > 1 else [],
                speakers=batch['sessions'] if dataset.corpus == 'swbd' else batch['speakers'])
            ys = [batch['text'][i] for i in perm_ids]

            for b in range(len(batch['xs'])):
                ref = ys[b]
                hyp = dataset.idx2phone(best_hyps[b])

                # Write to trn
                utt_id = str(batch['utt_ids'][b])
                speaker = str(batch['speakers'][b]).replace('-', '_')
                f_ref.write(ref + ' (' + speaker + '-' + utt_id + ')\n')
                f_hyp.write(hyp + ' (' + speaker + '-' + utt_id + ')\n')
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

                if progressbar:
                    pbar.update(1)

            if is_new_epoch:
                break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    per /= n_phone
    n_sub /= n_phone
    n_ins /= n_phone
    n_del /= n_phone

    logger.info('PER (%s): %.2f %%' % (dataset.set, per))
    logger.info('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub, n_ins, n_del))

    return per, n_sub, n_ins, n_del
