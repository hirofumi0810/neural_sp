#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate a phene-level model by PER."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import six
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
        num_sub (int): the number of substitution errors
        num_ins (int): the number of insertion errors
        num_del (int): the number of deletion errors
        decode_dir (str):

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

    per = 0
    num_sub, num_ins, num_del = 0, 0, 0
    num_phones = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))

    with open(hyp_trn_save_path, 'w') as f_hyp, open(ref_trn_save_path, 'w') as f_ref:
        while True:
            batch, is_new_epoch = dataset.next(decode_params['batch_size'])
            best_hyps, _, perm_idx = model.decode(batch['xs'], decode_params,
                                                  exclude_eos=True)
            ys = [batch['ys'][i] for i in perm_idx]

            for b in six.moves.range(len(batch['xs'])):
                # Reference
                if dataset.is_test:
                    ref = ys[b]
                else:
                    ref = dataset.idx2phone(ys[b])

                # Hypothesis
                hyp = dataset.idx2phone(best_hyps[b])

                # Write to trn
                speaker = '_'.join(batch['utt_ids'][b].replace('-', '_').split('_')[:-2])
                start = batch['utt_ids'][b].replace('-', '_').split('_')[-2]
                end = batch['utt_ids'][b].replace('-', '_').split('_')[-1]
                f_ref.write(ref + ' (' + speaker + '-' + start + '-' + end + ')\n')
                f_hyp.write(hyp + ' (' + speaker + '-' + start + '-' + end + ')\n')
                logger.info('utt-id: %s' % batch['utt_ids'][b])
                logger.info('Ref: %s' % ref.lower())
                logger.info('Hyp: %s' % hyp)

                # Compute PER
                per_b, sub_b, ins_b, del_b = compute_wer(ref=ref.split(' '),
                                                         hyp=hyp.split(' '),
                                                         normalize=False)
                per += per_b
                num_sub += sub_b
                num_ins += ins_b
                num_del += del_b
                num_phones += len(ref.split(' '))
                # logger.info('PER: %d%%' % (per_b / len(ref.split(' '))))

                if progressbar:
                    pbar.update(1)

            if is_new_epoch:
                break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    per /= num_phones
    num_sub /= num_phones
    num_ins /= num_phones
    num_del /= num_phones

    return per, num_sub, num_ins, num_del, os.path.join(model.save_path, decode_dir)
