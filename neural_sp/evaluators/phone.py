#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Define evaluation method of phene-level models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
from tqdm import tqdm

from neural_sp.evaluators.edit_distance import compute_wer
from neural_sp.utils.general import mkdir_join


def eval_phone(models, dataset, decode_params, progressbar=False):
    """Evaluate a phone-level model.

    Args:
        models (list): the models to evaluate
        dataset: An instance of a `Dataset' class
        decode_params (dict):
        batch_size (int): the batch size when evaluating the model
        progressbar (bool): if True, visualize the progressbar
    Returns:
        per (float): Phone error rate
        num_sub (int): the number of substitution errors
        num_ins (int): the number of insertion errors
        num_del (int): the number of deletion errors

    """
    # Reset data counter
    dataset.reset()

    model = models[0]
    # TODO(hirofumi): ensemble decoding

    ref_trn_save_path = mkdir_join(model.save_path, 'decode_' + dataset.set + '_ep' +
                                   str(dataset.epoch + 1) + '_beam' + str(decode_params.beam_width), 'ref.trn')
    hyp_trn_save_path = mkdir_join(model.save_path, 'decode_' + dataset.set + '_ep' +
                                   str(dataset.epoch + 1) + '_beam' + str(decode_params.beam_width), 'hyp.trn')

    per = 0
    num_sub, num_ins, num_del = 0, 0, 0
    num_phones = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))

    with codecs.open(hyp_trn_save_path, 'w') as f_hyp, codecs.open(ref_trn_save_path, 'w') as f_ref:
        while True:
            batch, is_new_epoch = dataset.next(decode_params['batch_size'])
            best_hyps, _, perm_idx = model.decode(batch['xs'], decode_params,
                                                  exclude_eos=True)
            ys = [batch['ys'][i] for i in perm_idx]

            for b in range(len(batch['xs'])):
                # Reference
                if dataset.is_test:
                    text_ref = ys[b]  # NOTE: transcript is seperated by space('_')
                else:
                    text_ref = dataset.idx2phone(ys[b])

                # Hypothesis
                text_hyp = dataset.idx2phone(best_hyps[b])

                # Write to trn
                speaker = '_'.join(batch['utt_ids'][b].replace('-', '_').split('_')[:-2])
                start = batch['utt_ids'][b].replace('-', '_').split('_')[-2]
                end = batch['utt_ids'][b].replace('-', '_').split('_')[-1]
                f_ref.write(text_ref + ' (' + speaker + '-' + start + '-' + end + ')\n')
                f_hyp.write(text_hyp + ' (' + speaker + '-' + start + '-' + end + ')\n')

                # Compute PER
                per_b, sub_b, ins_b, del_b = compute_wer(ref=text_ref.split(' '),
                                                         hyp=text_hyp.split(' '),
                                                         normalize=False)
                per += per_b
                num_sub += sub_b
                num_ins += ins_b
                num_del += del_b
                num_phones += len(text_ref.split(' '))

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

    return per, num_sub, num_ins, num_del
