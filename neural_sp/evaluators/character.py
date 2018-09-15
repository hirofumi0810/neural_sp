#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate the character-level model by WER & CER."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tqdm import tqdm

from neural_sp.evaluators.edit_distance import compute_wer
from neural_sp.utils.general import mkdir_join


def eval_char(models, dataset, decode_params, epoch, progressbar=False):
    """Evaluate the character-level model by WER & CER.

    Args:
        models (list): the models to evaluate
        dataset: An instance of a `Dataset' class
        decode_params (dict):
        epoch (int):
        progressbar (bool): if True, visualize the progressbar
    Returns:
        wer (float): Word error rate
        num_sub (int): the number of substitution errors
        num_ins (int): the number of insertion errors
        num_del (int): the number of deletion errors
        cer (float): Character error rate
        num_sub (int): the number of substitution errors
        num_ins (int): the number of insertion errors
        num_del (int): the number of deletion errors
        decode_dir (str):

    """
    # Reset data counter
    dataset.reset()

    model = models[0]

    decode_dir = 'decode_' + dataset.set + '_ep' + str(epoch) + '_beam' + str(decode_params['beam_width'])
    decode_dir += '_lp' + str(decode_params['length_penalty'])
    decode_dir += '_cp' + str(decode_params['coverage_penalty'])
    decode_dir += '_' + str(decode_params['min_len_ratio']) + '_' + str(decode_params['max_len_ratio'])
    decode_dir += '_rnnlm' + str(decode_params['rnnlm_weight'])

    ref_trn_save_path = mkdir_join(model.save_path, decode_dir, 'ref.trn')
    hyp_trn_save_path = mkdir_join(model.save_path, decode_dir, 'hyp.trn')

    wer, cer = 0, 0
    num_sub_w, num_ins_w, num_del_w = 0, 0, 0
    num_sub_c, num_ins_c, num_del_c = 0, 0, 0
    num_words, num_chars = 0, 0
    if progressbar:
        pbar = tqdm(total=len(dataset))

    with open(hyp_trn_save_path, 'w') as f_hyp, open(ref_trn_save_path, 'w') as f_ref:
        while True:
            batch, is_new_epoch = dataset.next(decode_params['batch_size'])
            best_hyps, aw, perm_idx = model.decode(batch['xs'], decode_params,
                                                   exclude_eos=True)
            # task_index = 0
            ys = [batch['ys'][i] for i in perm_idx]

            for b in range(len(batch['xs'])):
                # Reference
                if dataset.is_test:
                    text_ref = ys[b]
                else:
                    text_ref = dataset.idx2char(ys[b])

                # Hypothesis
                text_hyp = dataset.idx2char(best_hyps[b])

                # Write to trn
                speaker = '_'.join(batch['utt_ids'][b].replace('-', '_').split('_')[:-2])
                start = batch['utt_ids'][b].replace('-', '_').split('_')[-2]
                end = batch['utt_ids'][b].replace('-', '_').split('_')[-1]
                f_ref.write(text_ref + ' (' + speaker + '-' + start + '-' + end + ')\n')
                f_hyp.write(text_hyp + ' (' + speaker + '-' + start + '-' + end + ')\n')

                if ('character' in dataset.label_type and 'nowb' not in dataset.label_type) or (task_index > 0 and dataset.label_type_sub == 'character'):
                    # Compute WER
                    wer_b, sub_b, ins_b, del_b = compute_wer(
                        ref=text_ref.split(' '),
                        hyp=text_hyp.split(' '),
                        normalize=False)
                    wer += wer_b
                    num_sub_w += sub_b
                    num_ins_w += ins_b
                    num_del_w += del_b
                    num_words += len(text_ref.split(' '))

                # Compute CER
                cer_b, sub_b, ins_b, del_b = compute_wer(
                    ref=list(text_ref.replace(' ', '')),
                    hyp=list(text_hyp.replace(' ', '')),
                    normalize=False)
                cer += cer_b
                num_sub_c += sub_b
                num_ins_c += ins_b
                num_del_c += del_b
                num_chars += len(text_ref)

                if progressbar:
                    pbar.update(1)

            if is_new_epoch:
                break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    if ('character' in dataset.label_type and 'nowb' not in dataset.label_type) or (task_index > 0 and dataset.label_type_sub == 'character'):
        wer /= num_words
        num_sub_w /= num_words
        num_ins_w /= num_words
        num_del_w /= num_words
    else:
        wer = num_sub_w = num_ins_w = num_del_w = 0

    cer /= num_chars
    num_sub_c /= num_chars
    num_ins_c /= num_chars
    num_del_c /= num_chars

    return (wer, num_sub_w, num_ins_w, num_del_w), (cer, num_sub_c, num_ins_c, num_del_c), os.path.join(model.save_path, decode_dir)
