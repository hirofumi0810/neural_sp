#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate the wordpiece-level model by WER."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sentencepiece as spm
from tqdm import tqdm

from neural_sp.evaluators.edit_distance import compute_wer
from neural_sp.utils.general import mkdir_join


def eval_wordpiece(models, dataset, decode_params, wp_model, epoch, progressbar=False):
    """Evaluate the wordpiece-level model by WER.

    Args:
        models (list): the models to evaluate
        dataset: An instance of a `Dataset' class
        decode_params (dict):
        wp_model ():
        epoch (int):
        progressbar (bool): if True, visualize the progressbar
    Returns:
        wer (float): Word error rate
        num_sub (int): the number of substitution errors
        num_ins (int): the number of insertion errors
        num_del (int): the number of deletion errors
        decode_dir (str):

    """
    # Reset data counter
    dataset.reset()

    model = models[0]
    # TODO(hirofumi): ensemble decoding

    decode_dir = 'decode_' + dataset.set + '_ep' + str(epoch) + '_beam' + str(decode_params['beam_width'])
    decode_dir += '_lp' + str(decode_params['length_penalty'])
    decode_dir += '_cp' + str(decode_params['coverage_penalty'])
    decode_dir += '_' + str(decode_params['min_len_ratio']) + '_' + str(decode_params['max_len_ratio'])
    decode_dir += '_rnnlm' + str(decode_params['rnnlm_weight'])

    ref_trn_save_path = mkdir_join(model.save_path, decode_dir, 'ref.trn')
    hyp_trn_save_path = mkdir_join(model.save_path, decode_dir, 'hyp.trn')

    sp = spm.SentencePieceProcessor()
    sp.Load(wp_model)

    wer = 0
    num_sub, num_ins, num_del, = 0, 0, 0
    num_words = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))  # TODO(hirofumi): fix this

    with open(hyp_trn_save_path, 'w') as f_hyp, open(ref_trn_save_path, 'w') as f_ref:
        while True:
            batch, is_new_epoch = dataset.next(decode_params['batch_size'])
            best_hyps, aw, perm_idx = model.decode(batch['xs'], decode_params,
                                                   exclude_eos=True)
            ys = [batch['ys'][i] for i in perm_idx]

            for b in range(len(batch['xs'])):
                # Reference
                if dataset.is_test:
                    text_ref = ys[b]
                else:
                    wp_list_ref = dataset.idx2word(ys[b], return_list=True)
                    text_ref = sp.DecodePieces(wp_list_ref)

                # Hypothesis
                wp_list_hyp = dataset.idx2word(best_hyps[b], return_list=True)
                text_hyp = sp.DecodePieces(wp_list_hyp)

                # Write to trn
                speaker = '_'.join(batch['utt_ids'][b].replace('-', '_').split('_')[:-2])
                start = batch['utt_ids'][b].replace('-', '_').split('_')[-2].encode('utf-8')
                end = batch['utt_ids'][b].replace('-', '_').split('_')[-1].encode('utf-8')
                f_ref.write(text_ref + ' (' + speaker + '-' + start + '-' + end + ')\n')
                f_hyp.write(text_hyp + ' (' + speaker + '-' + start + '-' + end + ')\n')

                # Compute WER
                wer_b, sub_b, ins_b, del_b = compute_wer(ref=text_ref.split(' '),
                                                         hyp=text_hyp.split(' '),
                                                         normalize=False)
                wer += wer_b
                num_sub += sub_b
                num_ins += ins_b
                num_del += del_b
                num_words += len(text_ref.split(' '))

                if progressbar:
                    pbar.update(1)

            if is_new_epoch:
                break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    wer /= num_words
    num_sub /= num_words
    num_ins /= num_words
    num_del /= num_words

    return wer, num_sub, num_ins, num_del, os.path.join(model.save_path, decode_dir)
