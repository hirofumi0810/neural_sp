#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Define evaluation method of word-level models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm

from neural_sp.evaluators.edit_distance import compute_wer
from neural_sp.evaluators.resolving_unk import resolve_unk
from neural_sp.utils.general import mkdir_join


def eval_word(models, dataset, decode_params, progressbar=False):
    """Evaluate a word-level model.

    Args:
        models (list): the models to evaluate
        dataset: An instance of a `Dataset' class
        decode_params (dict):
        batch_size (int): the batch size when evaluating the model
        progressbar (bool): if True, visualize the progressbar
    Returns:
        wer (float): Word error rate
        num_sub (int): the number of substitution errors
        num_ins (int): the number of insertion errors
        num_del (int): the number of deletion errors

    """
    # Reset data counter
    dataset.reset()

    model = models[0]
    # TODO(hirofumi): ensemble decoding

    ref_trn_save_path = mkdir_join(model.save_path, 'decode_' + dataset.set + '_ep' +
                                   str(dataset.epoch + 1) + '_beam' + str(decode_params['beam_width']), 'ref.trn')
    hyp_trn_save_path = mkdir_join(model.save_path, 'decode_' + dataset.set + '_ep' +
                                   str(dataset.epoch + 1) + '_beam' + str(decode_params['beam_width']), 'hyp.trn')

    wer = 0
    num_sub, num_ins, num_del, = 0, 0, 0
    num_words = 0
    num_oov_total = 0
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
                    text_ref = dataset.idx2word(ys[b])

                # Hypothesis
                text_hyp = dataset.idx2word(best_hyps[b])
                num_oov_total += text_hyp.count('<unk>')

                # Resolving UNK
                if decode_params['resolving_unk'] and '<unk>' in text_hyp:
                    best_hyps_sub, aw_sub, _ = model.decode(
                        batch['xs'][b:b + 1], batch['xs'], decode_params, exclude_eos=True)
                    # task_index=1

                    text_hyp = resolve_unk(
                        text_hyp, best_hyps_sub[0], aw[b], aw_sub[0], dataset.idx2char,
                        diff_time_resolution=2 ** sum(model.subsample_list) // 2 ** sum(model.subsample_list[:model.encoder_num_layers_sub - 1]))
                    text_hyp = text_hyp.replace('*', '')

                # Write to trn
                speaker = '_'.join(batch['utt_ids'][b].replace('-', '_').split('_')[:-2])
                start = batch['utt_ids'][b].replace('-', '_').split('_')[-2]
                end = batch['utt_ids'][b].replace('-', '_').split('_')[-1]
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

    return wer, num_sub, num_ins, num_del
