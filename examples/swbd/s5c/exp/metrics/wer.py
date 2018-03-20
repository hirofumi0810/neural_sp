#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method by Word Error Rate (Switchboard corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import pandas as pd

from examples.swbd.s5c.exp.metrics.glm import GLM
from examples.swbd.s5c.exp.metrics.post_processing import fix_trans
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.evaluation.edit_distance import compute_wer
from utils.evaluation.resolving_unk import resolve_unk


def do_eval_wer(model, dataset, beam_width, max_decode_len,
                eval_batch_size=None, progressbar=False, resolving_unk=False):
    """Evaluate trained model by Word Error Rate.
    Args:
        model: the model to evaluate
        dataset: An instance of a `Dataset' class
        beam_width: (int): the size of beam
        max_decode_len (int): the length of output sequences
            to stop prediction when EOS token have not been emitted.
            This is used for seq2seq models.
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
        resolving_unk (bool, optional):
    Returns:
        wer (float): Word error rate
        df_wer (pd.DataFrame): dataframe of substitution, insertion, and deletion
    """
    # Reset data counter
    dataset.reset()

    idx2word = Idx2word(vocab_file_path=dataset.vocab_file_path)
    idx2char = Idx2char(vocab_file_path=dataset.vocab_file_path_sub,
                        capital_divide=dataset.label_type_sub == 'character_capital_divide')

    # Read GLM file
    glm = GLM(
        glm_path='/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002T43/reference/en20000405_hub5.glm')

    wer = 0
    sub, ins, dele, = 0, 0, 0
    num_words = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))  # TODO: fix this
    while True:
        batch, is_new_epoch = dataset.next(batch_size=eval_batch_size)

        # Decode
        if model.model_type == 'nested_attention':
            best_hyps, _, perm_idx = model.decode(
                batch['xs'], batch['x_lens'],
                beam_width=beam_width,
                max_decode_len=max_decode_len,
                max_decode_len_sub=100)
        else:
            if resolving_unk:
                best_hyps, aw, perm_idx = model.decode(
                    batch['xs'], batch['x_lens'],
                    beam_width=beam_width,
                    max_decode_len=max_decode_len,
                    resolving_unk=True)
                best_hyps_sub, aw_sub, _ = model.decode(
                    batch['xs'], batch['x_lens'],
                    beam_width=beam_width,
                    max_decode_len=max_decode_len * 3,
                    is_sub_task=True, resolving_unk=True)
            else:
                best_hyps, perm_idx = model.decode(
                    batch['xs'], batch['x_lens'],
                    beam_width=beam_width,
                    max_decode_len=max_decode_len)

        ys = batch['ys'][perm_idx]
        y_lens = batch['y_lens'][perm_idx]

        for b in range(len(batch['xs'])):

            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                str_ref = ys[b][0]
                # NOTE: transcript is seperated by space('_')
            else:
                # Convert from list of index to string
                str_ref = idx2word(ys[b][:y_lens[b]])

            ##############################
            # Hypothesis
            ##############################
            str_hyp = idx2word(best_hyps[b])
            if 'attention' in model.model_type:
                str_hyp = str_hyp.split('>')[0]
                # NOTE: Trancate by the first <EOS>

                # Remove the last space
                if len(str_hyp) > 0 and str_hyp[-1] == '_':
                    str_hyp = str_hyp[:-1]

            ##############################
            # Resolving UNK
            ##############################
            if resolving_unk and 'OOV' in str_hyp:
                str_hyp = resolve_unk(
                    str_hyp, best_hyps_sub[b], aw[b], aw_sub[b], idx2char)
                str_hyp = str_hyp.replace('*', '')

            ##############################
            # Post-proccessing
            ##############################
            str_ref = fix_trans(str_ref, glm)
            str_hyp = fix_trans(str_hyp, glm)

            # print('REF: %s' % str_ref)
            # print('HYP: %s' % str_hyp)

            if len(str_ref) == 0:
                if progressbar:
                    pbar.update(1)
                continue

            try:
                # Compute WER
                wer_b, sub_b, ins_b, del_b = compute_wer(
                    ref=str_ref.split('_'),
                    hyp=str_hyp.split('_'),
                    normalize=False)
                wer += wer_b
                sub += sub_b
                ins += ins_b
                dele += del_b
                num_words += len(str_ref.split('_'))
            except:
                # print('REF: %s' % str_ref)
                # print('HYP: %s' % str_hyp)
                pass

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    wer /= num_words
    sub /= num_words
    ins /= num_words
    dele /= num_words

    df_wer = pd.DataFrame(
        {'SUB': [sub * 100], 'INS': [ins * 100], 'DEL': [dele * 100]},
        columns=['SUB', 'INS', 'DEL'], index=['WER'])

    return wer, df_wer
