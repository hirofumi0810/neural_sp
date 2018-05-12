#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method by Word Error Rate (Switchboard corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import pandas as pd
import numpy as np

from examples.swbd.s5c.exp.metrics.glm import GLM
from examples.swbd.s5c.exp.metrics.post_processing import fix_trans
from utils.io.labels.character import Idx2char, Char2idx
from utils.io.labels.word import Idx2word
from utils.evaluation.edit_distance import compute_wer
from utils.evaluation.resolving_unk import resolve_unk


def eval_word(models, dataset, beam_width, max_decode_len,
              beam_width_sub=1, max_decode_len_sub=300,
              eval_batch_size=None, length_penalty=0,
              progressbar=False, temperature=1,
              resolving_unk=False, a2c_oracle=False):
    """Evaluate trained model by Word Error Rate.
    Args:
        models (list): the models to evaluate
        dataset: An instance of a `Dataset' class
        max_decode_len (int): the length of output sequences
            to stop prediction. This is used for seq2seq models.
        beam_width_sub (int, optional): the size of beam in ths sub task
            This is used for the nested attention
        max_decode_len_sub (int, optional): the length of output sequences
            to stop prediction. This is used for the nested attention
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
        temperature (int, optional):
        resolving_unk (bool, optional):
        a2c_oracle (bool, optional):
    Returns:
        wer (float): Word error rate
        df_wer (pd.DataFrame): dataframe of substitution, insertion, and deletion
    """
    # Reset data counter
    dataset.reset()

    idx2word = Idx2word(dataset.vocab_file_path)
    if models[0].model_type == 'nested_attention':
        char2idx = Char2idx(dataset.vocab_file_path_sub)
    if models[0] in ['ctc', 'attention'] and resolving_unk:
        idx2char = Idx2char(dataset.vocab_file_path_sub,
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

        batch_size = len(batch['xs'])

        # Decode
        if len(models) > 1:
            assert models[0].model_type in ['ctc']
            for i, model in enumerate(models):
                probs, x_lens, perm_idx = model.posteriors(
                    batch['xs'], batch['x_lens'])
                if i == 0:
                    probs_ensenmble = probs
                else:
                    probs_ensenmble += probs
            probs_ensenmble /= len(models)

            best_hyps = models[0].decode_from_probs(
                probs_ensenmble, x_lens, beam_width=1)
        else:
            model = models[0]
            # TODO: fix this

            if model.model_type == 'nested_attention':
                if a2c_oracle:
                    if dataset.is_test:
                        max_label_num = 0
                        for b in range(batch_size):
                            if max_label_num < len(list(batch['ys_sub'][b][0])):
                                max_label_num = len(
                                    list(batch['ys_sub'][b][0]))

                        ys_sub = np.zeros(
                            (batch_size, max_label_num), dtype=np.int32)
                        ys_sub -= 1  # pad with -1
                        y_lens_sub = np.zeros((batch_size,), dtype=np.int32)
                        for b in range(batch_size):
                            indices = char2idx(batch['ys_sub'][b][0])
                            ys_sub[b, :len(indices)] = indices
                            y_lens_sub[b] = len(indices)
                            # NOTE: transcript is seperated by space('_')
                else:
                    ys_sub = batch['ys_sub']
                    y_lens_sub = batch['y_lens_sub']

                best_hyps, aw, best_hyps_sub, aw_sub, perm_idx = model.decode(
                    batch['xs'], batch['x_lens'],
                    beam_width=beam_width,
                    beam_width_sub=beam_width_sub,
                    max_decode_len=max_decode_len,
                    max_decode_len_sub=max_label_num if a2c_oracle else max_decode_len_sub,
                    length_penalty=length_penalty,
                    teacher_forcing=a2c_oracle,
                    ys_sub=ys_sub,
                    y_lens_sub=y_lens_sub)
            else:
                best_hyps, aw, perm_idx = model.decode(
                    batch['xs'], batch['x_lens'],
                    beam_width=beam_width,
                    max_decode_len=max_decode_len,
                    length_penalty=length_penalty)
                if resolving_unk:
                    best_hyps_sub, aw_sub, _ = model.decode(
                        batch['xs'], batch['x_lens'],
                        beam_width=beam_width,
                        max_decode_len=max_decode_len_sub,
                        length_penalty=length_penalty,
                        task_index=1)

        ys = batch['ys'][perm_idx]
        y_lens = batch['y_lens'][perm_idx]

        for b in range(batch_size):

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

            if len(str_ref) == 0:
                if progressbar:
                    pbar.update(1)
                continue

            # Compute WER
            try:
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
