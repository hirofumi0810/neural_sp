#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method by Character Error Rate (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tqdm import tqdm
import pandas as pd

from utils.io.labels.character import Idx2char
from utils.evaluation.edit_distance import compute_wer


def do_eval_cer(models, model_type, dataset, label_type, beam_width,
                max_decode_len, eval_batch_size=None, temperature=1,
                progressbar=False):
    """Evaluate trained models by Character Error Rate.
    Args:
        models (list): the model to evaluate
        model_type (string): ctc or attention or hierarchical_ctc or
            hierarchical_attention
        dataset: An instance of a `Dataset' class
        label_type (string): character or character_capital_divide
        beam_width: (int): the size of beam
        max_decode_len (int): the length of output sequences
            to stop prediction when EOS token have not been emitted.
            This is used for seq2seq models.
        eval_batch_size (int, optional): the batch size when evaluating the model
        temperature (int, optional):
        progressbar (bool, optional): if True, visualize the progressbar
    Returns:
        wer (float): Word error rate
        cer (float): Character error rate
        df_wer_cer (pd.DataFrame): dataframe of substitution, insertion, and deletion
    """
    # Reset data counter
    dataset.reset()

    idx2char = Idx2char(
        vocab_file_path=dataset.vocab_file_path,
        capital_divide=(dataset.label_type == 'character_capital_divide'))

    cer, wer = 0, 0
    sub_char, ins_char, del_char = 0, 0, 0
    sub_word, ins_word, del_word = 0, 0, 0
    num_words, num_chars = 0, 0
    if progressbar:
        pbar = tqdm(total=len(dataset))  # TODO: fix this
    while True:
        batch, is_new_epoch = dataset.next(batch_size=eval_batch_size)

        # Decode the ensemble
        if model_type in ['attention', 'ctc']:
            for i, model in enumerate(models):
                probs_i, perm_idx = model.posteriors(
                    batch['xs'], batch['x_lens'], temperature=temperature)
                if i == 0:
                    probs = probs_i
                else:
                    probs += probs_i
                # NOTE: probs: `[1 (B), T, num_classes]`
            probs /= len(models)

            best_hyps = model.decode_from_probs(
                probs, batch['x_lens'][perm_idx],
                beam_width=beam_width,
                max_decode_len=max_decode_len)
            ys = batch['ys'][perm_idx]
            y_lens = batch['y_lens'][perm_idx]

        elif model_type in['hierarchical_attention', 'hierarchical_ctc']:
            raise NotImplementedError

        for b in range(len(batch['xs'])):

            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                str_ref = ys[b][0]
                # NOTE: transcript is seperated by space('_')
            else:
                # Convert from list of index to string
                str_ref = idx2char(ys[b][:y_lens[b]])

            ##############################
            # Hypothesis
            ##############################
            str_hyp = idx2char(best_hyps[b])
            if 'attention' in model.model_type:
                str_hyp = str_hyp.split('>')[0]
                # NOTE: Trancate by the first <EOS>

                # Remove the last space
                if len(str_hyp) > 0 and str_hyp[-1] == '_':
                    str_hyp = str_hyp[:-1]

            # Remove consecutive spaces
            str_hyp = re.sub(r'[_]+', '_', str_hyp)

            ##############################
            # Post-proccessing
            ##############################
            # Remove garbage labels
            str_ref = re.sub(r'[\'>]+', '', str_ref)
            str_hyp = re.sub(r'[\'>]+', '', str_hyp)
            # TODO: WER計算するときに消していい？

            # Compute WER
            wer_b, sub_b, ins_b, del_b = compute_wer(
                ref=str_ref.split('_'),
                hyp=str_hyp.split('_'),
                normalize=False)
            wer += wer_b
            sub_word += sub_b
            ins_word += ins_b
            del_word += del_b
            num_words += len(str_ref.split('_'))

            # Compute CER
            cer_b, sub_b, ins_b, del_b = compute_wer(
                ref=list(str_ref.replace('_', '')),
                hyp=list(str_hyp.replace('_', '')),
                normalize=False)
            cer += cer_b
            sub_char += sub_b
            ins_char += ins_b
            del_char += del_b
            num_chars += len(str_ref.replace('_', ''))

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    wer /= num_words
    cer /= num_chars
    sub_char /= num_chars
    ins_char /= num_chars
    del_char /= num_chars
    sub_word /= num_words
    ins_word /= num_words
    del_word /= num_words

    df_wer_cer = pd.DataFrame(
        {'SUB': [sub_char * 100, sub_word * 100],
         'INS': [ins_char * 100, ins_word * 100],
         'DEL': [del_char * 100, del_word * 100]},
        columns=['SUB', 'INS', 'DEL'], index=['CER', 'WER'])

    return cer, wer, df_wer_cer
