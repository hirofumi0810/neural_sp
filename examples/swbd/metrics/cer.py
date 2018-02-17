#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method by Character Error Rate (Switchboard corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import pandas as pd

from utils.io.labels.character import Idx2char
from utils.evaluation.edit_distance import compute_wer
from examples.swbd.metrics.glm import GLM
from examples.swbd.metrics.post_processing import fix_trans


def do_eval_cer(model, dataset, beam_width,
                max_decode_len, eval_batch_size=None, progressbar=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        model: the model to evaluate
        dataset: An instance of a `Dataset' class
        beam_width: (int): the size of beam
        max_decode_len (int): the length of output sequences
            to stop prediction when EOS token have not been emitted.
            This is used for seq2seq models.
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
    Returns:
        cer_mean (float): An average of CER
        wer_mean (float): An average of WER
        df_cer ():
    """
    # Reset data counter
    dataset.reset()

    if model.model_type in ['ctc', 'attention']:
        idx2char = Idx2char(
            vocab_file_path=dataset.vocab_file_path,
            capital_divide=(dataset.label_type == 'character_capital_divide'))
    else:
        idx2char = Idx2char(
            vocab_file_path=dataset.vocab_file_path_sub,
            capital_divide=(dataset.label_type_sub == 'character_capital_divide'))

    # Read GLM file
    glm = GLM(
        glm_path='/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002T43/reference/en20000405_hub5.glm')

    cer_mean, wer_mean = 0, 0
    sub_char, ins_char, del_char = 0, 0, 0
    sub_word, ins_word, del_word = 0, 0, 0
    skip_utt_num = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))  # TODO: fix this
    while True:
        batch, is_new_epoch = dataset.next(batch_size=eval_batch_size)

        # Decode
        if model.model_type in ['ctc', 'attention']:
            best_hyps, perm_idx = model.decode(batch['xs'], batch['x_lens'],
                                               beam_width=beam_width,
                                               max_decode_len=max_decode_len)
            ys = batch['ys'][perm_idx]
            y_lens = batch['y_lens'][perm_idx]
        elif 'attention' in model.model_type:
            best_hyps, perm_idx = model.decode(batch['xs'], batch['x_lens'],
                                               beam_width=beam_width,
                                               max_decode_len=max_decode_len,
                                               is_sub_task=True)
            ys = batch['ys_sub'][perm_idx]
            y_lens = batch['y_lens_sub'][perm_idx]

        for b in range(len(batch['xs'])):

            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                str_ref = ys[b][0]
                # NOTE: transcript is seperated by space('_')
            else:
                # Convert from list of index to string
                if model.model_type in ['ctc', 'hierarchical_ctc']:
                    str_ref = idx2char(ys[b][:y_lens[b]])
                elif 'attention' in model.model_type:
                    str_ref = idx2char(ys[b][1:y_lens[b] - 1])
                    # NOTE: Exclude <SOS> and <EOS>

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

            ##############################
            # Post-proccessing
            ##############################
            str_ref = fix_trans(str_ref, glm)
            str_hyp = fix_trans(str_hyp, glm)

            # print('REF: %s' % str_ref)
            # print('HYP: %s' % str_hyp)

            # Compute WER
            if len(str_ref) > 0:
                # Compute WER
                wer_b, sub_b, ins_b, del_b = compute_wer(
                    ref=str_ref.split('_'),
                    hyp=str_hyp.split('_'),
                    normalize=True)
                wer_mean += wer_b
                sub_word += sub_b
                ins_word += ins_b
                del_word += del_b

                # Compute CER
                cer_b, sub_b, ins_b, del_b = compute_wer(
                    ref=list(str_ref.replace('_', '')),
                    hyp=list(str_hyp.replace('_', '')),
                    normalize=True)
                cer_mean += cer_b
                sub_char += sub_b
                ins_char += ins_b
                del_char += del_b
            else:
                skip_utt_num += 1

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    cer_mean /= (len(dataset) - skip_utt_num)
    wer_mean /= (len(dataset) - skip_utt_num)

    df_cer = pd.DataFrame(
        {'SUB': [sub_char, sub_word],
         'INS': [ins_char, ins_word],
         'DEL': [del_char, del_word]},
        columns=['SUB', 'INS', 'DEL'],
        index=['CER', 'WER'])

    return cer_mean, wer_mean, df_cer
