#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method by Character Error Rate (Switchboard corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm

from utils.io.labels.character import Idx2char
from utils.evaluation.edit_distance import compute_cer, compute_wer, wer_align
from examples.swbd.metrics.glm import GLM
from examples.swbd.metrics.post_processing import fix_trans


def do_eval_cer(model, model_type, dataset, label_type, beam_width,
                max_decode_len, eval_batch_size=None, progressbar=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        model: the model to evaluate
        model_type (string): ctc or attention or hierarchical_ctc or
            hierarchical_attention
        dataset: An instance of a `Dataset' class
        label_type (string): character or character_capital_divide
        beam_width: (int): the size of beam
        max_decode_len (int): the length of output sequences
            to stop prediction when EOS token have not been emitted.
            This is used for seq2seq models.
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
    Returns:
        cer_mean (float): An average of CER
        wer_mean (float): An average of WER
    """
    # Reset data counter
    dataset.reset()

    if label_type == 'character':
        idx2char = Idx2char(
            vocab_file_path='../metrics/vocab_files/character_' + dataset.data_size + '.txt')
    elif label_type == 'character_capital_divide':
        idx2char = Idx2char(
            vocab_file_path='../metrics/vocab_files/character_capital_divide_' +
            dataset.data_size + '.txt',
            capital_divide=True)

    # Read GLM file
    glm = GLM(
        glm_path='/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002T43/reference/en20000405_hub5.glm')

    cer_mean, wer_mean = 0, 0
    skip_utt_num = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))  # TODO: fix this
    while True:
        batch, is_new_epoch = dataset.next(batch_size=eval_batch_size)

        # Decode
        if model_type in ['ctc', 'attention']:
            best_hyps = model.decode(batch['xs'], batch['x_lens'],
                                     beam_width=beam_width,
                                     max_decode_len=max_decode_len)
            ys = batch['ys']
            y_lens = batch['y_lens']
        elif model_type in ['hierarchical_ctc', 'hierarchical_attention']:
            best_hyps = model.decode(batch['xs'], batch['x_lens'],
                                     beam_width=beam_width,
                                     max_decode_len=max_decode_len,
                                     is_sub_task=True)
            ys = batch['ys_sub']
            y_lens = batch['y_lens_sub']

        for i_batch in range(len(batch['xs'])):

            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                str_ref = ys[i_batch][0]
                # NOTE: transcript is seperated by space('_')
            else:
                # Convert from list of index to string
                if model_type in ['ctc', 'hierarchical_ctc']:
                    str_ref = idx2char(ys[i_batch][:y_lens[i_batch]])
                elif model_type in ['attention', 'hierarchical_attention']:
                    str_ref = idx2char(ys[i_batch][1:y_lens[i_batch] - 1])
                    # NOTE: Exclude <SOS> and <EOS>

            ##############################
            # Hypothesis
            ##############################
            str_hyp = idx2char(best_hyps[i_batch])
            if model_type in ['attention', 'hierarchical_attention']:
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

            # print('\n' + str_ref)
            # print(str_hyp)

            # Compute WER
            if len(str_ref) > 0:
                # Compute WER
                wer_mean += compute_wer(ref=str_ref.split('_'),
                                        hyp=str_hyp.split('_'),
                                        normalize=True)
                # substitute, insert, delete = wer_align(
                #     ref=str_hyp.split('_'),
                #     hyp=str_ref.split('_'))
                # print('SUB: %d' % substitute)
                # print('INS: %d' % insert)
                # print('DEL: %d' % delete)

                # Compute CER
                cer_mean += compute_cer(ref=str_ref.replace('_', ''),
                                        hyp=str_hyp.replace('_', ''),
                                        normalize=True)
                # NOTE: remove space
            else:
                skip_utt_num += 1

            if progressbar:
                pbar.update(len(batch['xs']))

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    cer_mean /= (len(dataset) - skip_utt_num)
    wer_mean /= (len(dataset) - skip_utt_num)

    return cer_mean, wer_mean
