#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method by Character Error Rate (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tqdm import tqdm

from utils.io.labels.character import Idx2char
from utils.evaluation.edit_distance import compute_cer, compute_wer, wer_align


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
        cer_mean (float): An average of CER
        wer_mean (float): An average of WER
    """
    # Reset data counter
    dataset.reset()

    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    idx2char = Idx2char(
        vocab_file_path='../metrics/vocab_files/' +
        label_type + '_' + dataset.data_size + '.txt')

    cer_mean, wer_mean = 0, 0
    if progressbar:
        pbar = tqdm(total=len(dataset))
    for batch, is_new_epoch in dataset:

        if model_type in ['ctc', 'attention']:
            inputs, labels, inputs_seq_len, labels_seq_len, _ = batch
        elif model_type in ['hierarchical_ctc', 'hierarchical_attention']:
            inputs, _, labels, inputs_seq_len, _, labels_seq_len, _ = batch

        # Decode the ensemble
        if model_type in ['attention', 'ctc']:
            for i, model in enumerate(models):
                probs_i = model.posteriors(
                    inputs, inputs_seq_len, temperature=temperature)
                if i == 0:
                    probs = probs_i
                else:
                    probs += probs_i
                # NOTE: probs: `[1 (B), T, num_classes]`
            probs /= len(models)

            labels_hyp = model.decode_from_probs(
                probs, inputs_seq_len,
                beam_width=beam_width,
                max_decode_len=max_decode_len)
        elif model_type in['hierarchical_attention', 'hierarchical_ctc']:
            raise NotImplementedError
            # labels_hyp = model.decode(
            #     inputs, inputs_seq_len,
            #     beam_width=beam_width,
            #     max_decode_len=max_decode_len,
            #     is_sub_task=True)

        for i_batch in range(inputs.shape[0]):

            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                str_ref = labels[i_batch][0]
                # NOTE: transcript is seperated by space('_')
            else:
                # Convert from list of index to string
                if model_type in ['ctc', 'hierarchical_ctc']:
                    str_ref = idx2char(
                        labels[i_batch][:labels_seq_len[i_batch]])
                elif model_type in ['attention', 'hierarchical_attention']:
                    str_ref = idx2char(
                        labels[i_batch][1:labels_seq_len[i_batch] - 1])
                    # NOTE: Exclude <SOS> and <EOS>

            ##############################
            # Hypothesis
            ##############################
            str_hyp = idx2char(labels_hyp[i_batch])

            if model_type in ['attention', 'hierarchical_attention']:
                str_hyp = str_hyp.split('>')[0]
                # NOTE: Trancate by the first <EOS>

            # Remove consecutive spaces
            str_hyp = re.sub(r'[_]+', '_', str_hyp)

            # Remove garbage labels
            str_ref = re.sub(r'[\'<>]+', '', str_ref)
            str_hyp = re.sub(r'[\'<>]+', '', str_hyp)

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
            cer_mean += compute_cer(ref=str_ref,
                                    hyp=str_hyp,
                                    normalize=True)

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    # Register original batch size
    dataset.reset()

    cer_mean /= len(dataset)
    wer_mean /= len(dataset)

    return cer_mean, wer_mean
