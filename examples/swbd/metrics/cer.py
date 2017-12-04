#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method by Character Error Rate (Switchboard corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tqdm import tqdm

from utils.io.labels.character import Idx2char
from utils.evaluation.edit_distance import compute_cer, compute_wer, wer_align


def do_eval_cer(model, model_type, dataset, label_type, data_size, beam_width,
                max_decode_length, eval_batch_size=None,
                progressbar=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        model: the model to evaluate
        model_type (string): ctc or attention or hierarchical_ctc or
            hierarchical_attention
        dataset: An instance of a `Dataset' class
        label_type (string): character or character_capital_divide
        data_size (string): 300h or 2000h
        beam_width: (int): the size of beam
        max_decode_length (int): the length of output sequences
            to stop prediction when EOS token have not been emitted.
            This is used for seq2seq models.
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
    Returns:
        cer_mean (float): An average of CER
        wer_mean (float): An average of WER
    """
    batch_size_original = dataset.batch_size

    # Reset data counter
    dataset.reset()

    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    idx2char = Idx2char(
        vocab_file_path='../metrics/vocab_files/' +
        label_type + '_' + data_size + '.txt')

    cer_mean, wer_mean = 0, 0
    if progressbar:
        pbar = tqdm(total=len(dataset))
    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini-batch
        if model_type in ['ctc', 'attention']:
            inputs, labels, inputs_seq_len, labels_seq_len, _ = data
        elif model_type in ['hierarchical_ctc', 'hierarchical_attention']:
            inputs, _, labels, inputs_seq_len, _, labels_seq_len, _ = data

        # Decode
        if model_type in ['attention', 'ctc']:
            labels_pred, perm_indices = model.decode(
                inputs, inputs_seq_len,
                beam_width=beam_width,
                max_decode_length=max_decode_length)
        elif model_type in['hierarchical_attention', 'hierarchical_ctc']:
            labels_pred, perm_indices = model.decode_sub(
                inputs, inputs_seq_len,
                beam_width=beam_width,
                max_decode_length=max_decode_length)

        for i_batch in range(inputs.shape[0]):

            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                str_true = labels[i_batch][0]
                # NOTE: transcript is seperated by space('_')
            else:
                # Permutate indices
                labels = labels[perm_indices]
                labels_seq_len = labels_seq_len[perm_indices]

                # Convert from list of index to string
                if model_type in ['ctc', 'hierarchical_ctc']:
                    str_true = idx2char(
                        labels[i_batch][:labels_seq_len[i_batch]])
                elif model_type in ['attention', 'hierarchical_attention']:
                    str_true = idx2char(
                        labels[i_batch][1:labels_seq_len[i_batch] - 1])
                    # NOTE: Exclude <SOS> and <EOS>

            ##############################
            # Hypothesis
            ##############################
            str_pred = idx2char(labels_pred[i_batch])

            if model_type in ['attention', 'hierarchical_attention']:
                str_pred = str_pred.split('>')[0]
                # NOTE: Trancate by the first <EOS>

            # Remove consecutive spaces
            str_pred = re.sub(r'[_]+', '_', str_pred)

            # Remove garbage labels
            str_true = re.sub(r'[\'<>]+', '', str_true)
            str_pred = re.sub(r'[\'<>]+', '', str_pred)

            # Compute WER
            wer_mean += compute_wer(ref=str_true.split('_'),
                                    hyp=str_pred.split('_'),
                                    normalize=True)
            # substitute, insert, delete = wer_align(
            #     ref=str_pred.split('_'),
            #     hyp=str_true.split('_'))
            # print('SUB: %d' % substitute)
            # print('INS: %d' % insert)
            # print('DEL: %d' % delete)

            # Compute CER
            cer_mean += compute_cer(ref=str_true,
                                    hyp=str_pred,
                                    normalize=True)

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    cer_mean /= len(dataset)
    wer_mean /= len(dataset)

    # Register original batch size
    if eval_batch_size is not None:
        dataset.batch_size = batch_size_original

    return cer_mean, wer_mean
