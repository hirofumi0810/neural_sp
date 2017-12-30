#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method by Word Error Rate (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tqdm import tqdm

from utils.io.labels.word import Idx2word
from utils.evaluation.edit_distance import compute_wer, wer_align


def do_eval_wer(model, model_type, dataset, label_type, beam_width,
                max_decode_len, eval_batch_size=None,
                progressbar=False):
    """Evaluate trained model by Word Error Rate.
    Args:
        model: the model to evaluate
        model_type (string): ctc or attention or hierarchical_ctc or
            hierarchical_attention
        dataset: An instance of a `Dataset' class
        label_type (string): word_freq1 or word_freq5 or word_freq10 or word_freq15
        beam_width: (int): the size of beam
        max_decode_len (int): the length of output sequences
            to stop prediction when EOS token have not been emitted.
            This is used for seq2seq models.
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
    Returns:
        wer_mean (float): An average of WER
    """
    # Reset data counter
    dataset.reset()

    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset._batch_size = eval_batch_size

    idx2word = Idx2word(
        vocab_file_path='../metrics/vocab_files/' +
        label_type + '_' + dataset.data_size + '.txt')

    wer_mean = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))
    for batch, is_new_epoch in dataset:

        # Decode
        if model_type in ['ctc', 'attention']:
            inputs, labels, inputs_seq_len, labels_seq_len, _ = batch
        elif model_type in ['hierarchical_ctc', 'hierarchical_attention']:
            inputs, labels, _, inputs_seq_len, labels_seq_len, _, _ = batch
        labels_hyp = model.decode(inputs, inputs_seq_len,
                                  beam_width=beam_width,
                                  max_decode_len=max_decode_len)

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
                    str_ref = idx2word(
                        labels[i_batch][:labels_seq_len[i_batch]])
                elif model_type in ['attention', 'hierarchical_attention']:
                    str_ref = idx2word(
                        labels[i_batch][1:labels_seq_len[i_batch] - 1])
                    # NOTE: Exclude <SOS> and <EOS>

            ##############################
            # Hypothesis
            ##############################
            str_hyp = idx2word(labels_hyp[i_batch])
            if model_type in ['attention', 'hierarchical_attention']:
                str_hyp = str_hyp.split('>')[0]
                # NOTE: Trancate by the first <EOS>

                # Remove the last space
                if len(str_hyp) > 0 and str_hyp[-1] == '_':
                    str_hyp = str_hyp[:-1]

            ##############################
            # Post-proccessing
            ##############################
            # Remove garbage labels
            str_ref = re.sub(r'[\'<>]+', '', str_ref)
            str_hyp = re.sub(r'[\'<>]+', '', str_hyp)
            # TODO: WER計算するときに消していい？

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

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    # Register original batch size
    dataset.reset()

    wer_mean /= len(dataset)

    return wer_mean
