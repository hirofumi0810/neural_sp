#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method for the Attention-based model (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tqdm import tqdm

from utils.io.labels.character import Idx2char
from utils.io.variable import np2var_pytorch
from utils.evaluation.edit_distance import compute_cer


def do_eval_cer(model, dataset, label_type, train_data_size, beam_width,
                is_test=False, eval_batch_size=None, progressbar=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        model: the model to evaluate
        dataset: An instance of a `Dataset' class
        label_type (string): kanji or kanji or kanji_divide or kana_divide
        beam_width: (int): the size of beam
        is_test (bool, optional): set to True when evaluating by the test set
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
    Returns:
        cer_mean (float): An average of CER
    """
    batch_size_original = dataset.batch_size

    # Reset data counter
    dataset.reset()

    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    if 'kanji' in label_type:
        map_file_path = '../metrics/mapping_files/' + \
            label_type + '_' + train_data_size + '.txt'
    elif 'kana' in label_type:
        map_file_path = '../metrics/mapping_files/' + label_type + '.txt'

    idx2char = Idx2char(map_file_path=map_file_path)

    cer_mean = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))
    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini-batch
        inputs, labels_true, _, labels_seq_len, _ = data
        inputs = np2var_pytorch(inputs, volatile=True)
        if model.use_cuda:
            inputs = inputs.cuda()

        batch_size = inputs[0].size()[0]

        # Evaluate by 39 phones
        labels_pred, _ = model.decode_infer(
            inputs[0], beam_width=beam_width)

        for i_batch in range(batch_size):

            # Convert from list of index to string
            if is_test:
                str_true = labels_true[0][i_batch][0]
                # NOTE: transcript is seperated by space('_')
            else:
                str_true = idx2char(
                    labels_true[0][i_batch][1:labels_seq_len[0][i_batch] - 1])
            str_pred = idx2char(labels_pred[i_batch]).split('>')[0]
            # NOTE: Trancate by <EOS>

            # Remove garbage labels
            str_true = re.sub(r'[_NZー・<>]+', '', str_true)
            str_pred = re.sub(r'[_NZー・<>]+', '', str_pred)

            # Compute CER
            cer_mean += compute_cer(str_pred=str_pred,
                                    str_true=str_true,
                                    normalize=True)

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    cer_mean /= len(dataset)

    # Register original batch size
    if eval_batch_size is not None:
        dataset.batch_size = batch_size_original

    return cer_mean
