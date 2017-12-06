#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method by Phone Error Rate (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm

from examples.timit.metrics.mapping import Map2phone39
from utils.io.labels.phone import Idx2phone
from utils.evaluation.edit_distance import compute_per


def do_eval_per(model, model_type, dataset, label_type, beam_width,
                max_decode_length, eval_batch_size=None,
                progressbar=False):
    """Evaluate trained model by Phone Error Rate.
    Args:
        model: the model to evaluate
        model_type (string): ctc or attention or joint_ctc_attention
        dataset: An instance of a `Dataset' class
        label_type (string): phone39 or phone48 or phone61
        beam_width: (int): the size of beam
        max_decode_length (int): the length of output sequences
            to stop prediction when EOS token have not been emitted.
            This is used for seq2seq models.
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
    Returns:
        per_mean (float): An average of PER
    """
    batch_size_original = dataset.batch_size

    # Reset data counter
    dataset.reset()

    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    train_label_type = label_type
    eval_label_type = dataset.label_type

    idx2phone_train = Idx2phone(
        '../metrics/vocab_files/' + train_label_type + '.txt')
    idx2phone_eval = Idx2phone(
        '../metrics/vocab_files/' + eval_label_type + '.txt')
    map2phone39_train = Map2phone39(
        label_type=train_label_type,
        map_file_path='../metrics/phone2phone.txt')
    map2phone39_eval = Map2phone39(
        label_type=eval_label_type,
        map_file_path='../metrics/phone2phone.txt')

    per_mean = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))
    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini-batch
        inputs, labels, inputs_seq_len, labels_seq_len, _ = data

        # Decode
        labels_pred, perm_indices = model.decode(
            inputs, inputs_seq_len,
            beam_width=beam_width,
            max_decode_length=max_decode_length)

        for i_batch in range(inputs.shape[0]):
            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                phone_true_list = labels[i_batch][0].split(' ')
                # NOTE: transcript is seperated by space(' ')
            else:
                # Permutate indices
                labels = labels[perm_indices]
                labels_seq_len = labels_seq_len[perm_indices]

                # Convert from index to phone (-> list of phone strings)
                if model_type == 'ctc':
                    phone_true_list = idx2phone_eval(
                        labels[i_batch][:labels_seq_len[i_batch]]).split(' ')
                elif model_type == 'attention':
                    phone_true_list = idx2phone_eval(
                        labels[i_batch][1:labels_seq_len[i_batch] - 1]).split(' ')
                    # NOTE: Exclude <SOS> and <EOS>

            ##############################
            # Hypothesis
            ##############################
            # Convert from index to phone (-> list of phone strings)
            str_pred = idx2phone_train(labels_pred[i_batch])

            if model_type == 'attention':
                str_pred = str_pred.split('>')[0]
                # NOTE: Trancate by the first <EOS>

                # Remove the last space
                if len(str_pred) > 0 and str_pred[-1] == ' ':
                    str_pred = str_pred[:-1]

            phone_pred_list = str_pred.split(' ')

            # Mapping to 39 phones (-> list of phone strings)
            phone_true_list = map2phone39_eval(phone_true_list)
            phone_pred_list = map2phone39_train(phone_pred_list)

            # Compute PER
            per_mean += compute_per(ref=phone_true_list,
                                    hyp=phone_pred_list,
                                    normalize=True)

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    per_mean /= len(dataset)

    # Register original batch size
    if eval_batch_size is not None:
        dataset.batch_size = batch_size_original

    return per_mean
