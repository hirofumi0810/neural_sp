#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method by Phone Error Rate (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import logging
logger = logging.getLogger('training')

from examples.timit.metrics.mapping import Map2phone39
from utils.io.labels.phone import Idx2phone
from utils.evaluation.edit_distance import compute_per


def do_eval_per(model, model_type, dataset, label_type, beam_width,
                max_decode_len, eval_batch_size=None,
                progressbar=False):
    """Evaluate trained model by Phone Error Rate.
    Args:
        model: the model to evaluate
        model_type (string): ctc or attention or joint_ctc_attention
        dataset: An instance of a `Dataset' class
        label_type (string): phone39 or phone48 or phone61
        beam_width: (int): the size of beam
        max_decode_len (int): the length of output sequences
            to stop prediction when EOS token have not been emitted.
            This is used for seq2seq models.
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
    Returns:
        per_mean (float): An average of PER
    """
    # Reset data counter
    dataset.reset()

    hyp_label_type = label_type
    ref_label_type = dataset.label_type

    idx2phone_hyp = Idx2phone(
        '../metrics/vocab_files/' + hyp_label_type + '.txt')
    idx2phone_ref = Idx2phone(
        '../metrics/vocab_files/' + ref_label_type + '.txt')
    map2phone39_hyp = Map2phone39(label_type=hyp_label_type,
                                  map_file_path='../metrics/phone2phone.txt')
    map2phone39_ref = Map2phone39(label_type=ref_label_type,
                                  map_file_path='../metrics/phone2phone.txt')

    per_mean = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))  # TODO: fix this
    while True:
        batch, is_new_epoch = dataset.next(batch_size=eval_batch_size)

        # Decode
        labels_hyp = model.decode(batch['xs'], batch['x_lens'],
                                  beam_width=beam_width,
                                  max_decode_len=max_decode_len)

        for i_batch in range(len(batch['xs'])):
            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                phone_ref_list = batch['ys'][i_batch][0].split(' ')
                # NOTE: transcript is seperated by space(' ')
            else:
                # Convert from index to phone (-> list of phone strings)
                if model_type == 'ctc':
                    phone_ref_list = idx2phone_ref(
                        batch['ys'][i_batch][:batch['y_lens'][i_batch]]).split(' ')
                elif model_type == 'attention':
                    phone_ref_list = idx2phone_ref(
                        batch['ys'][i_batch][1:batch['y_lens'][i_batch] - 1]).split(' ')
                    # NOTE: Exclude <SOS> and <EOS>

            ##############################
            # Hypothesis
            ##############################
            # Convert from index to phone (-> list of phone strings)
            str_hyp = idx2phone_hyp(labels_hyp[i_batch])

            if model_type == 'attention':
                str_hyp = str_hyp.split('>')[0]
                # NOTE: Trancate by the first <EOS>

                # Remove the last space
                if len(str_hyp) > 0 and str_hyp[-1] == ' ':
                    str_hyp = str_hyp[:-1]

            phone_hyp_list = str_hyp.split(' ')

            # Mapping to 39 phones (-> list of phone strings)
            if ref_label_type != 'phone39':
                phone_ref_list = map2phone39_ref(phone_ref_list)
            if hyp_label_type != 'phone39':
                phone_hyp_list = map2phone39_hyp(phone_hyp_list)

            # Compute PER
            per_mean += compute_per(ref=phone_ref_list,
                                    hyp=phone_hyp_list,
                                    normalize=True)

            if progressbar:
                pbar.update(len(batch['xs']))

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    per_mean /= len(dataset)

    return per_mean
