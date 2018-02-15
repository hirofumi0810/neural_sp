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
from utils.evaluation.edit_distance import compute_wer


def do_eval_per(model, dataset, beam_width,
                max_decode_len, eval_batch_size=None,
                progressbar=False):
    """Evaluate trained model by Phone Error Rate.
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
        per_mean (float): An average of PER
        substitution (int): the number of substitution
        insertion (int): the number of insertion
        deletion (int): the number of deletion
    """
    # Reset data counter
    dataset.reset()

    idx2phone = Idx2phone(
        '../metrics/vocab_files/' + dataset.label_type + '.txt')
    map2phone39 = Map2phone39(label_type=dataset.label_type,
                              map_file_path='../metrics/phone2phone.txt')

    per_mean = 0
    substitution, insertion, deletion, = 0, 0, 0
    if progressbar:
        pbar = tqdm(total=len(dataset))  # TODO: fix this
    while True:
        batch, is_new_epoch = dataset.next(batch_size=eval_batch_size)

        # Decode
        best_hyps, perm_idx = model.decode(batch['xs'], batch['x_lens'],
                                           beam_width=beam_width,
                                           max_decode_len=max_decode_len)
        ys = batch['ys'][perm_idx]
        y_lens = batch['y_lens'][perm_idx]

        for b in range(len(batch['xs'])):
            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                phone_ref_list = ys[b][0].split(' ')
                # NOTE: transcript is seperated by space(' ')
            else:
                # Convert from index to phone (-> list of phone strings)
                if model.model_type == 'ctc':
                    phone_ref_list = idx2phone(
                        ys[b][:y_lens[b]]).split(' ')
                elif model.model_type == 'attention':
                    phone_ref_list = idx2phone(
                        ys[b][1:y_lens[b] - 1]).split(' ')
                    # NOTE: Exclude <SOS> and <EOS>

            ##############################
            # Hypothesis
            ##############################
            # Convert from index to phone (-> list of phone strings)
            str_hyp = idx2phone(best_hyps[b])

            if model.model_type == 'attention':
                str_hyp = str_hyp.split('>')[0]
                # NOTE: Trancate by the first <EOS>

                # Remove the last space
                if len(str_hyp) > 0 and str_hyp[-1] == ' ':
                    str_hyp = str_hyp[:-1]

            phone_hyp_list = str_hyp.split(' ')

            # Mapping to 39 phones (-> list of phone strings)
            if dataset.label_type != 'phone39':
                phone_ref_list = map2phone39(phone_ref_list)
                phone_hyp_list = map2phone39(phone_hyp_list)

            # Compute PER
            per_b, sub_b, ins_b, del_b = compute_wer(
                ref=phone_ref_list,
                hyp=phone_hyp_list,
                normalize=True)
            per_mean += per_b
            substitution += sub_b
            insertion += ins_b
            deletion += del_b

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    per_mean /= len(dataset)

    return per_mean, substitution, insertion, deletion
