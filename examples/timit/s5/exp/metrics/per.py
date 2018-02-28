#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method by Phone Error Rate (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import pandas as pd

from utils.io.labels.phone import Idx2phone
from utils.evaluation.edit_distance import compute_wer


def do_eval_per(model, dataset, beam_width, max_decode_len,
                eval_batch_size=None, progressbar=False):
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
        per (float): Phone error rate
        df_per (pd.DataFrame): dataframe of substitution, insertion, and deletion
    """
    # Reset data counter
    dataset.reset()

    idx2phone = Idx2phone(vocab_file_path=dataset.vocab_file_path)
    map2phone39 = Map2phone39(label_type=dataset.label_type,
                              map_file_path='./conf/phones.60-48-39.map')

    per = 0
    sub, ins, dele = 0, 0, 0
    num_phones = 0
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
                phone_ref_list = idx2phone(ys[b][:y_lens[b]]).split(' ')

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
            try:
                per_b, sub_b, ins_b, del_b = compute_wer(
                    ref=phone_ref_list,
                    hyp=phone_hyp_list,
                    normalize=False)
                per += per_b
                sub += sub_b
                ins += ins_b
                dele += del_b
                num_phones += len(phone_ref_list)
            except:
                # print('REF: %s' % ' '.join(phone_ref_list))
                # print('HYP: %s' % str_hyp)
                pass

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    per /= num_phones
    sub /= num_phones
    ins /= num_phones
    dele /= num_phones

    df_per = pd.DataFrame(
        {'SUB': [sub * 100], 'INS': [ins * 100], 'DEL': [dele * 100]},
        columns=['SUB', 'INS', 'DEL'], index=['PER'])

    return per, df_per


class Map2phone39(object):
    """Map from 61 or 48 phones to 39 phones.
    Args:
        label_type (string): phone39 or phone48 or phone61
        map_file_path (string): path to the mapping file
    """

    def __init__(self, label_type, map_file_path):
        self.label_type = label_type

        # Read the mapping file
        self.map_dict = {}
        with open(map_file_path) as f:
            for line in f:
                line = line.strip().split()
                if label_type == 'phone61':
                    if len(line) >= 2:
                        self.map_dict[line[0]] = line[2]
                    else:
                        self.map_dict[line[0]] = ''
                elif label_type == 'phone48':
                    if len(line) >= 2:
                        self.map_dict[line[1]] = line[2]

    def __call__(self, phone_list):
        """
        Args:
            phone_list (list): list of phones (string)
        Returns:
            phone_list (list): list of 39 phones (string)
        """
        if self.label_type == 'phone39':
            return phone_list

        if len(phone_list) == 1 and phone_list[0] == '':
            return phone_list

        # Map to 39 phones
        for i in range(len(phone_list)):
            phone_list[i] = self.map_dict[phone_list[i]]

        # Ignore q (only if 61 phones)
        while '' in phone_list:
            phone_list.remove('')

        return phone_list
