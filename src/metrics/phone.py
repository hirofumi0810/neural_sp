#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method of phene-level models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import pandas as pd

from src.utils.evaluation.edit_distance import compute_wer
from src.utils.evaluation.normalization import normalize


def eval_phone(models, dataset, eval_batch_size, beam_width,
               max_decode_len, min_decode_len=0, min_decode_len_ratio=0,
               length_penalty=0, coverage_penalty=0,
               progressbar=False):
    """Evaluate a model by PER.
    Args:
        models (list): the models to evaluate
        dataset: An instance of a `Dataset' class
        eval_batch_size (int): the batch size when evaluating the model
        beam_width: (int): the size of beam
        max_decode_len (int): the maximum sequence length of tokens
        min_decode_len (int): the minimum sequence length of tokens
        min_decode_len_ratio (float):
        length_penalty (float): length penalty
        coverage_penalty (float): coverage penalty
        progressbar (bool): if True, visualize the progressbar
    Returns:
        per (float): Phone error rate
        df (pd.DataFrame): dataframe of substitution, insertion, and deletion
    """
    # Reset data counter
    dataset.reset()

    model = models[0]
    # TODO: fix this

    if dataset.corpus == 'timit':
        map2phone39 = Map2phone39(label_type=dataset.label_type,
                                  map_file_path=dataset.phone_map_path)

    per = 0
    sub, ins, dele = 0, 0, 0
    num_phones = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))  # TODO: fix this
    while True:
        batch, is_new_epoch = dataset.next(batch_size=eval_batch_size)

        # Decode
        best_hyps, _, perm_idx = model.decode(
            batch['xs'],
            beam_width=beam_width,
            max_decode_len=max_decode_len,
            min_decode_len=min_decode_len,
            min_decode_len_ratio=min_decode_len_ratio,
            length_penalty=length_penalty,
            coverage_penalty=coverage_penalty)

        ys = [batch['ys'][i] for i in perm_idx]

        for b in range(len(batch['xs'])):
            # Reference
            if dataset.is_test:
                phone_ref_list = ys[b].split('_')
                # NOTE: transcript is seperated by space('_')
            else:
                phone_ref_list = dataset.idx2phone(ys[b]).split('_')

            # Hypothesis
            str_hyp = dataset.idx2phone(best_hyps[b])

            str_hyp = normalize(str_hyp, remove_tokens=['>'])

            phone_hyp_list = str_hyp.split('_')

            if dataset.corpus == 'timit':
                # Mapping to 39 phones (-> list of phone strings)
                if dataset.label_type != 'phone39':
                    phone_ref_list = map2phone39(phone_ref_list)
                    phone_hyp_list = map2phone39(phone_hyp_list)

            # Compute PER
            try:
                per_b, sub_b, ins_b, del_b = compute_wer(ref=phone_ref_list,
                                                         hyp=phone_hyp_list,
                                                         normalize=False)
                per += per_b
                sub += sub_b
                ins += ins_b
                dele += del_b
                num_phones += len(phone_ref_list)
            except:
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

    df = pd.DataFrame({'SUB': [sub], 'INS': [ins], 'DEL': [dele]},
                      columns=['SUB', 'INS', 'DEL'],
                      index=['PER'])

    return per, df


class Map2phone39(object):
    """Map from 61 or 48 phones to 39 phones for TIMIT corpus.
    Args:
        label_type (string): phone39 or phone48 or phone61
        map_file_path (string): path to the mapping file
    """

    def __init__(self, label_type, map_file_path):
        self.label_type = label_type

        # Load the mapping file
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

        self.map_dict['>'] = '>'

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
