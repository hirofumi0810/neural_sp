#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Resolve UNK tokens words from the character-based model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def resolve_unk(hyp_word, best_hyps_char, aw_word, aw_char, idx2char,
                subsample_factor_word, subsample_factor_char):
    """Revolving UNK.

    Args:
        hyp_word ():
        best_hyps_char ():
        aw_word ():
        aw_char ():
        idx2char ():
        subsample_factor_word (int):
        subsample_factor_char (int):
    Returns:
        hyp_no_unk (str):

    """
    oov_info = []
    # [[id_oov_0, t_sub_0], ...]

    diff_time_resolution = subsample_factor_word // subsample_factor_char

    if diff_time_resolution > 1:
        assert diff_time_resolution == 2
        aw_char1 = aw_char[:, ::diff_time_resolution]
        aw_char1 = aw_char1[:, :aw_word.shape[1]]
        aw_char2 = aw_char[:, 1::diff_time_resolution]
        aw_char2 = aw_char2[:, :aw_word.shape[1]]
        aw_char = (aw_char1 + aw_char2) / 2

    # Store places for <unk>
    for offset, w in enumerate(hyp_word.split(' ')):
        if w == '<unk>':
            oov_info.append([offset, -1])

    # Point to characters
    for i in range(len(oov_info)):
        max_attn_overlap = 0
        for t_char in range(len(aw_char)):
            # print(np.sum(aw_word[oov_info[i][0]] * aw_char[t_char]))
            if np.sum(aw_word[oov_info[i][0]] * aw_char[t_char]) > max_attn_overlap:
                # Check if the correcsponding character is space
                max_char = idx2char(best_hyps_char[t_char: t_char + 1])
                if max_char == ' ':
                    continue

                max_attn_overlap = np.sum(
                    aw_word[oov_info[i][0]] * aw_char[t_char])
                oov_info[i][1] = t_char

    hyp_no_unk = ''
    n_oovs = 0
    for offset, w in enumerate(hyp_word.split(' ')):
        if w == '<unk>':
            t_char = oov_info[n_oovs][1]
            covered_word = idx2char(best_hyps_char[t_char: t_char + 1])

            # Search until space (forward pass)
            fwd = 1
            while True:
                if t_char - fwd < 0:
                    break
                elif idx2char(best_hyps_char[t_char - fwd: t_char - fwd + 1]) not in [' ', '>']:
                    covered_word = idx2char(best_hyps_char[t_char - fwd: t_char - fwd + 1]) + covered_word
                    fwd += 1
                else:
                    break

            # Search until space (backward pass)
            bwd = 1
            while True:
                if t_char + bwd > len(best_hyps_char) - 1:
                    break
                elif idx2char(best_hyps_char[t_char + bwd: t_char + bwd + 1]) not in [' ', '>']:
                    covered_word += idx2char(best_hyps_char[t_char + bwd: t_char + bwd + 1])
                    bwd += 1
                else:
                    break

            if offset == 0:
                # First word in a sentence
                hyp_no_unk += '***' + covered_word + '***'
            else:
                hyp_no_unk += ' ***' + covered_word + '***'
            n_oovs += 1
        else:
            hyp_no_unk += ' ' + w

    if hyp_no_unk[0] == ' ':
        hyp_no_unk = hyp_no_unk[1:]

    return hyp_no_unk
