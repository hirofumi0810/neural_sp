#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Resolve UNK tokens words from the character-based model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def resolve_unk(str_hyp, best_hyps_sub, aw, aw_sub, idx2char):

    oov_info = []
    # [[idx_oov_0, t_sub_0], ...]

    # Store OOV places
    for idx_oov, w in enumerate(str_hyp.split('_')):
        if w == 'OOV':
            oov_info.append([idx_oov, -1])

    # Point to characters
    for i in range(len(oov_info)):
        max_attention_overlap = 0
        for t_sub in range(len(aw_sub)):
            # print(np.sum(aw[oov_info[i][0]] * aw_sub[t_sub]))
            if np.sum(aw[oov_info[i][0]] * aw_sub[t_sub]) > max_attention_overlap:
                # Check if the correcsponding character is space
                max_char = idx2char(best_hyps_sub[t_sub: t_sub + 1])
                if max_char == '_':
                    continue

                max_attention_overlap = np.sum(
                    aw[oov_info[i][0]] * aw_sub[t_sub])
                oov_info[i][1] = t_sub

    str_hyp_no_unk = ''
    oov_count = 0
    for idx_oov, w in enumerate(str_hyp.split('_')):
        if w == 'OOV':
            t_sub = oov_info[oov_count][1]
            covered_word = idx2char(best_hyps_sub[t_sub: t_sub + 1])

            # Search until space (forward pass)
            forward = 1
            while True:
                if t_sub - forward < 0:
                    break
                elif idx2char(best_hyps_sub[t_sub - forward: t_sub - forward + 1]) != '_':
                    covered_word = idx2char(
                        best_hyps_sub[t_sub - forward: t_sub - forward + 1]) + covered_word
                    forward += 1
                else:
                    break

            # Search until space (backward pass)
            backward = 1
            while True:
                if t_sub + backward > len(best_hyps_sub) - 1:
                    break
                elif idx2char(best_hyps_sub[t_sub + backward: t_sub + backward + 1]) not in ['_', '>']:
                    covered_word += idx2char(
                        best_hyps_sub[t_sub + backward: t_sub + backward + 1])
                    backward += 1
                else:
                    break

            str_hyp_no_unk += '_**' + covered_word + '**'
            oov_count += 1
        else:
            str_hyp_no_unk += '_' + w

    if str_hyp_no_unk[0] == '_':
        str_hyp_no_unk = str_hyp_no_unk[1:]

    return str_hyp_no_unk
