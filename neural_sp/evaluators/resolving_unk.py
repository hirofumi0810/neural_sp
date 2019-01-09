#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Resolve UNK tokens words from the character-based model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def resolve_unk(hyp, best_hyps_sub, aw, aw_sub, id2char, diff_time_resolution=1):
    """
    Args:
        hyp:
        best_hyps_sub:
        aw:
        aw_sub:
        id2char:
        diff_time_resolution:

    """
    oov_info = []
    # [[id_oov_0, t_sub_0], ...]

    if diff_time_resolution > 1:
        assert diff_time_resolution == 2
        aw_sub_1 = aw_sub[:, ::diff_time_resolution]
        aw_sub_1 = aw_sub_1[:, :aw.shape[1]]
        aw_sub_2 = aw_sub[:, 1::diff_time_resolution]
        aw_sub_2 = aw_sub_2[:, :aw.shape[1]]
        aw_sub = (aw_sub_1 + aw_sub_2) / 2

    # Store places for <unk>
    for id_oov, w in enumerate(hyp.split('_')):
        if w == '<unk>':
            oov_info.append([id_oov, -1])

    # Point to characters
    for i in range(len(oov_info)):
        max_attn_overlap = 0
        for t_sub in range(len(aw_sub)):
            # print(np.sum(aw[oov_info[i][0]] * aw_sub[t_sub]))
            if np.sum(aw[oov_info[i][0]] * aw_sub[t_sub]) > max_attn_overlap:
                # Check if the correcsponding character is space
                max_char = id2char(best_hyps_sub[t_sub: t_sub + 1])
                if max_char == '_':
                    continue

                max_attn_overlap = np.sum(
                    aw[oov_info[i][0]] * aw_sub[t_sub])
                oov_info[i][1] = t_sub

    hyp_no_unk = ''
    oov_count = 0
    for id_oov, w in enumerate(hyp.split('_')):
        if w == '<unk>':
            t_sub = oov_info[oov_count][1]
            covered_word = id2char(best_hyps_sub[t_sub: t_sub + 1])

            # Search until space (forward pass)
            fwd = 1
            while True:
                if t_sub - fwd < 0:
                    break
                elif id2char(best_hyps_sub[t_sub - fwd: t_sub - fwd + 1]) not in ['_', '>']:
                    covered_word = id2char(best_hyps_sub[t_sub - fwd: t_sub - fwd + 1]) + covered_word
                    fwd += 1
                else:
                    break

            # Search until space (backward pass)
            bwd = 1
            while True:
                if t_sub + bwd > len(best_hyps_sub) - 1:
                    break
                elif id2char(best_hyps_sub[t_sub + bwd: t_sub + bwd + 1]) not in ['_', '>']:
                    covered_word += id2char(best_hyps_sub[t_sub + bwd: t_sub + bwd + 1])
                    bwd += 1
                else:
                    break

            hyp_no_unk += '_**' + covered_word + '**'
            oov_count += 1
        else:
            hyp_no_unk += '_' + w

    if hyp_no_unk[0] == '_':
        hyp_no_unk = hyp_no_unk[1:]

    return hyp_no_unk
