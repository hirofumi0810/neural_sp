#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Frame stacking."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def stack_frame(feat, nstacks, nskips, dtype=np.float32):
    """Stack & skip some frames. This implementation is based on

       https://arxiv.org/abs/1507.06947.
           Sak, Haşim, et al.
           "Fast and accurate recurrent neural network acoustic models for speech recognition."
           arXiv preprint arXiv:1507.06947 (2015).

    Args:
        feat (list): `[T, input_dim]`
        nstacks (int): the number of frames to stack
        nskips (int): the number of frames to skip
        dtype ():
    Returns:
        stacked_feat (np.ndarray): `[floor(T / nskips), input_dim * nstacks]`

    """
    if nstacks == 1 and nstacks == 1:
        return feat

    if nstacks < nskips:
        raise ValueError('nskips must be less than nstacks.')

    frame_num, input_dim = feat.shape
    frame_num_new = (frame_num + 1) // nskips

    stacked_feat = np.zeros((frame_num_new, input_dim * nstacks), dtype=dtype)
    stack_count = 0
    stack = []
    for t, frame_t in enumerate(feat):

        # final frame
        if t == len(feat) - 1:
            # Stack the final frame
            stack.append(frame_t)

            while stack_count != int(frame_num_new):
                # Concatenate stacked frames
                for i in range(len(stack)):
                    stacked_feat[stack_count][input_dim *
                                              i:input_dim * (i + 1)] = stack[i]
                stack_count += 1

                # Delete some frames to skip
                for _ in range(nskips):
                    if len(stack) != 0:
                        stack.pop(0)

        # first & middle frames
        elif len(stack) < nstacks:
            # Stack some frames until stack is filled
            stack.append(frame_t)

            if len(stack) == nstacks:
                    # Concatenate stacked frames
                for i in range(nstacks):
                    stacked_feat[stack_count][input_dim *
                                              i:input_dim * (i + 1)] = stack[i]
                stack_count += 1

                # Delete some frames to skip
                for _ in range(nskips):
                    stack.pop(0)

    return stacked_feat
