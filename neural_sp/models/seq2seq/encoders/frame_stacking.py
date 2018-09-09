#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def stack_frame(feat, num_stack, num_skip, dtype=np.float32):
    """Stack & skip some frames. This implementation is based on

       https://arxiv.org/abs/1507.06947.
           Sak, Haşim, et al.
           "Fast and accurate recurrent neural network acoustic models for speech recognition."
           arXiv preprint arXiv:1507.06947 (2015).

    Args:
        feat (list): A tensor of size `[T, input_dim]`
        num_stack (int): the number of frames to stack
        num_skip (int): the number of frames to skip
        dtype (, optional):
    Returns:
        stacked_feat (np.ndarray): A tensor of size
            `[floor(T / num_skip), input_dim * num_stack]`

    """
    if num_stack == 1 and num_stack == 1:
        return feat

    if num_stack < num_skip:
        raise ValueError('num_skip must be less than num_stack.')

    frame_num, input_dim = feat.shape
    frame_num_new = (frame_num + 1) // num_skip

    stacked_feat = np.zeros((frame_num_new, input_dim * num_stack), dtype=dtype)
    stack_count = 0  # counter
    stack = []
    for t, frame_t in enumerate(feat):

        # final frame
        if t == len(feat) - 1:
            # Stack the final frame
            stack.append(frame_t)

            while stack_count != int(frame_num_new):
                # Concatenate stacked frames
                for i_stack in range(len(stack)):
                    stacked_feat[stack_count][input_dim *
                                              i_stack:input_dim * (i_stack + 1)] = stack[i_stack]
                stack_count += 1

                # Delete some frames to skip
                for _ in range(num_skip):
                    if len(stack) != 0:
                        stack.pop(0)

        # first & middle frames
        elif len(stack) < num_stack:
            # Stack some frames until stack is filled
            stack.append(frame_t)

            if len(stack) == num_stack:
                # Concatenate stacked frames
                for i_stack in range(num_stack):
                    stacked_feat[stack_count][input_dim *
                                              i_stack:input_dim * (i_stack + 1)] = stack[i_stack]
                stack_count += 1

                # Delete some frames to skip
                for _ in range(num_skip):
                    stack.pop(0)

    return stacked_feat
