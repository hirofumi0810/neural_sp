#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math


def stack_frame(inputs, num_stack, num_skip):
    """Stack & skip some frames. This implementation is based on
       https://arxiv.org/abs/1507.06947.
           Sak, Haşim, et al.
           "Fast and accurate recurrent neural network acoustic models for speech recognition."
           arXiv preprint arXiv:1507.06947 (2015).
    Args:
        inputs (list): A tensor of size `[T, input_size]`
        num_stack (int): the number of frames to stack
        num_skip (int): the number of frames to skip
    Returns:
        stacked_inputs (np.ndarray): A tensor of size
            `[T / num_skip, input_size * num_stack]`
    """
    if num_stack == 1 and num_stack == 1:
        return inputs

    if num_stack < num_skip:
        raise ValueError('num_skip must be less than num_stack.')

    frame_num, input_size = inputs.shape
    frame_num_new = math.ceil(frame_num / num_skip)

    stacked_inputs = np.zeros((frame_num_new, input_size * num_stack))
    stack_count = 0  # counter
    stack = []
    for t, frame_t in enumerate(inputs):
        #####################
        # final frame
        #####################
        if t == len(inputs) - 1:
            # Stack the final frame
            stack.append(frame_t)

            while stack_count != int(frame_num_new):
                # Concatenate stacked frames
                for i_stack in range(len(stack)):
                    stacked_inputs[stack_count][input_size *
                                                i_stack:input_size * (i_stack + 1)] = stack[i_stack]
                stack_count += 1

                # Delete some frames to skip
                for _ in range(num_skip):
                    if len(stack) != 0:
                        stack.pop(0)

        ########################
        # first & middle frames
        ########################
        elif len(stack) < num_stack:
            # Stack some frames until stack is filled
            stack.append(frame_t)

            if len(stack) == num_stack:
                # Concatenate stacked frames
                for i_stack in range(num_stack):
                    stacked_inputs[stack_count][input_size *
                                                i_stack:input_size * (i_stack + 1)] = stack[i_stack]
                stack_count += 1

                # Delete some frames to skip
                for _ in range(num_skip):
                    stack.pop(0)

    return stacked_inputs
