#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

from utils.progressbar import wrap_iterator


def stack_frame(input_list, num_stack, num_skip, progressbar=False):
    """Stack & skip some frames. This implementation is based on
       https://arxiv.org/abs/1507.06947.
           Sak, Haşim, et al.
           "Fast and accurate recurrent neural network acoustic models for speech recognition."
           arXiv preprint arXiv:1507.06947 (2015).
    Args:
        input_list (list): list of input data
        num_stack (int): the number of frames to stack
        num_skip (int): the number of frames to skip
        progressbar (bool, optional): if True, visualize progressbar
    Returns:
        input_list_new (list): list of frame-stacked inputs
    """
    if num_stack == 1 and num_stack == 1:
        return input_list

    if num_stack < num_skip:
        raise ValueError('num_skip must be less than num_stack.')

    batch_size = len(input_list)

    input_list_new = []
    for i_batch in wrap_iterator(range(batch_size), progressbar):

        frame_num, input_size = input_list[i_batch].shape
        frame_num_new = math.ceil(frame_num / num_skip)

        stacked_frames = np.zeros((frame_num_new, input_size * num_stack))
        stack_count = 0  # counter
        stack = []
        for t, frame_t in enumerate(input_list[i_batch]):
            #####################
            # final frame
            #####################
            if t == len(input_list[i_batch]) - 1:
                # Stack the final frame
                stack.append(frame_t)

                while stack_count != int(frame_num_new):
                    # Concatenate stacked frames
                    for i_stack in range(len(stack)):
                        stacked_frames[stack_count][input_size *
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
                        stacked_frames[stack_count][input_size *
                                                    i_stack:input_size * (i_stack + 1)] = stack[i_stack]
                    stack_count += 1

                    # Delete some frames to skip
                    for _ in range(num_skip):
                        stack.pop(0)

        input_list_new.append(stacked_frames)

    return np.array(input_list_new)
