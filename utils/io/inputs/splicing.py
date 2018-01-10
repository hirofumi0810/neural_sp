#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Splice data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def do_splice(inputs, splice=1, num_stack=1, dtype=np.float32):
    """Splice input data. This is expected to be used for CNN-like models.
    Args:
        inputs (np.ndarray): A tensor of size
            `[T, input_size (freq * 3 * num_stack)]'
        splice (int): frames to splice. Default is 1 frame.
            ex.) if splice == 11
                [t-5, ..., t-1, t, t+1, ..., t+5] (total 11 frames)
        num_stack (int, optional): the number of frames to stack
        dtype (, optional):
    Returns:
        data_spliced (np.ndarray): A tensor of size
            `[T, freq * (splice * num_stack) * 3 (static + Δ + ΔΔ)]`
    """
    assert isinstance(inputs, np.ndarray), 'inputs should be np.ndarray.'
    assert len(inputs.shape) == 2, 'inputs must be 2 demension.'
    assert inputs.shape[-1] % 3 == 0

    if splice == 1:
        return inputs

    max_time, input_size = inputs.shape
    freq = (input_size // 3) // num_stack
    input_data_spliced = np.zeros(
        (max_time, freq * (splice * num_stack) * 3), dtype=dtype)

    for i_time in range(max_time):
        spliced_frames = np.zeros((splice * num_stack, freq, 3))
        for i_splice in range(0, splice, 1):
            #########################
            # padding left frames
            #########################
            if i_time <= splice - 1 and i_splice < splice - i_time:
                # copy the first frame to left side
                copy_frame = inputs[0]

            #########################
            # padding right frames
            #########################
            elif max_time - splice <= i_time and i_time + (i_splice - splice) > max_time - 1:
                # copy the last frame to right side
                copy_frame = inputs[-1]

            #########################
            # middle of frames
            #########################
            else:
                copy_frame = inputs[i_time + (i_splice - splice)]

            # `[freq * 3 * num_stack]` -> `[freq, 3, num_stack]`
            copy_frame = copy_frame.reshape((freq, 3, num_stack))

            # `[freq, 3, num_stack]` -> `[num_stack, freq, 3]`
            copy_frame = np.transpose(copy_frame, (2, 0, 1))

            spliced_frames[i_splice: i_splice + num_stack] = copy_frame

        # `[splice * num_stack, freq, 3] -> `[freq, splice * num_stack, 3]`
        spliced_frames = np.transpose(spliced_frames, (1, 0, 2))

        input_data_spliced[i_time] = spliced_frames.reshape(
            (freq * (splice * num_stack) * 3))

    return input_data_spliced
