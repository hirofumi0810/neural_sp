#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Splice data."""

import numpy as np


def do_splice(inputs, splice=1, batch_size=1, num_stack=1):
    """Splice input data. This is expected to be used for CNN-like models.
    Args:
        inputs (np.ndarray): list of size
            `[B, T, input_size (num_channels * 3 * num_stack)]'
        splice (int): frames to splice. Default is 1 frame.
            ex.) if splice == 11
                [t-5, ..., t-1, t, t+1, ..., t+5] (total 11 frames)
        batch_size (int): the size of mini-batch
        num_stack (int, optional): the number of frames to stack
    Returns:
        data_spliced (np.ndarray): A tensor of size
            `[B, T, num_channels * (splice * num_stack) * 3 (static + Δ + ΔΔ)]`
    """
    assert isinstance(inputs, np.ndarray), 'inputs should be np.ndarray.'
    assert len(inputs.shape) == 3, 'inputs must be 3 demension.'
    assert inputs.shape[-1] % 3 == 0

    if splice == 1:
        return inputs

    batch_size, max_time, input_size = inputs.shape
    num_channels = (input_size // 3) // num_stack
    input_data_spliced = np.zeros(
        (batch_size, max_time, num_channels * (splice * num_stack) * 3))

    for i_batch in range(batch_size):
        for i_time in range(max_time):
            spliced_frames = np.zeros((splice * num_stack, num_channels, 3))
            for i_splice in range(0, splice, 1):
                #########################
                # padding left frames
                #########################
                if i_time <= splice - 1 and i_splice < splice - i_time:
                    # copy the first frame to left side
                    copy_frame = inputs[i_batch][0]

                #########################
                # padding right frames
                #########################
                elif max_time - splice <= i_time and i_time + (i_splice - splice) > max_time - 1:
                    # copy the last frame to right side
                    copy_frame = inputs[i_batch][-1]

                #########################
                # middle of frames
                #########################
                else:
                    copy_frame = inputs[i_batch][i_time + (i_splice - splice)]

                # `[num_channels * 3 * num_stack]` -> `[num_channels, 3, num_stack]`
                copy_frame = copy_frame.reshape((num_channels, 3, num_stack))

                # `[num_channels, 3, num_stack]` -> `[num_stack, num_channels, 3]`
                copy_frame = np.transpose(copy_frame, (2, 0, 1))

                spliced_frames[i_splice: i_splice + num_stack] = copy_frame

            # `[splice * num_stack, num_channels, 3] -> `[num_channels, splice * num_stack, 3]`
            spliced_frames = np.transpose(spliced_frames, (1, 0, 2))

            input_data_spliced[i_batch][i_time] = spliced_frames.reshape(
                (num_channels * (splice * num_stack) * 3))

    return input_data_spliced


def test():
    sequence = np.zeros((3, 100, 5))
    for i_batch in range(sequence.shape[0]):
        for i_frame in range(sequence.shape[1]):
            sequence[i_batch][i_frame][0] = i_frame
    sequence_spliced = do_splice(sequence, splice=11)
    assert sequence_spliced.shape == (3, 100, 5 * 11)

    # for i in range(sequence_spliced.shape[1]):
    #     print(sequence_spliced[0][i])


if __name__ == '__main__':
    test()
