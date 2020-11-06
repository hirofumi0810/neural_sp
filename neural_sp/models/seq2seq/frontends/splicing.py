# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Splice data."""

import numpy as np


def splice(feat, n_splices=1, n_stacks=1, dtype=np.float32):
    """Splice input data. This is expected to be used for CNN-like encoder.

    Args:
        feat (np.ndarray): A tensor of size
            `[T, input_dim (freq * 3 * n_stacks)]'
        n_splices (int): frames to n_splices. Default is 1 frame.
            ex.) if n_splices == 11
                [t-5, ..., t-1, t, t+1, ..., t+5] (total 11 frames)
        n_stacks (int): the number of frames to stack
        dtype ():
    Returns:
        feat_splice (np.ndarray): A tensor of size
            `[T, freq * (n_splices * n_stacks) * 3 (static + Δ + ΔΔ)]`

    """
    assert isinstance(feat, np.ndarray), 'feat should be np.ndarray.'
    assert len(feat.shape) == 2, 'feat must be 2 dimension.'
    assert feat.shape[-1] % 3 == 0

    if n_splices == 1:
        return feat

    max_xlen, input_dim = feat.shape
    freq = (input_dim // 3) // n_stacks
    feat_splice = np.zeros((max_xlen, freq * (n_splices * n_stacks) * 3), dtype=dtype)

    for i_time in range(max_xlen):
        spliced_frames = np.zeros((n_splices * n_stacks, freq, 3))
        for i_splice in range(0, n_splices, 1):
            if i_time <= n_splices - 1 and i_splice < n_splices - i_time:
                # copy the first frame to left side (padding left frames)
                copy_frame = feat[0]
            elif max_xlen - n_splices <= i_time and i_time + (i_splice - n_splices) > max_xlen - 1:
                # copy the last frame to right side (padding right frames)
                copy_frame = feat[-1]
            else:
                copy_frame = feat[i_time + (i_splice - n_splices)]

            # `[freq * 3 * n_stacks]` -> `[freq, 3, n_stacks]`
            copy_frame = copy_frame.reshape((freq, 3, n_stacks))

            # `[freq, 3, n_stacks]` -> `[n_stacks, freq, 3]`
            copy_frame = np.transpose(copy_frame, (2, 0, 1))

            spliced_frames[i_splice: i_splice + n_stacks] = copy_frame

        # `[n_splices * n_stacks, freq, 3] -> `[freq, n_splices * n_stacks, 3]`
        spliced_frames = np.transpose(spliced_frames, (1, 0, 2))

        feat_splice[i_time] = spliced_frames.reshape((freq * (n_splices * n_stacks) * 3))

    return feat_splice
