# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Splice data."""

import numpy as np


def splice(x, n_splices=1, n_stacks=1, dtype=np.float32):
    """Splice input data. This is expected to be used for CNN-like encoder.

    Args:
        x (np.ndarray): A tensor of size
            `[T, input_dim (F * 3 * n_stacks)]'
        n_splices (int): frames to n_splices. Default is 1 frame.
            ex.) if n_splices == 11
                [t-5, ..., t-1, t, t+1, ..., t+5] (total 11 frames)
        n_stacks (int): the number of stacked frames in frame stacking
        dtype ():
    Returns:
        feat_splice (np.ndarray): A tensor of size
            `[T, F * (n_splices * n_stacks) * 3 (static + Δ + ΔΔ)]`

    """
    if n_splices == 1:
        return x
    assert isinstance(x, np.ndarray), 'x should be np.ndarray.'
    assert len(x.shape) == 2, 'x must be 2 dimension.'
    is_delta = ((x.shape[-1] // n_stacks) % 3 == 0)
    n_delta = 3 if is_delta else 1

    print(x.shape)
    T, input_dim = x.shape
    F = (input_dim // n_delta) // n_stacks
    feat_splice = np.zeros((T, F * (n_splices * n_stacks) * n_delta), dtype=dtype)

    for i_time in range(T):
        spliced_frames = np.zeros((n_splices * n_stacks, F, n_delta))
        for i_splice in range(0, n_splices, 1):
            if i_time <= n_splices - 1 and i_splice < n_splices - i_time:
                # copy the first frame to left side (padding left frames)
                copy_frame = x[0]
            elif T - n_splices <= i_time and i_time + (i_splice - n_splices) > T - 1:
                # copy the last frame to right side (padding right frames)
                copy_frame = x[-1]
            else:
                copy_frame = x[i_time + (i_splice - n_splices)]

            # `[F * n_delta * n_stacks]` -> `[F, n_delta, n_stacks]`
            copy_frame = copy_frame.reshape((F, n_delta, n_stacks))

            # `[F, n_delta, n_stacks]` -> `[n_stacks, F, n_delta]`
            copy_frame = np.transpose(copy_frame, (2, 0, 1))

            spliced_frames[i_splice: i_splice + n_stacks] = copy_frame

        # `[n_splices * n_stacks, F, n_delta] -> `[F, n_splices * n_stacks, n_delta]`
        spliced_frames = np.transpose(spliced_frames, (1, 0, 2))

        feat_splice[i_time] = spliced_frames.reshape((F * (n_splices * n_stacks) * n_delta))

    return feat_splice
