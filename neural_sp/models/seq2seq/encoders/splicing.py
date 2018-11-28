#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Splice data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def splice(feat, nsplices=1, nstacks=1, dtype=np.float32):
    """Splice input data. This is expected to be used for CNN-like encoder.

    Args:
        feat (np.ndarray): A tensor of size
            `[T, input_dim (freq * 3 * nstacks)]'
        nsplices (int): frames to nsplices. Default is 1 frame.
            ex.) if nsplices == 11
                [t-5, ..., t-1, t, t+1, ..., t+5] (total 11 frames)
        nstacks (int): the number of frames to stack
        dtype ():
    Returns:
        feat_splice (np.ndarray): A tensor of size
            `[T, freq * (nsplices * nstacks) * 3 (static + Δ + ΔΔ)]`

    """
    assert isinstance(feat, np.ndarray), 'feat should be np.ndarray.'
    assert len(feat.shape) == 2, 'feat must be 2 demension.'
    assert feat.shape[-1] % 3 == 0

    if nsplices == 1:
        return feat

    max_time, input_dim = feat.shape
    freq = (input_dim // 3) // nstacks
    feat_splice = np.zeros((max_time, freq * (nsplices * nstacks) * 3), dtype=dtype)

    for i_time in range(max_time):
        spliced_frames = np.zeros((nsplices * nstacks, freq, 3))
        for i_splice in range(0, nsplices, 1):
            if i_time <= nsplices - 1 and i_splice < nsplices - i_time:
                # copy the first frame to left side (padding left frames)
                copy_frame = feat[0]
            elif max_time - nsplices <= i_time and i_time + (i_splice - nsplices) > max_time - 1:
                # copy the last frame to right side (padding right frames)
                copy_frame = feat[-1]
            else:
                copy_frame = feat[i_time + (i_splice - nsplices)]

            # `[freq * 3 * nstacks]` -> `[freq, 3, nstacks]`
            copy_frame = copy_frame.reshape((freq, 3, nstacks))

            # `[freq, 3, nstacks]` -> `[nstacks, freq, 3]`
            copy_frame = np.transpose(copy_frame, (2, 0, 1))

            spliced_frames[i_splice: i_splice + nstacks] = copy_frame

        # `[nsplices * nstacks, freq, 3] -> `[freq, nsplices * nstacks, 3]`
        spliced_frames = np.transpose(spliced_frames, (1, 0, 2))

        feat_splice[i_time] = spliced_frames.reshape((freq * (nsplices * nstacks) * 3))

    return feat_splice
