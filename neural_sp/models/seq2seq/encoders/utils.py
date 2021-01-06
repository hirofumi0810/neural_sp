# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utility functions for encoders."""

import logging
import math
import torch

logger = logging.getLogger(__name__)


def chunkwise(xs, N_l, N_c, N_r, padding=True):
    """Slice input frames chunk by chunk and regard each chunk (with left and
        right contexts) as a single utterance for efficient training of
        latency-controlled bidirectional encoder.

    Args:
        xs (FloatTensor): `[B, T, input_dim]`
        N_l (int): number of frames for left context
        N_c (int): number of frames for current context
        N_r (int): number of frames for right context
    Returns:
        xs (FloatTensor): `[B * n_chunks, N_l + N_c + N_r, input_dim]`
            where n_chunks = ceil(T / N_c)

    """
    bs, xmax, idim = xs.size()
    n_chunks = math.ceil(xmax / N_c) if padding else xmax // (N_l + N_c + N_r)

    xs_tmp = xs.new_zeros(bs, n_chunks, N_l + N_c + N_r, idim)
    if padding:
        # pad the left side in the first chunk and the right side in the last chunk
        xs = torch.cat([xs.new_zeros(bs, N_l, idim),
                        xs,
                        xs.new_zeros(bs, N_r, idim)], dim=1)

    t = N_l
    for chunk_idx in range(n_chunks):
        xs_chunk = xs[:, t - N_l:t + (N_c + N_r)]
        xs_tmp[:, chunk_idx, :xs_chunk.size(1), :] = xs_chunk
        t += N_c
    xs = xs_tmp.view(bs * n_chunks, N_l + N_c + N_r, idim)

    return xs
