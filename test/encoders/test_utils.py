#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for encoder utility functions."""

import importlib
import numpy as np
import pytest
import torch

from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


@pytest.mark.parametrize(
    "N_l, N_c, N_r",
    [
        (96, 64, 32),
        (64, 64, 64),
        (40, 40, 40),
        (40, 40, 20),
    ]
)
def test_chunkwise(N_l, N_c, N_r):
    batch_size = 4
    xmaxs = [800, 855]
    input_dim = 80
    device = "cpu"

    module = importlib.import_module('neural_sp.models.seq2seq.encoders.utils')

    for xmax in xmaxs:
        xs = np.random.randn(batch_size, xmax, input_dim).astype(np.float32)
        xs = pad_list([np2tensor(x, device).float() for x in xs], 0.)

        xs_chunk = module.chunkwise(xs, N_l, N_c, N_r)

        # Extract the center region
        xs_chunk = xs_chunk[:, N_l:N_l + N_c]  # `[B * n_chunks, N_c, input_dim]`
        xs_chunk = xs_chunk.contiguous().view(batch_size, -1, xs_chunk.size(2))
        xs_chunk = xs_chunk[:, :xmax]

        assert xs_chunk.size() == xs.size()
        assert torch.equal(xs_chunk, xs)
