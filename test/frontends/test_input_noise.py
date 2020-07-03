#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for input noise injection."""

import numpy as np

from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list
from neural_sp.models.seq2seq.frontends.input_noise import add_input_noise


def test_forward():
    batch_size = 4
    xmax = 40
    input_dim = 80
    device_id = -1

    xs = np.random.randn(batch_size, xmax, input_dim).astype(np.float32)
    xs = pad_list([np2tensor(x, device_id).float() for x in xs], 0.)

    out = add_input_noise(xs, std=0.075)
    assert out.size() == xs.size()
