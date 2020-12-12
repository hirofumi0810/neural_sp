#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for frame stacking."""

import importlib
import math
import numpy as np
import pytest

from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


def make_args(**kwargs):
    args = dict(
        n_stacks=1,
        n_skips=1,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        ({'n_stacks': 1, 'n_skips': 1}),
        ({'n_stacks': 2, 'n_skips': 2}),
        ({'n_stacks': 3, 'n_skips': 3}),
        ({'n_stacks': 3, 'n_skips': 1}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    xmax = 40
    input_dim = 80
    device = "cpu"

    xs = [np.random.randn(xlen, input_dim).astype(np.float32)
          for xlen in range(xmax - batch_size, xmax)]
    xs_pad = pad_list([np2tensor(x, device).float() for x in xs], 0.)

    module = importlib.import_module('neural_sp.models.seq2seq.frontends.frame_stacking')

    out = [module.stack_frame(x, args['n_stacks'], args['n_skips'])
           for x in xs]
    out_pad = pad_list([np2tensor(x, device).float() for x in out], 0.)
    assert out_pad.size(0) == xs_pad.size(0)
    assert out_pad.size(1) == math.ceil(xs_pad.size(1) / args['n_skips'])
    assert out_pad.size(2) == xs_pad.size(2) * args['n_stacks']
