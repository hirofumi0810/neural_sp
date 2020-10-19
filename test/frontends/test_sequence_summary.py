#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for sequence summary network."""

import importlib
import numpy as np
import pytest
import torch

from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


def make_args(**kwargs):
    args = dict(
        input_dim=80,
        n_units=64,
        n_layers=2,
        bottleneck_dim=0,
        dropout=0.1,
        param_init=0.1,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        ({'n_layers': 2, 'bottleneck_dim': 0}),
        ({'n_layers': 2, 'bottleneck_dim': 100}),
        ({'n_layers': 3, 'bottleneck_dim': 0}),
        ({'n_layers': 3, 'bottleneck_dim': 100}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    xmax = 40
    device = "cpu"

    xs = np.random.randn(batch_size, xmax, args['input_dim']).astype(np.float32)
    xlens = torch.IntTensor([len(x) for x in xs])
    xs = pad_list([np2tensor(x, device).float() for x in xs], 0.)

    module = importlib.import_module('neural_sp.models.seq2seq.frontends.sequence_summary')
    ssn = module.SequenceSummaryNetwork(**args)
    ssn = ssn.to(device)

    out = ssn(xs, xlens)
    assert out.size() == xs.size()
