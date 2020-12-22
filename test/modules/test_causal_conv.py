#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for dilated causal convolution module."""

import importlib
import pytest
import torch
import warnings


def make_args(**kwargs):
    args = dict(
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        dilation=1,
        param_init=''
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        ({'kernel_size': 3}),
        ({'kernel_size': 5}),
        ({'kernel_size': 7}),
        ({'kernel_size': 15}),
        ({'kernel_size': 31}),
        ({'param_init': 'lecun'}),
        ({'param_init': 'xavier_uniform'}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    max_len = 40
    device = "cpu"

    xs = torch.FloatTensor(batch_size, max_len, args['in_channels'], device=device)

    module = importlib.import_module('neural_sp.models.modules.causal_conv')
    conv1d = module.CausalConv1d(**args)
    conv1d = conv1d.to(device)

    out = conv1d(xs)
    assert out.size() == (batch_size, max_len, args['out_channels'])

    # incremental check
    out_incremental = []
    for t in range(max_len):
        out_incremental.append(conv1d(xs[:, :t + 1])[:, -1:])
    out_incremental = torch.cat(out_incremental, dim=1)
    assert out.size() == out_incremental.size()
    if not torch.allclose(out[:, :t + 1], out_incremental, equal_nan=True):
        warnings.warn("Incremental output did not match.", UserWarning)
