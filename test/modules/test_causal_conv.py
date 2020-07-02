#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for dilated causal convolution."""

import importlib
import pytest
import torch


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
        ({'param_init': 'xavier_uniform'}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    max_len = 40
    xs = torch.FloatTensor(batch_size, max_len, args['in_channels'])

    module = importlib.import_module('neural_sp.models.modules.causal_conv')
    conv1d = module.CausalConv1d(**args)

    out = conv1d(xs)
    assert out.size() == (batch_size, max_len, args['out_channels'])
