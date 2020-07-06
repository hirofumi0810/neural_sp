#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for positionwise fully-connected feed-forward neural network (FFN)."""

import importlib
import pytest
import torch


def make_args(**kwargs):
    args = dict(
        d_model=32,
        d_ff=128,
        dropout=0.1,
        activation='relu',
        param_init='',
        bottleneck_dim=0,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        # activation
        ({'activation': 'relu'}),
        ({'activation': 'gelu'}),
        ({'activation': 'gelu_accurate'}),
        ({'activation': 'glu'}),
        ({'activation': 'swish'}),
        # initialization
        ({'param_init': 'xavier_uniform'}),
        # bottleneck
        ({'bottleneck_dim': 16}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    max_len = 40
    device = "cpu"

    ffn_in = torch.FloatTensor(batch_size, max_len, args['d_model'], device=device)

    module = importlib.import_module('neural_sp.models.modules.positionwise_feed_forward')
    ffn = module.PositionwiseFeedForward(**args)
    ffn = ffn.to(device)

    ffn_out = ffn(ffn_in)
    assert ffn_in.size() == ffn_out.size()
