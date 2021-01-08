#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for Conformer convolution module."""

import importlib
import pytest
import torch
import warnings

torch.manual_seed(0)


def make_args(**kwargs):
    args = dict(
        d_model=256,
        kernel_size=3,
        param_init='',
        causal=False,
        normalization='batch_norm',
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        ({'kernel_size': 3}),
        ({'kernel_size': 7}),
        ({'kernel_size': 17}),
        ({'kernel_size': 31}),
        ({'kernel_size': 33}),
        ({'kernel_size': 65}),
        ({'param_init': 'xavier_uniform'}),
        ({'param_init': 'lecun'}),
        ({'kernel_size': 7, 'causal': True}),
        ({'normalization': 'group_norm'}),
        ({'normalization': 'layer_norm'}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    xmaxs = [40, 45]
    device = "cpu"

    module = importlib.import_module('neural_sp.models.modules.conformer_convolution')
    conv = module.ConformerConvBlock(**args)
    conv = conv.to(device)

    for xmax in xmaxs:
        xs = torch.randn(batch_size, xmax, args['d_model'], device=device)
        out = conv(xs)
        assert out.size() == (batch_size, xmax, args['d_model'])

        # incremental check
        if args['causal']:
            out_incremental = []
            for t in range(xmax):
                out_incremental.append(conv(xs[:, :t + 1])[:, -1:])
            out_incremental = torch.cat(out_incremental, dim=1)
            assert out.size() == out_incremental.size()
            if not torch.allclose(out, out_incremental, equal_nan=True):
                warnings.warn("Incremental output did not match.", UserWarning)
