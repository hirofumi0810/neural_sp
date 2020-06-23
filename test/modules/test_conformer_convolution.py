#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for Conformer convolution module."""

import importlib
import pytest
import torch


def make_args(**kwargs):
    args = dict(
        d_model=256,
        kernel_size=32,
        param_init='xavier_init'
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args", [
        ({'kernel_size': 3}),
        ({'kernel_size': 7}),
        ({'kernel_size': 17}),
        ({'kernel_size': 31}),
        ({'kernel_size': 33}),
        ({'kernel_size': 65}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    xmaxs = [40, 45]
    module = importlib.import_module('neural_sp.models.modules.conformer_convolution')
    conv = module.ConformerConvBlock(**args)

    for xmax in xmaxs:
        xs = torch.FloatTensor(batch_size, xmax, args['d_model'])
        xs = conv(xs)

        assert xs.size() == (batch_size, xmax, args['d_model'])
