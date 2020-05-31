#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for CNN encoders."""

import importlib
import numpy as np
import pytest
import torch

from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


def make_args(**kwargs):
    args = dict(
        input_dim=80,
        in_channel=1,
        channels="32_32_32",
        kernel_sizes="(3,3)_(3,3)_(3,3)",
        strides="(1,1)_(1,1)_(1,1)",
        poolings="(2,2)_(2,2)_(2,2)",
        dropout=0.1,
        batch_norm=False,
        layer_norm=False,
        residual=False,
        bottleneck_dim=0,
        param_init=0.1,
        layer_norm_eps=1e-12
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args", [
        # subsample4
        ({'channels': "32_32", 'kernel_sizes': "(3,3)_(3,3)",
          'strides': "(1,1)_(1,1)", 'poolings': "(2, 2)_(2, 2)"}),
        ({'channels': "32_32", 'kernel_sizes': "(3,3)_(3,3)",
          'strides': "(1,1)_(1,1)", 'poolings': "(2, 2)_(2, 1)"}),
        # ({'channels': "32_32", 'kernel_sizes': "(3,3)_(3,3)",
        #   'strides': "(1,1)_(1,1)", 'poolings': "(2, 2)_(1, 2)"}),
        ({'channels': "32_32", 'kernel_sizes': "(3,3)_(3,3)",
          'strides': "(1,1)_(1,1)", 'poolings': "(1, 1)_(1, 1)"}),
        # subsample8
        ({'channels': "32_32_32", 'kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'poolings': "(2, 2)_(2, 2)_(2, 2)"}),
        ({'channels': "32_32_32", 'kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'poolings': "(2, 2)_(2, 2)_(2, 1)"}),
        ({'channels': "32_32_32", 'kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'poolings': "(2, 2)_(2, 1)_(2, 1)"}),
        # ({'channels': "32_32_32", 'kernel_sizes': "(3,3)_(3,3)_(3,3)",
        #   'poolings': "(2, 2)_(2, 2)_(1, 2)"}),
        # ({'channels': "32_32_32", 'kernel_sizes': "(3,3)_(3,3)_(3,3)",
        #   'poolings': "(2, 2)_(1, 2)_(1, 2)"}),
        ({'channels': "32_32_32", 'kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'poolings': "(1, 1)_(1, 1)_(1, 1)"}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    xmaxs = [40, 45]
    device_id = -1
    module = importlib.import_module('neural_sp.models.seq2seq.encoders.conv')
    channels, kernel_sizes, strides, poolings = module.parse_config(
        args['channels'], args['kernel_sizes'],
        args['strides'], args['poolings'])
    enc = module.ConvEncoder(**args)
    for xmax in xmaxs:
        xs = np.random.randn(batch_size, xmax, args['input_dim']).astype(np.float32)
        xlens = torch.IntTensor([len(x) for x in xs])
        xs = pad_list([np2tensor(x, device_id).float() for x in xs], 0.)
        xs, xlens = enc(xs, xlens)

        assert xs.size(0) == batch_size
        assert xs.size(1) == xlens[0]
