#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for CNN encoder."""

import importlib
import numpy as np
import pytest
import torch

from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


def make_args_2d(**kwargs):
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
    "args",
    [
        # subsample4
        ({'channels': "32_32", 'kernel_sizes': "(3,3)_(3,3)",
          'strides': "(1,1)_(1,1)", 'poolings': "(2,2)_(2,2)"}),
        ({'channels': "32_32", 'kernel_sizes': "(3,3)_(3,3)",
          'strides': "(1,1)_(1,1)", 'poolings': "(2,2)_(2,1)"}),
        # ({'channels': "32_32", 'kernel_sizes': "(3,3)_(3,3)",
        #   'strides': "(1,1)_(1,1)", 'poolings': "(2,2)_(1,2)"}),
        ({'channels': "32_32", 'kernel_sizes': "(3,3)_(3,3)",
          'strides': "(1,1)_(1,1)", 'poolings': "(1,1)_(1,1)"}),
        # subsample8
        ({'channels': "32_32_32", 'kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'poolings': "(2,2)_(2,2)_(2,2)"}),
        ({'channels': "32_32_32", 'kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'poolings': "(2,2)_(2,2)_(2,1)"}),
        # ({'channels': "32_32_32", 'kernel_sizes': "(3,3)_(3,3)_(3,3)",
        #   'poolings': "(2,2)_(2,2)_(1,2)"}),
        ({'channels': "32_32_32", 'kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'poolings': "(2,2)_(2,1)_(2,1)"}),
        # ({'channels': "32_32_32", 'kernel_sizes': "(3,3)_(3,3)_(3,3)",
        #   'poolings': "(2,2)_(1,2)_(1,2)"}),
        ({'channels': "32_32_32", 'kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'poolings': "(2,2)_(1,1)_(1,1)"}),
        ({'channels': "32_32_32", 'kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'poolings': "(2,1)_(1,1)_(1,1)"}),
        # ({'channels': "32_32_32", 'kernel_sizes': "(3,3)_(3,3)_(3,3)",
        #   'poolings': "(1,2)_(1,1)_(1,1)"}),
        ({'channels': "32_32_32", 'kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'poolings': "(1,1)_(1,1)_(1,1)"}),
        # others
        ({'batch_norm': True}),
        ({'layer_norm': True}),
        ({'residual': True}),
        ({'bottleneck_dim': 8}),
    ]
)
def test_forward_2d(args):
    args = make_args_2d(**args)

    batch_size = 4
    xmaxs = [40, 45]
    device = "cpu"

    module = importlib.import_module('neural_sp.models.seq2seq.encoders.conv')
    (channels, kernel_sizes, strides, poolings), is_1dconv = module.parse_cnn_config(
        args['channels'], args['kernel_sizes'],
        args['strides'], args['poolings'])
    assert not is_1dconv
    enc = module.ConvEncoder(**args)
    enc = enc.to(device)

    for xmax in xmaxs:
        xs = np.random.randn(batch_size, xmax, args['input_dim']).astype(np.float32)
        xlens = torch.IntTensor([len(x) - i * enc.subsampling_factor for i, x in enumerate(xs)])
        xs = pad_list([np2tensor(x, device).float() for x in xs], 0.)
        xs, xlens = enc(xs, xlens)

        assert xs.size(0) == batch_size
        assert xs.size(1) == xlens.max(), (xs.size(), xlens)


def make_args_1d(**kwargs):
    args = dict(
        input_dim=80,
        in_channel=1,
        channels="32_32_32",
        kernel_sizes="3_3_3",
        strides="1_1_1",
        poolings="2_2_2",
        dropout=0.1,
        batch_norm=False,
        layer_norm=True,
        residual=True,
        bottleneck_dim=0,
        param_init=0.1,
        layer_norm_eps=1e-12
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        # subsample4
        ({'channels': "32_32", 'kernel_sizes': "3_3",
          'strides': "1_1", 'poolings': "2_2"}),
        ({'channels': "32_32", 'kernel_sizes': "3_3",
          'strides': "1_1", 'poolings': "2_1"}),
        ({'channels': "32_32", 'kernel_sizes': "3_3",
          'strides': "1_1", 'poolings': "1_1"}),
        # subsample8
        ({'channels': "32_32_32", 'kernel_sizes': "3_3_3",
          'poolings': "2_2_2"}),
        ({'channels': "32_32_32", 'kernel_sizes': "3_3_3",
          'poolings': "2_2_1"}),
        ({'channels': "32_32_32", 'kernel_sizes': "3_3_3",
            'poolings': "2_1_1"}),
        ({'channels': "32_32_32", 'kernel_sizes': "3_3_3",
          'poolings': "1_1_1"}),
        # bottleneck
        ({'bottleneck_dim': 8}),
    ]
)
def test_forward_1d(args):
    args = make_args_1d(**args)

    batch_size = 4
    xmaxs = [40, 45]
    device = "cpu"

    module = importlib.import_module('neural_sp.models.seq2seq.encoders.conv')
    (channels, kernel_sizes, strides, poolings), is_1dconv = module.parse_cnn_config(
        args['channels'], args['kernel_sizes'],
        args['strides'], args['poolings'])
    assert is_1dconv
    enc = module.ConvEncoder(**args)
    enc = enc.to(device)

    for xmax in xmaxs:
        xs = np.random.randn(batch_size, xmax, args['input_dim']).astype(np.float32)
        xlens = torch.IntTensor([len(x) - i * enc.subsampling_factor for i, x in enumerate(xs)])
        xs = pad_list([np2tensor(x, device).float() for x in xs], 0.)

        xs, xlens = enc(xs, xlens)
        assert xs.size(0) == batch_size
        assert xs.size(1) == xlens.max(), (xs.size(), xlens)
