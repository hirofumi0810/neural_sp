#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for time-depth separable convolution (TDS) encoder."""

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
        channels="10_10_14_14_14_18_18_18_18_18",
        kernel_sizes="(21,1)_(21,1)_(21,1)_(21,1)_(21,1)_(21,1)_(21,1)_(21,1)_(21,1)_(21,1)",
        dropout=0.1,
        last_proj_dim=0,
        layer_norm_eps=1e-12,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        # FAIR Interspeech2019 setting
        ({'channels': "10_10_14_14_14_18_18_18_18_18",
          'kernel_sizes': "(21,1)_(21,1)_(21,1)_(21,1)_(21,1)_(21,1)_(21,1)_(21,1)_(21,1)_(21,1)"}),
        # FAIR Interspeech2020 setting
        ({'channels': "15_15_19_19_19_23_23_23_23_27_27_27_27_27",
          'kernel_sizes': "(15,1)_(15,1)_(19,1)_(19,1)_(19,1)_(23,1)_(23,1)_(23,1)_(23,1)_(27,1)_(27,1)_(27,1)_(27,1)_(27,1)"}),
        ({'last_proj_dim': 64}),
        ({'input_dim': 3}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    xmaxs = [40, 45]
    device = "cpu"

    module = importlib.import_module('neural_sp.models.seq2seq.encoders.tds')
    enc = module.TDSEncoder(**args)
    enc = enc.to(device)

    for xmax in xmaxs:
        xs = np.random.randn(batch_size, xmax, args['input_dim'] * args['in_channel']).astype(np.float32)
        xlens = torch.IntTensor([len(x) - i * enc.subsampling_factor for i, x in enumerate(xs)])
        xs = pad_list([np2tensor(x, device).float() for x in xs], 0.)
        enc_out_dict = enc(xs, xlens, task='all')

        assert enc_out_dict['ys']['xs'].size(0) == batch_size
        assert enc_out_dict['ys']['xs'].size(1) == enc_out_dict['ys']['xlens'].max()
