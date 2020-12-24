#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for SpecAugment."""

import importlib
import numpy as np
import pytest
import torch

from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


def make_args(**kwargs):
    args = dict(
        F=27,
        T=100,
        n_freq_masks=2,
        n_time_masks=2,
        p=1.0,
        W=40,
        adaptive_number_ratio=0,
        adaptive_size_ratio=0,
        max_n_time_masks=20,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        ({'F': 13, 'T': 50}),
        ({'F': 27, 'T': 50}),
        ({'F': 27, 'T': 100}),
        ({'F': 27, 'T': 100, 'p': 0.2}),
        ({'n_freq_masks': 1, 'n_time_masks': 1}),
        ({'n_freq_masks': 3, 'n_time_masks': 3}),
        ({'adaptive_number_ratio': 0.04, 'adaptive_size_ratio': 0.04}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    xmax = 400
    input_dim = 80
    device = "cpu"

    xs = np.random.randn(batch_size, xmax, input_dim).astype(np.float32)
    xs = pad_list([np2tensor(x, device).float() for x in xs], 0.)

    module = importlib.import_module('neural_sp.models.seq2seq.frontends.spec_augment')
    specaug = module.SpecAugment(**args)

    out = specaug(xs)
    assert out.size() == xs.size()


@pytest.mark.parametrize(
    "args",
    [
        ({}),
    ]
)
def test_fixed_config_forward(args):
    args = make_args(**args)

    batch_size = 4
    xmax = 400
    input_dim = 80
    device = "cpu"

    xs = np.random.randn(batch_size, xmax, input_dim).astype(np.float32)
    xs = pad_list([np2tensor(x, device).float() for x in xs], 0.)

    module = importlib.import_module('neural_sp.models.seq2seq.frontends.spec_augment')
    specaug = module.SpecAugment(**args)

    # fixed setting
    specaug.librispeech_basic()
    out = specaug(xs)
    assert out.size() == xs.size()

    specaug.librispeech_double()
    out = specaug(xs)
    assert out.size() == xs.size()

    specaug.switchboard_mild()
    out = specaug(xs)
    assert out.size() == xs.size()

    specaug.switchboard_strong()
    out = specaug(xs)
    assert out.size() == xs.size()
