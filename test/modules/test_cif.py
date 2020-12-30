#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for CIF."""

import importlib
import pytest
import torch


def make_args(**kwargs):
    args = dict(
        enc_dim=32,
        window=3,
        threshold=1.0,
        param_init='',
        layer_norm_eps=1e-12,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        ({'threshold': 1.0}),
        ({'threshold': 0.9}),
        ({'param_init': 'xavier_uniform'}),
    ]
)
def test_forward_parallel(args):
    args = make_args(**args)

    batch_size = 1
    xmax = 40
    ymax = 5
    device = "cpu"

    eouts = torch.FloatTensor(batch_size, xmax, args['enc_dim'], device=device)
    elens = torch.IntTensor([i for i in range(xmax, xmax - batch_size, -1)])
    ylens = torch.IntTensor([i for i in range(ymax, ymax - batch_size, -1)])

    module = importlib.import_module('neural_sp.models.modules.cif')
    cif = module.CIF(**args)
    cif = cif.to(device)
    cif.train()

    out = cif(eouts, elens, ylens, mode='parallel')
    assert len(out) == 3
    cv, aws, attn_state = out
    assert cv.size() == (batch_size, ymax, args['enc_dim'])
    assert aws.size() == (batch_size, ymax, xmax)
    assert isinstance(attn_state, dict)
    alpha = attn_state['alpha']
    assert alpha.size() == (batch_size, xmax)


@pytest.mark.parametrize(
    "args",
    [
        ({'threshold': 1.0}),
        ({'threshold': 0.9}),
    ]
)
def test_forward_incremental(args):
    args = make_args(**args)

    batch_size = 1
    xmax = 40
    ymax = 5
    device = "cpu"

    eouts = torch.FloatTensor(batch_size, xmax, args['enc_dim'], device=device)
    elens = torch.IntTensor([len(x) for x in eouts])

    module = importlib.import_module('neural_sp.models.modules.cif')
    cif = module.CIF(**args)
    cif = cif.to(device)
    cif.eval()

    for i in range(ymax):
        out = cif(eouts, elens, mode='incremental')
        assert len(out) == 3
        cv, aws, attn_state = out
        assert cv.size() == (batch_size, 1, args['enc_dim'])
        assert aws.size() == (batch_size, 1, xmax)
        assert isinstance(attn_state, dict)
        alpha = attn_state['alpha']
        assert alpha.size() == (batch_size, xmax)
