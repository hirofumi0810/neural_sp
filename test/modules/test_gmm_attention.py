#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for GMM atteniton."""

import importlib
import pytest
import torch


def make_args(**kwargs):
    args = dict(
        kdim=32,
        qdim=32,
        adim=16,
        n_mixtures=5,
        dropout=0.1,
        param_init='',
        nonlinear='exp',
        vfloor=1e-6,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        ({'n_mixtures': 1}),
        ({'n_mixtures': 4}),
        ({'param_init': 'xavier_uniform'}),
        ({'nonlinear': 'softplus'}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    klen = 40
    qlen = 5
    device = "cpu"

    key = torch.FloatTensor(batch_size, klen, args['kdim'], device=device)
    value = torch.FloatTensor(batch_size, klen, args['kdim'], device=device)
    query = torch.FloatTensor(batch_size, qlen, args['qdim'], device=device)
    src_mask = torch.ones(batch_size, 1, klen, device=device).byte()

    module = importlib.import_module('neural_sp.models.modules.gmm_attention')
    attention = module.GMMAttention(**args)
    attention = attention.to(device)

    attention.train()
    myu = None
    for i in range(qlen):
        out = attention(key, value, query[:, i:i + 1], mask=src_mask, aw_prev=myu,
                        mode='parallel', cache=True)
        assert len(out) == 3
        cv, aws, attn_state = out
        assert cv.size() == (batch_size, 1, value.size(2))
        assert aws.size() == (batch_size, 1, 1, klen)
        assert isinstance(attn_state, dict)
        myu = attn_state['myu']
        assert myu.size() == (batch_size, 1, args['n_mixtures'])
