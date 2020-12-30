#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for single-head atteniton."""

import importlib
import pytest
import torch


def make_args(**kwargs):
    args = dict(
        kdim=32,
        qdim=32,
        adim=16,
        atype='location',
        sharpening_factor=1,
        sigmoid_smoothing=False,
        conv_out_channels=10,
        conv_kernel_size=201,
        dropout=0.1,
        lookahead=2,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        # attention type
        ({'atype': 'location'}),
        ({'atype': 'add'}),
        ({'atype': 'dot'}),
        ({'atype': 'luong_dot'}),
        ({'atype': 'luong_general'}),
        ({'atype': 'luong_concat'}),
        # others
        ({'sharpening_factor': 2.0}),
        ({'sigmoid_smoothing': True}),
        ({'sigmoid_smoothing': True}),
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

    module = importlib.import_module('neural_sp.models.modules.attention')
    attention = module.AttentionMechanism(**args)
    attention = attention.to(device)

    attention.train()
    aws = None
    for i in range(qlen):
        out = attention(key, value, query[:, i:i + 1], mask=src_mask, aw_prev=aws,
                        mode='parallel', cache=True)
        assert len(out) == 3
        cv, aws, attn_state = out
        assert cv.size() == (batch_size, 1, value.size(2))
        assert aws.size() == (batch_size, 1, 1, klen)
        assert isinstance(attn_state, dict)
