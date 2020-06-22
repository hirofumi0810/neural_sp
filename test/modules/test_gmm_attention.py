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
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args", [
        ({'n_mixtures': 1}),
        ({'n_mixtures': 4}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    klen = 40
    qlen = 5
    key = torch.FloatTensor(batch_size, klen, args['kdim'])
    value = torch.FloatTensor(batch_size, klen, args['kdim'])
    query = torch.FloatTensor(batch_size, qlen, args['qdim'])
    src_mask = torch.ones(batch_size, 1, klen).byte()

    module = importlib.import_module('neural_sp.models.modules.gmm_attention')
    attention = module.GMMAttention(**args)
    attention.train()
    aws = None
    for i in range(qlen):
        out = attention(key, value, query[:, i:i + 1], mask=src_mask, aw_prev=aws,
                        mode='parallel', cache=True)
        assert len(out) == 4
        cv, aws, _, _ = out
        assert cv.size() == (batch_size, 1, value.size(2))
        assert aws.size() == (batch_size, 1, 1, klen)
