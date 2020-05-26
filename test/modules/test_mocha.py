#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for monotonic chunkwise atteniton (MoChA)."""

import importlib
import pytest
import torch


def make_args(**kwargs):
    args = dict(
        kdim=32,
        qdim=32,
        adim=16,
        atype='add',
        chunk_size=1,
        n_heads_mono=1,
        n_heads_chunk=1,
        conv1d=False,
        init_r=-4,
        eps=1e-6,
        noise_std=1.0,
        no_denominator=False,
        sharpening_factor=1.0,
        dropout=0.,
        dropout_head=0.,
        bias=True,
        param_init='',
        decot=False,
        lookahead=2
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args", [
        # hard monotonic attention
        ({'n_heads_mono': 1, 'chunk_size': 1}),
        ({'n_heads_mono': 1, 'chunk_size': 1, 'conv1d': True}),
        ({'n_heads_mono': 1, 'chunk_size': 1, 'no_denominator': True}),
        ({'n_heads_mono': 1, 'chunk_size': 1, 'bias': False}),
        # mocha
        ({'n_heads_mono': 1, 'chunk_size': 4}),
        # MMA
        ({'n_heads_mono': 4, 'n_heads_chunk': 1, 'chunk_size': 1, 'atype': 'scaled_dot'}),
        ({'n_heads_mono': 4, 'n_heads_chunk': 1, 'chunk_size': 4, 'atype': 'scaled_dot'}),
        ({'n_heads_mono': 1, 'n_heads_chunk': 4, 'chunk_size': 4, 'atype': 'scaled_dot'}),
        ({'n_heads_mono': 4, 'n_heads_chunk': 4, 'chunk_size': 4, 'atype': 'scaled_dot'}),
    ]
)
def test_forward_hard(args):
    args = make_args(**args)

    batch_size = 4
    klen = 40
    qlen = 5
    key = torch.FloatTensor(batch_size, klen, args['kdim'])
    value = torch.FloatTensor(batch_size, klen, args['kdim'])
    query = torch.FloatTensor(batch_size, qlen, args['qdim'])

    mocha = importlib.import_module('neural_sp.models.modules.mocha')
    attention = mocha.MoChA(**args)
    attention.eval()
    alpha = None
    for i in range(qlen):
        out = attention(key, value, query[:, i:i + 1], mask=None, aw_prev=alpha,
                        mode='hard', cache=False, eps_wait=-1,
                        efficient_decoding=False)
        assert len(out) == 3
        cv, alpha, beta = out
        assert cv.size() == (batch_size, 1, value.size(2))
        assert alpha.size() == (batch_size, args['n_heads_mono'], 1, klen)
        if args['chunk_size'] > 1:
            assert beta is not None
            assert beta.size() == (batch_size, args['n_heads_mono'] * args['n_heads_chunk'], 1, klen)


@pytest.mark.parametrize(
    "args", [
        # hard monotonic attention
        ({'n_heads_mono': 1, 'chunk_size': 1}),
        ({'n_heads_mono': 1, 'chunk_size': 1, 'conv1d': True}),
        ({'n_heads_mono': 1, 'chunk_size': 1, 'no_denominator': True}),
        ({'n_heads_mono': 1, 'chunk_size': 1, 'bias': False}),
        # mocha
        ({'n_heads_mono': 1, 'chunk_size': 4}),
        # MMA
        ({'n_heads_mono': 4, 'n_heads_chunk': 1, 'chunk_size': 1, 'atype': 'scaled_dot'}),
        ({'n_heads_mono': 4, 'n_heads_chunk': 1, 'chunk_size': 4, 'atype': 'scaled_dot'}),
        ({'n_heads_mono': 1, 'n_heads_chunk': 4, 'chunk_size': 4, 'atype': 'scaled_dot'}),
        ({'n_heads_mono': 4, 'n_heads_chunk': 4, 'chunk_size': 4, 'atype': 'scaled_dot'}),
        # HeadDrop
        ({'n_heads_mono': 4, 'n_heads_chunk': 1, 'chunk_size': 1, 'atype': 'scaled_dot',
          'dropout_head': 0.5}),
        ({'n_heads_mono': 4, 'n_heads_chunk': 1, 'chunk_size': 4, 'atype': 'scaled_dot',
          'dropout_head': 0.5}),
        ({'n_heads_mono': 1, 'n_heads_chunk': 4, 'chunk_size': 4, 'atype': 'scaled_dot',
          'dropout_head': 0.5}),
        ({'n_heads_mono': 4, 'n_heads_chunk': 4, 'chunk_size': 4, 'atype': 'scaled_dot',
          'dropout_head': 0.5}),
    ]
)
def test_forward_soft_parallel(args):
    args = make_args(**args)

    batch_size = 4
    klen = 40
    qlen = 5
    key = torch.FloatTensor(batch_size, klen, args['kdim'])
    value = torch.FloatTensor(batch_size, klen, args['kdim'])
    query = torch.FloatTensor(batch_size, qlen, args['qdim'])
    src_mask = torch.ones(batch_size, 1, klen).byte()

    mocha = importlib.import_module('neural_sp.models.modules.mocha')
    attention = mocha.MoChA(**args)
    attention.train()
    alpha = None
    for i in range(qlen):
        out = attention(key, value, query[:, i:i + 1], mask=src_mask, aw_prev=alpha,
                        mode='parallel', cache=True)
        assert len(out) == 3
        cv, alpha, beta = out
        assert cv.size() == (batch_size, 1, value.size(2))
        assert alpha.size() == (batch_size, args['n_heads_mono'], 1, klen)
        if args['chunk_size'] > 1:
            assert beta is not None
            assert beta.size() == (batch_size, args['n_heads_mono'] * args['n_heads_chunk'], 1, klen)
