#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for relative multihead atteniton."""

import importlib
import pytest
import torch


def make_args(**kwargs):
    args = dict(
        kdim=32,
        qdim=32,
        adim=16,
        odim=32,
        n_heads=4,
        dropout=0.1,
        bias=True,
        param_init='',
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args, learnable",
    [
        ({'n_heads': 1}, False),
        ({'n_heads': 1}, True),
        ({'n_heads': 4}, False),
        ({'n_heads': 4}, True),
        ({'bias': False}, False),
        ({'param_init': 'xavier_uniform'}, False),
    ]
)
def test_forward(args, learnable):
    args = make_args(**args)

    batch_size = 4
    klen = 40
    mlen = 20
    qlen = 5
    device = "cpu"

    key = torch.FloatTensor(batch_size, klen, args['kdim'], device=device)
    memory = torch.FloatTensor(batch_size, mlen, args['kdim'], device=device)
    query = torch.FloatTensor(batch_size, qlen, args['qdim'], device=device)

    # Create the self-attention mask
    causal_mask = torch.ones(qlen, klen + mlen, device=device).byte()
    causal_mask = torch.tril(causal_mask, diagonal=0 + mlen, out=causal_mask).unsqueeze(0)
    causal_mask = causal_mask.repeat([batch_size, 1, 1])  # `[B, qlen, klen+mlen]`

    module_embedding = importlib.import_module('neural_sp.models.modules.positional_embedding')
    pos_emb = module_embedding.XLPositionalEmbedding(args['kdim'], args['dropout'])

    if learnable:
        u = torch.nn.Parameter(torch.Tensor(args['n_heads'], args['adim'] // args['n_heads']))
        u = u.to(device)
        v = torch.nn.Parameter(torch.Tensor(args['n_heads'], args['adim'] // args['n_heads']))
        v = v.to(device)
    else:
        u, v = None, None

    module_mha = importlib.import_module('neural_sp.models.modules.relative_multihead_attention')
    attention = module_mha.RelativeMultiheadAttentionMechanism(**args)
    attention = attention.to(device)

    attention.train()
    aws = None
    for i in range(qlen):
        pos_idxs = torch.arange(klen + mlen - 1, -1, -1.0, dtype=torch.float, device=device)
        pos_embs = pos_emb(pos_idxs)

        out = attention(key, query[:, i:i + 1], memory, mask=causal_mask[:, i:i + 1],
                        pos_embs=pos_embs, u=u, v=v)
        assert len(out) == 2
        cv, aws = out
        assert cv.size() == (batch_size, 1, memory.size(2))
        assert aws.size() == (batch_size, args['n_heads'], 1, klen + mlen)
