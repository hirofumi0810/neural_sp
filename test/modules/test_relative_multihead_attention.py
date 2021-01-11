#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for relative multihead atteniton."""

import importlib
import pytest
import torch
import warnings

torch.manual_seed(0)


def make_args(**kwargs):
    args = dict(
        kdim=32,
        qdim=32,
        adim=16,
        odim=32,
        n_heads=4,
        dropout=0.1,
        bias=False,
        param_init='',
        xl_like=False,
        clamp_len=-1,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        ({'n_heads': 1}),
        ({'n_heads': 4}),
        ({'bias': True}),
        ({'param_init': 'xavier_uniform'}),
        ({'clamp_len': 4}),
        # TransformerXL like
        ({'n_heads': 1, 'xl_like': True}),
        ({'n_heads': 4, 'xl_like': True}),
        ({'bias': True, 'xl_like': True}),
        ({'param_init': 'xavier_uniform', 'xl_like': True}),
        ({'clamp_len': 4, 'xl_like': True}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    mlen = 20 if args['xl_like'] else 0
    qlen = 5
    device = "cpu"

    query = torch.FloatTensor(batch_size, qlen, args['qdim'], device=device)
    if mlen > 0:
        memory = torch.FloatTensor(batch_size, mlen, args['kdim'], device=device)
        cat = torch.cat([memory, query], dim=1)
    else:
        cat = query

    # Create the self-attention mask
    causal_mask = torch.ones(qlen, qlen + mlen, device=device).byte()
    causal_mask = torch.tril(causal_mask, diagonal=0 + mlen, out=causal_mask).unsqueeze(0)
    causal_mask = causal_mask.repeat([batch_size, 1, 1])  # `[B, qlen, mlen+qlen]`

    module_embedding = importlib.import_module('neural_sp.models.modules.positional_embedding')
    pos_emb = module_embedding.XLPositionalEmbedding(args['kdim'], args['dropout'])

    if args['xl_like']:
        u_bias = torch.nn.Parameter(torch.Tensor(args['n_heads'], args['adim'] // args['n_heads']))
        u_bias = u_bias.to(device)
        v_bias = torch.nn.Parameter(torch.Tensor(args['n_heads'], args['adim'] // args['n_heads']))
        v_bias = v_bias.to(device)
    else:
        u_bias, v_bias = None, None

    module_mha = importlib.import_module('neural_sp.models.modules.relative_multihead_attention')
    attention = module_mha.RelativeMultiheadAttentionMechanism(**args)
    attention = attention.to(device)

    attention.train()
    aws = None
    pos_embs = pos_emb(query, mlen=mlen)

    out = attention(cat, query, pos_embs, causal_mask,
                    u_bias=u_bias, v_bias=v_bias)
    assert len(out) == 2
    cv, aws = out
    assert cv.size() == (batch_size, qlen, args['kdim'])
    assert aws.size() == (batch_size, args['n_heads'], qlen, qlen + mlen)

    # incremental check
    cv_incremental = []
    for t in range(qlen):
        pos_embs_t = pos_emb(query[:, t:t + 1], mlen=mlen + t)
        cv_incremental.append(attention(cat[:, :mlen + t + 1],
                                        query[:, t:t + 1],
                                        pos_embs_t,
                                        causal_mask[:, t:t + 1, :mlen + t + 1],
                                        u_bias=u_bias, v_bias=v_bias)[0])
    cv_incremental = torch.cat(cv_incremental, dim=1)
    assert cv.size() == cv_incremental.size()
    if not torch.allclose(cv, cv_incremental, equal_nan=True):
        warnings.warn("Incremental output did not match.", UserWarning)
