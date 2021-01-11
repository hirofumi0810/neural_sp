#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for Transformer-XL LM."""

import argparse
import importlib
import numpy as np
import pytest


VOCAB = 100  # large for adaptive softmax


def make_args(**kwargs):
    args = dict(
        lm_type='transformer',
        transformer_attn_type='scaled_dot',
        transformer_n_heads=4,
        n_layers=2,
        transformer_d_model=16,
        transformer_d_ff=64,
        transformer_layer_norm_eps=1e-12,
        transformer_ffn_activation='relu',
        transformer_pe_type='add',
        vocab=VOCAB,
        dropout_in=0.1,
        dropout_hidden=0.1,
        dropout_att=0.1,
        dropout_layer=0.0,
        # dropout_out=0.1,
        lsm_prob=0.0,
        transformer_param_init='xavier_uniform',
        bptt=200,
        mem_len=100,
        recog_mem_len=1000,
        adaptive_softmax=False,
        tie_embedding=False,
    )
    args.update(kwargs)
    return argparse.Namespace(**args)


@pytest.mark.parametrize(
    "args", [
        # head
        ({'transformer_n_heads': 1}),
        ({'transformer_n_heads': 4}),
        # positional encoding
        ({'pe_type': 'none'}),
        ({'pe_type': '1dconv3L'}),
        # activation
        ({'ffn_activation': 'relu'}),
        ({'ffn_activation': 'gelu'}),
        # ({'ffn_activation': 'glu'}),
        # regularization
        ({'lsm_prob': 0.1}),
        ({'dropout_layer': 0.1}),
        ({'tie_embedding': True}),
        # embedding
        ({'adaptive_softmax': True}),
        # memory
        ({'mem_len': 0}),
        ({'recog_mem_len': 0}),
        ({'mem_len': 0, 'recog_mem_len': 0}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    ylens = [4, 5, 3, 7] * 200
    ys = [np.random.randint(0, VOCAB, ylen).astype(np.int64) for ylen in ylens]
    device = "cpu"

    module = importlib.import_module('neural_sp.models.lm.transformer_xl')
    lm = module.TransformerXL(args)
    lm = lm.to(device)
    loss, state, observation = lm(ys, state=None, n_caches=0)
    # assert loss.dim() == 1
    # assert loss.size(0) == 1
    assert loss.item() >= 0
    assert isinstance(observation, dict)
