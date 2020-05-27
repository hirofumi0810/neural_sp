#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for TransformerLM."""

import argparse
import importlib
import numpy as np
import pytest


ENC_N_UNITS = 64
VOCAB = 100


def make_args(**kwargs):
    args = dict(
        lm_type='transformer',
        transformer_attn_type='scaled_dot',
        transformer_n_heads=4,
        n_layers=2,
        transformer_d_model=64,
        transformer_d_ff=256,
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
        mem_len=0,
        recog_mem_len=0,
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
        ({'mem_len': 5}),
        ({'recog_mem_len': 5}),
        ({'mem_len': 5, 'recog_mem_len': 5}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    ylens = [4, 5, 3, 7] * 200
    ys = [np.random.randint(0, VOCAB, ylen).astype(np.int64) for ylen in ylens]

    module = importlib.import_module('neural_sp.models.lm.transformerlm')
    lm = module.TransformerLM(args)
    loss, state, observation = lm(ys, state=None, n_caches=0)
    # assert loss.dim() == 1, loss
    # assert loss.size(0) == 1, loss
    assert loss.item() >= 0
    assert isinstance(observation, dict)
