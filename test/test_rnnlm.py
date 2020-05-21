#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for RNNLM."""

import argparse
import importlib
import numpy as np
import pytest


ENC_N_UNITS = 64
VOCAB = 100


def make_args(**kwargs):
    args = dict(
        lm_type='lstm',
        n_units=64,
        n_projs=0,
        n_layers=2,
        residual=False,
        use_glu=False,
        n_units_null_context=0,
        bottleneck_dim=32,
        emb_dim=16,
        vocab=VOCAB,
        dropout=0.1,
        dropout_in=0.1,
        dropout_hidden=0.1,
        # dropout_out=0.1,
        lsm_prob=0.0,
        param_init=0.1,
        adaptive_softmax=False,
        tie_embedding=False,
    )
    args.update(kwargs)
    return argparse.Namespace(**args)


@pytest.mark.parametrize(
    "args", [
        # RNN type
        ({'lm_type': 'lstm', 'n_layers': 1}),
        ({'lm_type': 'lstm', 'n_layers': 2}),
        ({'lm_type': 'gru', 'n_layers': 1}),
        ({'lm_type': 'gru', 'n_layers': 2}),
        # projection
        ({'n_projs': 32}),
        # regularization
        ({'lsm_prob': 0.1}),
        ({'residual': True}),
        ({'n_units_null_context': 32}),
        ({'use_glu': True}),
        ({'use_glu': True, 'residual': True}),
        ({'use_glu': True, 'residual': True, 'n_units_null_context': 32s}),
        # embedding
        ({'adaptive_softmax': True}),
        ({'tie_embedding': True}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    ylens = [4, 5, 3, 7]
    ys = [np.random.randint(0, VOCAB, ylen).astype(np.int64) for ylen in ylens]

    rnnlm = importlib.import_module('neural_sp.models.lm.rnnlm')
    lm = rnnlm.RNNLM(args)
    loss, state, observation = lm(ys, state=None, n_caches=0)
    # assert loss.dim() == 1
    # assert loss.size(0) == 1
    assert loss.item() >= 0
    assert isinstance(observation, dict)
