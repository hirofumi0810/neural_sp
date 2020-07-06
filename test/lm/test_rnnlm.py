#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for RNNLM."""

import argparse
import importlib
import numpy as np
import pytest


VOCAB = 100  # large for adaptive softmax


def make_args(**kwargs):
    args = dict(
        lm_type='lstm',
        n_units=32,
        n_projs=0,
        n_layers=2,
        residual=False,
        use_glu=False,
        n_units_null_context=0,
        bottleneck_dim=16,
        emb_dim=16,
        vocab=VOCAB,
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
        ({'n_projs': 16}),
        # regularization
        ({'lsm_prob': 0.1}),
        ({'residual': True}),
        ({'n_units_null_context': 16}),
        ({'use_glu': True}),
        ({'use_glu': True, 'residual': True}),
        ({'use_glu': True, 'residual': True, 'n_units_null_context': 16}),
        # embedding
        ({'adaptive_softmax': True}),
        ({'tie_embedding': True}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    ylens = [4, 5, 3, 7] * 200
    ys = [np.random.randint(0, VOCAB, ylen).astype(np.int64) for ylen in ylens]
    device = "cpu"

    module = importlib.import_module('neural_sp.models.lm.rnnlm')
    lm = module.RNNLM(args)
    lm = lm.to(device)
    loss, state, observation = lm(ys, state=None, n_caches=0)
    # assert loss.dim() == 1
    # assert loss.size(0) == 1
    assert loss.item() >= 0
    assert isinstance(observation, dict)
