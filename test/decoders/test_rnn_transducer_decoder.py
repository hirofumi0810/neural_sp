#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for RNN Transducer."""

import importlib
import numpy as np
import pytest
import torch

from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


ENC_N_UNITS = 64
VOCAB = 10


def make_args(**kwargs):
    args = dict(
        special_symbols={'blank': 0, 'unk': 1, 'eos': 2, 'pad': 3},
        enc_n_units=ENC_N_UNITS,
        rnn_type='lstm_transducer',
        n_units=64,
        n_projs=0,
        n_layers=2,
        bottleneck_dim=32,
        emb_dim=16,
        vocab=VOCAB,
        dropout=0.1,
        dropout_emb=0.1,
        lsm_prob=0.0,
        ctc_weight=0.0,
        ctc_lsm_prob=0.1,
        ctc_fc_list='128_128',
        external_lm=None,
        global_weight=1.0,
        mtl_per_batch=False,
        param_init=0.1,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args", [
        # RNN type
        ({'rnn_type': 'lstm_transducer', 'n_layers': 1}),
        ({'rnn_type': 'lstm_transducer', 'n_layers': 2}),
        ({'rnn_type': 'gru_transducer', 'n_layers': 1}),
        ({'rnn_type': 'gru_transducer', 'n_layers': 2}),
        # CTC
        ({'ctc_weight': 0.5}),
        ({'ctc_weight': 1.0}),
        ({'ctc_weight': 1.0, 'ctc_lsm_prob': 0.0}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    emax = 40
    device_id = -1
    eouts = np.random.randn(batch_size, emax, ENC_N_UNITS).astype(np.float32)
    elens = torch.IntTensor([len(x) for x in eouts])
    eouts = pad_list([np2tensor(x, device_id).float() for x in eouts], 0.)

    ylens = [4, 5, 3, 7]
    ys = [np.random.randint(0, VOCAB, ylen).astype(np.int32) for ylen in ylens]

    module = importlib.import_module('neural_sp.models.seq2seq.decoders.rnn_transducer')
    dec = module.RNNTransducer(**args)
    loss, observation = dec(eouts, elens, ys, task='all')
    assert loss.dim() == 1
    assert loss.size(0) == 1
    assert loss.item() >= 0
    assert isinstance(observation, dict)
