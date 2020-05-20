#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for decoders."""

import importlib
import numpy as np
import pytest
import torch

from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


ENC_N_UNITS = 64
VOCAB = 10


def make_rnn_args(**kwargs):
    args = dict(
        special_symbols={'blank': 0, 'unk': 1, 'eos': 2, 'pad': 3},
        enc_n_units=ENC_N_UNITS,
        attn_type='location',
        rnn_type='lstm',
        n_units=64,
        n_projs=0,
        n_layers=2,
        bottleneck_dim=32,
        emb_dim=16,
        vocab=VOCAB,
        tie_embedding=False,
        attn_dim=128,
        attn_sharpening_factor=1.0,
        attn_sigmoid_smoothing=False,
        attn_conv_out_channels=10,
        attn_conv_kernel_size=201,
        attn_n_heads=1,
        dropout=0.1,
        dropout_emb=0.1,
        dropout_att=0.1,
        lsm_prob=0.1,
        ss_prob=0.2,
        ss_type='constant',
        ctc_weight=0.0,
        ctc_lsm_prob=0.1,
        ctc_fc_list='128_128',
        mbr_training=False,
        mbr_ce_weight=0.01,
        external_lm=None,
        lm_fusion='',
        lm_init=False,
        backward=False,
        global_weight=1.0,
        mtl_per_batch=False,
        param_init=0.1,
        mocha_chunk_size=4,
        mocha_n_heads_mono=1,
        mocha_init_r=-4,
        mocha_eps=1e-6,
        mocha_std=1.0,
        mocha_no_denominator=False,
        mocha_1dconv=False,
        mocha_quantity_loss_weight=0.0,
        latency_metric=False,
        latency_loss_weight=0.0,
        gmm_attn_n_mixtures=1,
        replace_sos=False,
        distillation_weight=0.0,
        discourse_aware=False
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args", [
        # RNN type
        ({'rnn_type': 'lstm', 'n_layers': 1}),
        ({'rnn_type': 'lstm', 'n_layers': 2}),
        ({'rnn_type': 'gru', 'n_layers': 1}),
        ({'rnn_type': 'gru', 'n_layers': 2}),
        # attention
        ({'attn_type': 'add'}),
        ({'attn_type': 'dot'}),
        ({'attn_type': 'luong_dot'}),
        ({'attn_type': 'luong_general'}),
        ({'attn_type': 'luong_concat'}),
        ({'attn_type': 'gmm', 'gmm_attn_n_mixtures': 5}),
        # MoChA
        ({'attn_type': 'mocha', 'mocha_chunk_size': 1}),
        ({'attn_type': 'mocha', 'mocha_chunk_size': 4}),
        ({'attn_type': 'mocha', 'mocha_chunk_size': 4, 'mocha_no_denominator': True}),
        ({'attn_type': 'mocha', 'mocha_chunk_size': 4, 'mocha_1dconv': True}),
        ({'attn_type': 'mocha', 'mocha_chunk_size': 4, 'mocha_quantity_loss_weight': 1.0}),
        # ({'attn_type': 'mocha', 'mocha_chunk_size': 4,
        #   'ctc_weight': 0.5, 'latency_metric': 'ctc_sync', 'latency_loss_weight': 1.0}),
        # ({'attn_type': 'mocha', 'mocha_chunk_size': 4, 'mocha_quantity_loss_weight': 1.0,
        #   'ctc_weight': 0.5, 'latency_metric': 'ctc_sync', 'latency_loss_weight': 1.0}),
        # multihead attention
        ({'attn_type': 'add', 'attn_n_heads': 4}),
        ({'attn_type': 'mocha', 'mocha_chunk_size': 1, 'mocha_n_heads_mono': 4}),
        ({'attn_type': 'mocha', 'mocha_chunk_size': 4, 'mocha_n_heads_mono': 4}),
        # CTC
        ({'ctc_weight': 0.5}),
        ({'ctc_weight': 1.0}),
        # forward-backward decoder
        ({'backward': True}),
        ({'backward': True, 'ctc_weight': 0.5}),
        ({'backward': True, 'ctc_weight': 1.0}),
    ]
)
def test_rnn_forward(args):
    args = make_rnn_args(**args)

    batch_size = 4
    emax = 40
    device_id = -1
    eouts = np.random.randn(batch_size, emax, ENC_N_UNITS).astype(np.float32)
    elens = torch.IntTensor([len(x) for x in eouts])
    eouts = pad_list([np2tensor(x, device_id).float() for x in eouts], 0.)

    ylens = [4, 5, 3, 7]
    ys = [np.random.randint(0, VOCAB, ylen).astype(np.int32) for ylen in ylens]

    las = importlib.import_module('neural_sp.models.seq2seq.decoders.las')
    dec = las.RNNDecoder(**args)
    loss, observation = dec(eouts, elens, ys, task='all')
    assert loss.dim() == 1
    assert loss.size(0) == 1
    assert loss.item() >= 0
    assert isinstance(observation, dict)
