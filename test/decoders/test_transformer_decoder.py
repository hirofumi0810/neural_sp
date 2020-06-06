#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for Transformer decoder."""

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
        attn_type='scaled_dot',
        n_heads=4,
        n_layers=6,
        d_model=64,
        d_ff=256,
        d_ff_bottleneck_dim=0,
        layer_norm_eps=1e-12,
        ffn_activation='relu',
        pe_type='add',
        vocab=VOCAB,
        tie_embedding=False,
        dropout=0.1,
        dropout_emb=0.1,
        dropout_att=0.1,
        dropout_layer=0.0,
        dropout_head=0.0,
        lsm_prob=0.0,
        ctc_weight=0.0,
        ctc_lsm_prob=0.1,
        ctc_fc_list='128_128',
        backward=False,
        global_weight=1.0,
        mtl_per_batch=False,
        param_init='xavier_uniform',
        memory_transformer=False,
        mem_len=0,
        mocha_chunk_size=4,
        mocha_n_heads_mono=1,
        mocha_n_heads_chunk=1,
        mocha_init_r=-4,
        mocha_eps=1e-6,
        mocha_std=1.0,
        mocha_no_denominator=False,
        mocha_1dconv=False,
        mocha_quantity_loss_weight=0.0,
        mocha_head_divergence_loss_weight=0.0,
        latency_metric=False,
        latency_loss_weight=0.0,
        mocha_first_layer=1,
        share_chunkwise_attention=False,
        external_lm=None,
        lm_fusion='',
        # lm_init=False,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args", [
        # head
        ({'n_heads': 1}),
        ({'n_heads': 4}),
        # positional encoding
        ({'pe_type': 'none'}),
        ({'pe_type': '1dconv3L'}),
        # activation
        ({'ffn_activation': 'relu'}),
        ({'ffn_activation': 'gelu'}),
        # ({'ffn_activation': 'glu'}),
        ({'ffn_activation': 'swish'}),
        # MMA
        ({'attn_type': 'mocha', 'mocha_chunk_size': 1, 'mocha_n_heads_mono': 1}),
        ({'attn_type': 'mocha', 'mocha_chunk_size': 1, 'mocha_n_heads_mono': 4}),
        ({'attn_type': 'mocha', 'mocha_chunk_size': 4, 'mocha_n_heads_mono': 1, 'mocha_n_heads_chunk': 1}),
        ({'attn_type': 'mocha', 'mocha_chunk_size': 4, 'mocha_n_heads_mono': 4, 'mocha_n_heads_chunk': 1}),
        ({'attn_type': 'mocha', 'mocha_chunk_size': 4, 'mocha_n_heads_mono': 1, 'mocha_n_heads_chunk': 4}),
        ({'attn_type': 'mocha', 'mocha_chunk_size': 4, 'mocha_n_heads_mono': 4, 'mocha_n_heads_chunk': 4}),
        ({'attn_type': 'mocha', 'mocha_chunk_size': 4, 'mocha_n_heads_mono': 4, 'mocha_n_heads_chunk': 4,
          'share_chunkwise_attention': True}),
        # MMA + HeadDrop
        ({'attn_type': 'mocha', 'dropout_head': 0.1, 'mocha_chunk_size': 1, 'mocha_n_heads_mono': 1}),
        ({'attn_type': 'mocha', 'dropout_head': 0.1, 'mocha_chunk_size': 1, 'mocha_n_heads_mono': 4}),
        ({'attn_type': 'mocha', 'dropout_head': 0.1, 'mocha_chunk_size': 4, 'mocha_n_heads_mono': 1, 'mocha_n_heads_chunk': 1}),
        ({'attn_type': 'mocha', 'dropout_head': 0.1, 'mocha_chunk_size': 4, 'mocha_n_heads_mono': 4, 'mocha_n_heads_chunk': 1}),
        ({'attn_type': 'mocha', 'dropout_head': 0.1, 'mocha_chunk_size': 4, 'mocha_n_heads_mono': 1, 'mocha_n_heads_chunk': 4}),
        ({'attn_type': 'mocha', 'dropout_head': 0.1, 'mocha_chunk_size': 4, 'mocha_n_heads_mono': 4, 'mocha_n_heads_chunk': 4}),
        # regularization
        ({'lsm_prob': 0.1}),
        ({'dropout_layer': 0.1}),
        ({'dropout_head': 0.1}),
        ({'tie_embedding': True}),
        # CTC
        ({'ctc_weight': 0.5}),
        ({'ctc_weight': 1.0}),
        ({'ctc_weight': 1.0, 'ctc_lsm_prob': 0.0}),
        # forward-backward decoder
        ({'backward': True}),
        ({'backward': True, 'ctc_weight': 0.5}),
        ({'backward': True, 'ctc_weight': 1.0}),
        # bottleneck
        ({'d_ff_bottleneck_dim': 256}),
        # RNNLM init
        # LM integration
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

    module = importlib.import_module('neural_sp.models.seq2seq.decoders.transformer')
    dec = module.TransformerDecoder(**args)
    loss, observation = dec(eouts, elens, ys, task='all')
    assert loss.dim() == 1
    assert loss.size(0) == 1
    assert loss.item() >= 0
    assert isinstance(observation, dict)
