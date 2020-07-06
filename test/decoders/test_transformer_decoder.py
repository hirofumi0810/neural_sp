#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for Transformer decoder."""

import argparse
import importlib
import numpy as np
import pytest
import torch

from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


ENC_N_UNITS = 32
VOCAB = 10


def make_args(**kwargs):
    args = dict(
        special_symbols={'blank': 0, 'unk': 1, 'eos': 2, 'pad': 3},
        enc_n_units=ENC_N_UNITS,
        attn_type='scaled_dot',
        n_heads=4,
        n_layers=2,
        d_model=32,
        d_ff=128,
        ffn_bottleneck_dim=0,
        pe_type='add',
        layer_norm_eps=1e-12,
        ffn_activation='relu',
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
        ctc_fc_list='32_32',
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
    "args",
    [
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
        ({'ffn_bottleneck_dim': 32}),
        # TransformerLM init
        # LM integration
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    emax = 40
    device = "cpu"

    eouts = np.random.randn(batch_size, emax, ENC_N_UNITS).astype(np.float32)
    elens = torch.IntTensor([len(x) for x in eouts])
    eouts = pad_list([np2tensor(x, device).float() for x in eouts], 0.)

    ylens = [4, 5, 3, 7]
    ys = [np.random.randint(0, VOCAB, ylen).astype(np.int32) for ylen in ylens]

    module = importlib.import_module('neural_sp.models.seq2seq.decoders.transformer')
    dec = module.TransformerDecoder(**args)
    loss, observation = dec(eouts, elens, ys, task='all')
    assert loss.dim() == 1
    assert loss.size(0) == 1
    assert loss.item() >= 0
    assert isinstance(observation, dict)


def make_decode_params(**kwargs):
    args = dict(
        recog_batch_size=1,
        recog_beam_width=1,
        recog_ctc_weight=0.0,
        recog_lm_weight=0.0,
        recog_lm_second_weight=0.0,
        recog_lm_bwd_weight=0.0,
        recog_max_len_ratio=1.0,
        recog_min_len_ratio=0.2,
        recog_length_penalty=0.0,
        recog_coverage_penalty=0.0,
        recog_coverage_threshold=1.0,
        recog_length_norm=False,
        recog_eos_threshold=1.5,
        recog_asr_state_carry_over=False,
        recog_lm_state_carry_over=False,
        recog_softmax_smoothing=1.0,
        recog_mma_delay_threshold=-1,
        nbest=1,
        exclude_eos=False,
        cache_states=True,
    )
    args.update(kwargs)
    return args


def make_args_rnnlm(**kwargs):
    args = dict(
        lm_type='lstm',
        n_units=32,
        n_projs=0,
        n_layers=2,
        residual=False,
        use_glu=False,
        n_units_null_context=0,
        bottleneck_dim=32,
        emb_dim=16,
        vocab=VOCAB,
        dropout_in=0.1,
        dropout_hidden=0.1,
        lsm_prob=0.0,
        param_init=0.1,
        adaptive_softmax=False,
        tie_embedding=False,
    )
    args.update(kwargs)
    return argparse.Namespace(**args)


@pytest.mark.parametrize(
    "backward, params",
    [
        # !!! forward
        # greedy decoding
        (False, {'recog_beam_width': 1}),
        (False, {'recog_beam_width': 1, 'cache_states': False}),
        (False, {'recog_beam_width': 1, 'exclude_eos': True}),
        (False, {'recog_beam_width': 1, 'recog_batch_size': 4}),
        # beam search
        (False, {'recog_beam_width': 4}),
        (False, {'recog_beam_width': 4, 'cache_states': False}),
        (False, {'recog_beam_width': 4, 'exclude_eos': True}),
        (False, {'recog_beam_width': 4, 'nbest': 2}),
        (False, {'recog_beam_width': 4, 'nbest': 4}),
        (False, {'recog_beam_width': 4, 'nbest': 4, 'softmax_smoothing': 2.0}),
        (False, {'recog_beam_width': 4, 'recog_ctc_weight': 0.1}),
        # length penalty
        (False, {'recog_length_penalty': 0.1}),
        (False, {'recog_length_norm': True}),
        # shallow fusion
        (False, {'recog_beam_width': 4, 'recog_lm_weight': 0.1}),
        # rescoring
        (False, {'recog_beam_width': 4, 'recog_lm_second_weight': 0.1}),
        (False, {'recog_beam_width': 4, 'recog_lm_bwd_weight': 0.1}),
        # !!! backward
        # greedy decoding
        (True, {'recog_beam_width': 1}),
        (True, {'recog_beam_width': 1, 'cache_states': False}),
        (True, {'recog_beam_width': 1, 'exclude_eos': True}),
        (True, {'recog_beam_width': 1, 'recog_batch_size': 4}),
        # beam search
        (True, {'recog_beam_width': 4}),
        (True, {'recog_beam_width': 4, 'cache_states': False}),
        (True, {'recog_beam_width': 4, 'exclude_eos': True}),
        (True, {'recog_beam_width': 4, 'nbest': 2}),
        (True, {'recog_beam_width': 4, 'nbest': 4}),
        (True, {'recog_beam_width': 4, 'nbest': 4, 'softmax_smoothing': 2.0}),
        (True, {'recog_beam_width': 4, 'recog_ctc_weight': 0.1}),
    ]
)
def test_decoding(backward, params):
    args = make_args()
    params = make_decode_params(**params)
    params['backward'] = backward

    batch_size = params['recog_batch_size']
    emax = 40
    device = "cpu"

    eouts = np.random.randn(batch_size, emax, ENC_N_UNITS).astype(np.float32)
    elens = torch.IntTensor([len(x) for x in eouts])
    eouts = pad_list([np2tensor(x, device).float() for x in eouts], 0.)
    ctc_log_probs = None
    if params['recog_ctc_weight'] > 0:
        ctc_logits = torch.FloatTensor(batch_size, emax, VOCAB, device=device)
        ctc_log_probs = torch.softmax(ctc_logits, dim=-1)
    lm = None
    if params['recog_lm_weight'] > 0:
        args_lm = make_args_rnnlm()
        module = importlib.import_module('neural_sp.models.lm.rnnlm')
        lm = module.RNNLM(args_lm).to(device)
    lm_second = None
    if params['recog_lm_second_weight'] > 0:
        args_lm = make_args_rnnlm()
        module = importlib.import_module('neural_sp.models.lm.rnnlm')
        lm_second = module.RNNLM(args_lm).to(device)
    lm_second_bwd = None
    if params['recog_lm_bwd_weight'] > 0:
        args_lm = make_args_rnnlm()
        module = importlib.import_module('neural_sp.models.lm.rnnlm')
        lm_second_bwd = module.RNNLM(args_lm).to(device)

    ylens = [4, 5, 3, 7]
    ys = [np.random.randint(0, VOCAB, ylen).astype(np.int32) for ylen in ylens]

    module = importlib.import_module('neural_sp.models.seq2seq.decoders.transformer')
    dec = module.TransformerDecoder(**args)
    dec = dec.to(device)

    # TODO(hirofumi0810):
    # recog_lm_state_carry_over

    dec.eval()
    with torch.no_grad():
        if params['recog_beam_width'] == 1:
            out = dec.greedy(eouts, elens, max_len_ratio=1.0, idx2token=None,
                             exclude_eos=params['exclude_eos'],
                             refs_id=ys, utt_ids=None, speakers=None,
                             cache_states=params['cache_states'])
            assert len(out) == 2
            hyps, aws = out
            assert isinstance(hyps, list)
            assert len(hyps) == batch_size
            assert isinstance(aws, list)
            assert aws[0].shape == (args['n_heads'] * args['n_layers'], len(hyps[0]), emax)
        else:
            out = dec.beam_search(eouts, elens, params, idx2token=None,
                                  lm=lm, lm_second=lm_second, lm_second_bwd=lm_second_bwd,
                                  ctc_log_probs=ctc_log_probs,
                                  nbest=params['nbest'], exclude_eos=params['exclude_eos'],
                                  refs_id=None, utt_ids=None, speakers=None,
                                  cache_states=params['cache_states'])
            assert len(out) == 3
            nbest_hyps, aws, scores = out
            assert isinstance(nbest_hyps, list)
            assert len(nbest_hyps) == batch_size
            assert len(nbest_hyps[0]) == params['nbest']
            ymax = len(nbest_hyps[0][0])
            assert isinstance(aws, list)
            assert aws[0][0].shape == (args['n_heads'] * args['n_layers'], ymax, emax)
            assert isinstance(scores, list)
            assert len(scores) == batch_size
            assert len(scores[0]) == params['nbest']

            # ensemble
            ensmbl_eouts, ensmbl_elens, ensmbl_decs = [], [], []
            for _ in range(3):
                ensmbl_eouts += [eouts]
                ensmbl_elens += [elens]
                ensmbl_decs += [dec]

            out = dec.beam_search(eouts, elens, params, idx2token=None,
                                  lm=lm, lm_second=lm_second, lm_second_bwd=lm_second_bwd,
                                  ctc_log_probs=ctc_log_probs,
                                  nbest=params['nbest'], exclude_eos=params['exclude_eos'],
                                  refs_id=None, utt_ids=None, speakers=None,
                                  ensmbl_eouts=ensmbl_eouts, ensmbl_elens=ensmbl_elens, ensmbl_decs=ensmbl_decs,
                                  cache_states=params['cache_states'])
            assert len(out) == 3
            nbest_hyps, aws, scores = out
            assert isinstance(nbest_hyps, list)
            assert len(nbest_hyps) == batch_size
            assert len(nbest_hyps[0]) == params['nbest']
            ymax = len(nbest_hyps[0][0])
            assert isinstance(aws, list)
            assert aws[0][0].shape == (args['n_heads'] * args['n_layers'], ymax, emax)
            assert isinstance(scores, list)
            assert len(scores) == batch_size
            assert len(scores[0]) == params['nbest']
