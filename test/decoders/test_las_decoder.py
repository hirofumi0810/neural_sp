#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for attention-based RNN decoder."""

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
        attn_type='location',
        rnn_type='lstm',
        n_units=32,
        n_projs=0,
        n_layers=2,
        bottleneck_dim=16,
        emb_dim=16,
        vocab=VOCAB,
        tie_embedding=False,
        attn_dim=32,
        attn_sharpening_factor=1.0,
        attn_sigmoid_smoothing=False,
        attn_conv_out_channels=10,
        attn_conv_kernel_size=201,
        attn_n_heads=1,
        dropout=0.1,
        dropout_emb=0.1,
        dropout_att=0.1,
        lsm_prob=0.0,
        ss_prob=0.0,
        ss_type='constant',
        ctc_weight=0.0,
        ctc_lsm_prob=0.1,
        ctc_fc_list='32_32',
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
    "args",
    [
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
        # projection
        ({'n_projs': 16}),
        # multihead attention
        ({'attn_type': 'add', 'attn_n_heads': 4}),
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
        ({'attn_type': 'mocha', 'mocha_chunk_size': 1, 'mocha_n_heads_mono': 4}),
        ({'attn_type': 'mocha', 'mocha_chunk_size': 4, 'mocha_n_heads_mono': 4}),
        # CTC
        ({'ctc_weight': 0.5}),
        ({'ctc_weight': 1.0}),
        ({'ctc_weight': 1.0, 'ctc_lsm_prob': 0.0}),
        # forward-backward decoder
        ({'backward': True}),
        ({'backward': True, 'ctc_weight': 0.5}),
        ({'backward': True, 'ctc_weight': 1.0}),
        # others
        ({'tie_embedding': True, 'bottleneck_dim': 32, 'emb_dim': 32}),
        ({'lsm_prob': 0.1}),
        ({'ss_prob': 0.2}),
        # RNNLM init
        ({'lm_init': True}),
        # LM integration
        ({'lm_fusion': 'cold'}),
        ({'lm_fusion': 'cold_prob'}),
        ({'lm_fusion': 'deep'}),
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

    if args['lm_init'] or args['lm_fusion']:
        args_lm = make_args_lm()
        module = importlib.import_module('neural_sp.models.lm.rnnlm')
        args['external_lm'] = module.RNNLM(args_lm)

    module = importlib.import_module('neural_sp.models.seq2seq.decoders.las')
    dec = module.RNNDecoder(**args)
    loss, observation = dec(eouts, elens, ys, task='all')
    assert loss.dim() == 1
    assert loss.size(0) == 1
    assert loss.item() >= 0
    assert isinstance(observation, dict)

    # NOTE: this is performed in Speech2Text class
    # if args['lm_fusion']:
    #     for p in dec.lm.parameters():
    #         assert not p.requires_grad
    #     if args['lm_fusion'] == 'deep':
    #         for n, p in dec.named_parameters():
    #             if 'output' in n or 'output_bn' in n or 'linear' in n:
    #                 assert p.requires_grad
    #             else:
    #                 assert not p.requires_grad


def make_decode_params(**kwargs):
    args = dict(
        recog_batch_size=1,
        recog_beam_width=1,
        recog_ctc_weight=0.0,
        recog_lm_weight=0.0,
        recog_lm_second_weight=0.0,
        recog_lm_bwd_weight=0.0,
        recog_max_len_ratio=1.0,
        recog_min_len_ratio=0.1,
        recog_length_penalty=0.0,
        recog_coverage_penalty=0.0,
        recog_coverage_threshold=1.0,
        recog_length_norm=False,
        recog_gnmt_decoding=False,
        recog_eos_threshold=1.0,
        recog_asr_state_carry_over=False,
        recog_lm_state_carry_over=False,
        recog_softmax_smoothing=1.0,
        nbest=1,
        exclude_eos=False,
    )
    args.update(kwargs)
    return args


def make_args_lm(**kwargs):
    args = dict(
        lm_type='lstm',
        n_units=32,
        n_projs=0,
        n_layers=2,
        residual=False,
        use_glu=False,
        n_units_null_context=32,
        bottleneck_dim=16,
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
    "params",
    [
        # greedy decoding
        ({'recog_beam_width': 1}),
        ({'recog_beam_width': 1, 'exclude_eos': True}),
        ({'recog_beam_width': 1, 'recog_batch_size': 4}),
        # beam search
        ({'recog_beam_width': 4}),
        ({'recog_beam_width': 4, 'exclude_eos': True}),
        ({'recog_beam_width': 4, 'nbest': 2}),
        ({'recog_beam_width': 4, 'nbest': 4}),
        ({'recog_beam_width': 4, 'nbest': 4, 'softmax_smoothing': 2.0}),
        ({'recog_beam_width': 4, 'recog_ctc_weight': 0.1}),
        # length penalty
        ({'recog_length_penalty': 0.1}),
        ({'recog_length_penalty': 0.1, 'recog_gnmt_decoding': True}),
        ({'recog_length_norm': True}),
        # coverage
        ({'recog_coverage_penalty': 0.1}),
        ({'recog_coverage_penalty': 0.1, 'recog_gnmt_decoding': True}),
        # shallow fusion
        ({'recog_beam_width': 4, 'recog_lm_weight': 0.1}),
        # rescoring
        ({'recog_beam_width': 4, 'recog_lm_second_weight': 0.1}),
        ({'recog_beam_width': 4, 'recog_lm_bwd_weight': 0.1}),
    ]
)
def test_decoding(params):
    args = make_args()
    params = make_decode_params(**params)

    batch_size = params['recog_batch_size']
    emax = 40
    device_id = -1
    eouts = np.random.randn(batch_size, emax, ENC_N_UNITS).astype(np.float32)
    elens = torch.IntTensor([len(x) for x in eouts])
    eouts = pad_list([np2tensor(x, device_id).float() for x in eouts], 0.)
    ctc_log_probs = None
    if params['recog_ctc_weight'] > 0:
        ctc_log_probs = torch.softmax(torch.FloatTensor(batch_size, emax, VOCAB), dim=-1)
    lm = None
    if params['recog_lm_weight'] > 0:
        args_lm = make_args_lm()
        module = importlib.import_module('neural_sp.models.lm.rnnlm')
        lm = module.RNNLM(args_lm)
    lm_second = None
    if params['recog_lm_second_weight'] > 0:
        args_lm = make_args_lm()
        module = importlib.import_module('neural_sp.models.lm.rnnlm')
        lm_second = module.RNNLM(args_lm)
    lm_second_bwd = None
    if params['recog_lm_bwd_weight'] > 0:
        args_lm = make_args_lm()
        module = importlib.import_module('neural_sp.models.lm.rnnlm')
        lm_second_bwd = module.RNNLM(args_lm)

    ylens = [4, 5, 3, 7]
    ys = [np.random.randint(0, VOCAB, ylen).astype(np.int32) for ylen in ylens]

    module = importlib.import_module('neural_sp.models.seq2seq.decoders.las')
    dec = module.RNNDecoder(**args)

    # TODO(hirofumi0810):
    # recog_asr_state_carry_over
    # recog_lm_state_carry_over
    # cold fusion + beam search + shallow fusion
    # deep fusion + beam search + shallow fusion

    dec.eval()
    with torch.no_grad():
        if params['recog_beam_width'] == 1:
            out = dec.greedy(eouts, elens, max_len_ratio=1.0, idx2token=None,
                             exclude_eos=params['exclude_eos'],
                             refs_id=ys, utt_ids=None, speakers=None)
            assert len(out) == 2
            hyps, aws = out
            assert isinstance(hyps, list)
            assert len(hyps) == batch_size
            assert isinstance(aws, list)
            assert aws[0].shape == (args['attn_n_heads'], len(hyps[0]), emax)
        else:
            out = dec.beam_search(eouts, elens, params, idx2token=None,
                                  lm=lm, lm_second=lm_second, lm_second_bwd=lm_second_bwd,
                                  ctc_log_probs=ctc_log_probs,
                                  nbest=params['nbest'], exclude_eos=params['exclude_eos'],
                                  refs_id=None, utt_ids=None, speakers=None,
                                  cache_states=True)
            assert len(out) == 3
            nbest_hyps, aws, scores = out
            assert isinstance(nbest_hyps, list)
            assert len(nbest_hyps) == batch_size
            assert len(nbest_hyps[0]) == params['nbest']
            ymax = len(nbest_hyps[0][0])
            assert isinstance(aws, list)
            assert aws[0][0].shape == (args['attn_n_heads'], ymax, emax)
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
                                  cache_states=True)
            assert len(out) == 3
            nbest_hyps, aws, scores = out
            assert isinstance(nbest_hyps, list)
            assert len(nbest_hyps) == batch_size
            assert len(nbest_hyps[0]) == params['nbest']
            ymax = len(nbest_hyps[0][0])
            assert isinstance(aws, list)
            assert aws[0][0].shape == (args['attn_n_heads'], ymax, emax)
            assert isinstance(scores, list)
            assert len(scores) == batch_size
            assert len(scores[0]) == params['nbest']
