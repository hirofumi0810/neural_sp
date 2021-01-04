#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for RNN Transducer."""

import argparse
import importlib
import numpy as np
import pytest
import torch

from neural_sp.datasets.token_converter.character import Idx2char
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


ENC_N_UNITS = 16
VOCAB = 10

idx2token = Idx2char('test/decoders/dict.txt')


def make_args(**kwargs):
    args = dict(
        special_symbols={'blank': 0, 'unk': 1, 'eos': 2, 'pad': 3},
        enc_n_units=ENC_N_UNITS,
        rnn_type='lstm_transducer',
        n_units=16,
        n_projs=0,
        n_layers=2,
        bottleneck_dim=8,
        emb_dim=8,
        vocab=VOCAB,
        dropout=0.1,
        dropout_emb=0.1,
        ctc_weight=0.1,
        ctc_lsm_prob=0.1,
        ctc_fc_list='16_16',
        external_lm=None,
        global_weight=1.0,
        mtl_per_batch=False,
        param_init=0.1,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        # RNN type
        ({'rnn_type': 'lstm_transducer', 'n_layers': 1}),
        ({'rnn_type': 'lstm_transducer', 'n_layers': 2}),
        ({'rnn_type': 'gru_transducer', 'n_layers': 1}),
        ({'rnn_type': 'gru_transducer', 'n_layers': 2}),
        # projection
        ({'n_projs': 8}),
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
    device = "cpu"

    eouts = np.random.randn(batch_size, emax, ENC_N_UNITS).astype(np.float32)
    elens = torch.IntTensor([len(x) for x in eouts])
    eouts = pad_list([np2tensor(x, device).float() for x in eouts], 0.)

    ylens = [4, 5, 3, 7]
    ys = [np.random.randint(0, VOCAB, ylen).astype(np.int32) for ylen in ylens]

    module = importlib.import_module('neural_sp.models.seq2seq.decoders.rnn_transducer')
    dec = module.RNNTransducer(**args)
    dec = dec.to(device)

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
        recog_lm_state_carry_over=False,
        recog_softmax_smoothing=1.0,
        nbest=1,
    )
    args.update(kwargs)
    return args


def make_args_rnnlm(**kwargs):
    args = dict(
        lm_type='lstm',
        n_units=16,
        n_projs=0,
        n_layers=2,
        residual=False,
        use_glu=False,
        n_units_null_context=0,
        bottleneck_dim=8,
        emb_dim=8,
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
        ({'recog_beam_width': 1, 'recog_batch_size': 4}),
        # beam search
        ({'recog_beam_width': 4}),
        ({'recog_beam_width': 4, 'recog_batch_size': 4}),
        ({'recog_beam_width': 4, 'nbest': 2}),
        ({'recog_beam_width': 4, 'nbest': 4}),
        ({'recog_beam_width': 4, 'recog_softmax_smoothing': 0.8}),
        # ({'recog_beam_width': 4, 'recog_ctc_weight': 0.1}),
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

    module = importlib.import_module('neural_sp.models.seq2seq.decoders.rnn_transducer')
    dec = module.RNNTransducer(**args)
    dec = dec.to(device)

    dec.eval()
    with torch.no_grad():
        if params['recog_beam_width'] == 1:
            out = dec.greedy(eouts, elens, max_len_ratio=params['recog_max_len_ratio'],
                             idx2token=idx2token, exclude_eos=False,
                             refs_id=ys, utt_ids=None, speakers=None)
            assert len(out) == 2
            hyps, aws = out
            assert isinstance(hyps, list)
            assert len(hyps) == batch_size
            assert aws is None
        else:
            out = dec.beam_search(eouts, elens, params, idx2token=idx2token,
                                  lm=lm, lm_second=lm_second, lm_second_bwd=lm_second_bwd,
                                  ctc_log_probs=ctc_log_probs,
                                  nbest=params['nbest'], exclude_eos=False,
                                  refs_id=None, utt_ids=None, speakers=None,
                                  ensmbl_eouts=None, ensmbl_elens=None, ensmbl_decs=[])
            assert len(out) == 3
            nbest_hyps, aws, scores = out
            assert isinstance(nbest_hyps, list)
            assert len(nbest_hyps) == batch_size
            assert len(nbest_hyps[0]) == params['nbest']
            assert aws is None
            assert scores is None
