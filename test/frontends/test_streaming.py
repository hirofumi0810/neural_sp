#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for streaming interface."""

import importlib
import numpy as np
import pytest


def make_rnn_args(**kwargs):
    args = dict(
        input_dim=80,
        enc_type='blstm',
        n_units=16,
        n_projs=0,
        last_proj_dim=0,
        n_layers=2,
        n_layers_sub1=0,
        n_layers_sub2=0,
        dropout_in=0.1,
        dropout=0.1,
        subsample="1_1",
        subsample_type='drop',
        n_stacks=1,
        n_splices=1,
        conv_in_channel=1,
        conv_channels="32_32",
        conv_kernel_sizes="(3,3)_(3,3)",
        conv_strides="(1,1)_(1,1)",
        conv_poolings="(2,2)_(2,2)",
        conv_batch_norm=False,
        conv_layer_norm=False,
        conv_bottleneck_dim=0,
        bidir_sum_fwd_bwd=False,
        task_specific_layer=False,
        param_init=0.1,
        chunk_size_current="0",
        chunk_size_right="0",
        rsp_prob=0,
    )
    args.update(kwargs)
    return args


def make_transformer_args(**kwargs):
    args = dict(
        input_dim=80,
        enc_type='conv_transformer',
        n_heads=4,
        n_layers=3,
        n_layers_sub1=0,
        n_layers_sub2=0,
        d_model=16,
        d_ff=64,
        ffn_bottleneck_dim=0,
        ffn_activation='relu',
        pe_type='none',
        layer_norm_eps=1e-12,
        last_proj_dim=0,
        dropout_in=0.1,
        dropout=0.1,
        dropout_att=0.1,
        dropout_layer=0.1,
        subsample="1_1_1",
        subsample_type='max_pool',
        n_stacks=1,
        n_splices=1,
        conv_in_channel=1,
        conv_channels="32_32",
        conv_kernel_sizes="(3,3)_(3,3)",
        conv_strides="(1,1)_(1,1)",
        conv_poolings="(2,2)_(2,2)",
        conv_batch_norm=False,
        conv_layer_norm=False,
        conv_bottleneck_dim=0,
        conv_param_init=0.1,
        task_specific_layer=False,
        param_init='xavier_uniform',
        clamp_len=-1,
        lookahead="0",
        chunk_size_left="0",
        chunk_size_current="0",
        chunk_size_right="0",
        streaming_type='mask',
    )
    args.update(kwargs)
    return args


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
        recog_gnmt_decoding=False,
        recog_eos_threshold=1.5,
        recog_asr_state_carry_over=False,
        recog_lm_state_carry_over=False,
        recog_softmax_smoothing=1.0,
        nbest=1,
        exclude_eos=False,
        recog_block_sync=True,
        recog_block_sync_size=40,
        recog_ctc_vad=False,
        recog_ctc_vad_blank_threshold=40,
        recog_ctc_vad_spike_threshold=0.1,
        recog_ctc_vad_n_accum_frames=4000,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        # no CNN, UniLSTM, LC-BLSTM
        ({'enc_type': 'lstm'}),  # unidirectional
        ({'enc_type': 'conv_lstm'}),  # unidirectional
        ({'enc_type': 'blstm', 'chunk_size_current': "20", 'chunk_size_right': "20"}),
        ({'enc_type': 'conv_blstm', 'chunk_size_current': "20", 'chunk_size_right': "20"}),
        # no CNN, LC-Transformer
        ({'enc_type': 'transformer', 'streaming_type': 'reshape',
          'chunk_size_current': "32", 'chunk_size_current': "32", 'chunk_size_right': "32"}),
        ({'enc_type': 'conv_transformer', 'streaming_type': 'reshape',
          'chunk_size_current': "32", 'chunk_size_current': "32", 'chunk_size_right': "32"}),
    ]
)
def test_feature_extraction(args):
    if 'lstm' in args['enc_type']:
        args = make_rnn_args(**args)
        enc_module = importlib.import_module('neural_sp.models.seq2seq.encoders.rnn')
        enc = enc_module.RNNEncoder(**args)
    else:
        args = make_transformer_args(**args)
        enc_module = importlib.import_module('neural_sp.models.seq2seq.encoders.transformer')
        enc = enc_module.TransformerEncoder(**args)
    decode_args = make_decode_params()
    if args['enc_type'] in ['lstm', 'conv_lstm', 'uni_transformer', 'conv_uni_transformer']:
        args['chunk_size_current'] = 4

    xmaxs = [t for t in range(160, 192, 1)]
    device = "cpu"
    enc = enc.to(device)

    streaming_module = importlib.import_module('neural_sp.models.seq2seq.frontends.streaming')
    block_size = decode_args['recog_block_sync_size']  # before subsampling

    for xmax in xmaxs:
        xs = np.arange(xmax)[:, None].astype(np.float32)
        streaming = streaming_module.Streaming(xs, decode_args, enc, block_size)
        N_l = streaming.N_l
        conv_context = streaming.conv_context
        if args['enc_type'] in ['conv_lstm', 'conv_uni_transformer']:
            assert conv_context > 0

        xs_cat = []
        while True:
            x_block, is_last_block, cnn_lookback, cnn_lookahead, xlen_block = streaming.extract_feature()
            xs_cat.append(x_block[conv_context + N_l:conv_context + N_l + xlen_block])

            streaming.next_block()
            if is_last_block:
                break

        xs_cat = np.concatenate(xs_cat, axis=0)
        assert np.array_equal(xs, xs_cat)
