#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for RNN encoders."""

import importlib
import numpy as np
import pytest
import torch

from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


def make_args(**kwargs):
    args = dict(
        input_dim=80,
        rnn_type='blstm',
        n_units=128,
        n_projs=0,
        last_proj_dim=0,
        n_layers=5,
        n_layers_sub1=0,
        n_layers_sub2=0,
        dropout_in=0.1,
        dropout=0.1,
        subsample=[1, 1, 1, 1, 1],
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
        bidirectional_sum_fwd_bwd=False,
        task_specific_layer=False,
        param_init=0.1,
        chunk_size_left=-1,
        chunk_size_right=-1
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args", [
        # RNN type
        ({'rnn_type': 'blstm'}),
        ({'rnn_type': 'bgru'}),
        ({'rnn_type': 'conv_blstm'}),
        ({'rnn_type': 'conv_blstm', 'input_dim': 240, 'conv_in_channel': 3}),
        ({'rnn_type': 'conv_bgru'}),
        ({'rnn_type': 'lstm'}),
        ({'rnn_type': 'lstm', }),
        ({'rnn_type': 'gru'}),
        ({'rnn_type': 'conv_gru'}),
        # normalization
        ({'rnn_type': 'conv_blstm', 'conv_batch_norm': True}),
        ({'rnn_type': 'conv_blstm', 'conv_layer_norm': True}),
        # subsampling
        ({'rnn_type': 'blstm', 'subsample': [1, 2, 2, 1, 1], 'subsample_type': 'drop'}),
        ({'rnn_type': 'blstm', 'subsample': [1, 2, 2, 1, 1], 'subsample_type': 'concat'}),
        ({'rnn_type': 'blstm', 'subsample': [1, 2, 2, 1, 1], 'subsample_type': 'max_pool'}),
        ({'rnn_type': 'blstm', 'subsample': [1, 2, 2, 1, 1], 'subsample_type': '1dconv'}),
        ({'rnn_type': 'blstm', 'subsample': [1, 2, 2, 1, 1], 'subsample_type': 'drop',
          'bidirectional_sum_fwd_bwd': True}),
        # ({'rnn_type': 'blstm', 'subsample': [1, 2, 2, 1, 1], 'subsample_type': 'concat',
        #   'bidirectional_sum_fwd_bwd': True}),
        ({'rnn_type': 'blstm', 'subsample': [1, 2, 2, 1, 1], 'subsample_type': 'max_pool',
          'bidirectional_sum_fwd_bwd': True}),
        # ({'rnn_type': 'blstm', 'subsample': [1, 2, 2, 1, 1], 'subsample_type': '1dconv',
        #   'bidirectional_sum_fwd_bwd': True}),
        # projection
        ({'rnn_type': 'blstm', 'n_projs': 64}),
        ({'rnn_type': 'lstm', 'n_projs': 64}),
        ({'rnn_type': 'blstm', 'bidirectional_sum_fwd_bwd': True}),
        # ({'rnn_type': 'blstm', 'n_projs': 64, 'bidirectional_sum_fwd_bwd': True}),
        ({'rnn_type': 'blstm', 'last_proj_dim': 256}),
        ({'rnn_type': 'blstm', 'n_projs': 64, 'last_proj_dim': 256}),
        ({'rnn_type': 'lstm', 'n_projs': 64, 'last_proj_dim': 256}),
        ({'rnn_type': 'blstm', 'bidirectional_sum_fwd_bwd': True, 'last_proj_dim': 256}),
        # ({'rnn_type': 'blstm', 'n_projs': 64, 'bidirectional_sum_fwd_bwd': True, 'last_proj_dim': 256}),
        # LC-BLSTM
        ({'rnn_type': 'blstm', 'chunk_size_left': -1, 'chunk_size_right': 40}),
        ({'rnn_type': 'blstm', 'chunk_size_left': 40, 'chunk_size_right': 40}),
        ({'rnn_type': 'blstm', 'bidirectional_sum_fwd_bwd': True, 'chunk_size_left': 40, 'chunk_size_right': 40}),
        # Multi-task
        ({'rnn_type': 'blstm', 'n_layers_sub1': 4}),
        ({'rnn_type': 'blstm', 'n_layers_sub1': 4, 'task_specific_layer': True}),
        ({'rnn_type': 'blstm', 'n_layers_sub1': 4, 'n_layers_sub2': 3}),
        ({'rnn_type': 'blstm', 'n_layers_sub1': 4, 'n_layers_sub2': 3, 'task_specific_layer': True}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    xmaxs = [40, 45] if args['chunk_size_left'] == -1 else [1600, 1655]
    device_id = -1
    for xmax in xmaxs:
        xs = np.random.randn(batch_size, xmax, args['input_dim']).astype(np.float32)
        xlens = torch.IntTensor([len(x) for x in xs])
        xs = pad_list([np2tensor(x, device_id).float() for x in xs], 0.)

        rnn = importlib.import_module('neural_sp.models.seq2seq.encoders.rnn')
        enc = rnn.RNNEncoder(**args)
        enc_out_dict = enc(xs, xlens, task='all')
        assert enc_out_dict['ys']['xs'].size(0) == batch_size
        assert enc_out_dict['ys']['xs'].size(1) == enc_out_dict['ys']['xlens'][0]
        if args['n_layers_sub1'] > 0:
            assert enc_out_dict['ys_sub1']['xs'].size(0) == batch_size
            assert enc_out_dict['ys_sub1']['xs'].size(1) == enc_out_dict['ys_sub1']['xlens'][0]
        if args['n_layers_sub2'] > 0:
            assert enc_out_dict['ys_sub2']['xs'].size(0) == batch_size
            assert enc_out_dict['ys_sub2']['xs'].size(1) == enc_out_dict['ys_sub2']['xlens'][0]
