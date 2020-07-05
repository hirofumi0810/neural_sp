#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for RNN encoder."""

import importlib
import math
import numpy as np
import pytest
import torch

from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


def make_args(**kwargs):
    args = dict(
        input_dim=80,
        rnn_type='blstm',
        n_units=64,
        n_projs=0,
        last_proj_dim=0,
        n_layers=4,
        n_layers_sub1=0,
        n_layers_sub2=0,
        dropout_in=0.1,
        dropout=0.1,
        subsample="1_1_1_1_1",
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
        chunk_size_left=0,
        chunk_size_right=0,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        # RNN type
        ({'rnn_type': 'blstm'}),
        ({'rnn_type': 'bgru'}),
        ({'rnn_type': 'lstm'}),
        ({'rnn_type': 'gru'}),
        # 2dCNN-RNN
        ({'rnn_type': 'conv_blstm'}),
        ({'rnn_type': 'conv_blstm', 'input_dim': 240, 'conv_in_channel': 3}),
        ({'rnn_type': 'conv_bgru'}),
        ({'rnn_type': 'conv_gru'}),
        # 1dCNN-RNN
        ({'rnn_type': 'conv_blstm',
          'conv_kernel_sizes': "3_3", 'conv_strides': "1_1", 'conv_poolings': "2_2"}),
        ({'rnn_type': 'conv_blstm',
          'conv_kernel_sizes': "3_3", 'conv_strides': "1_1", 'conv_poolings': "2_2",
          'input_dim': 240, 'conv_in_channel': 3}),
        ({'rnn_type': 'conv_bgru',
          'conv_kernel_sizes': "3_3", 'conv_strides': "1_1", 'conv_poolings': "2_2"}),
        ({'rnn_type': 'conv_gru',
          'conv_kernel_sizes': "3_3", 'conv_strides': "1_1", 'conv_poolings': "2_2"}),
        # normalization
        ({'rnn_type': 'conv_blstm', 'conv_batch_norm': True}),
        ({'rnn_type': 'conv_blstm', 'conv_layer_norm': True}),
        # projection
        ({'rnn_type': 'blstm', 'n_projs': 32}),
        ({'rnn_type': 'lstm', 'n_projs': 32}),
        ({'rnn_type': 'blstm', 'bidir_sum_fwd_bwd': True}),
        ({'rnn_type': 'blstm', 'bidir_sum_fwd_bwd': True, 'n_projs': 32}),
        ({'rnn_type': 'blstm', 'last_proj_dim': 10}),
        ({'rnn_type': 'blstm', 'last_proj_dim': 10, 'n_projs': 32}),
        ({'rnn_type': 'lstm', 'last_proj_dim': 10, 'n_projs': 32}),
        ({'rnn_type': 'blstm', 'bidir_sum_fwd_bwd': True, 'last_proj_dim': 10}),
        ({'rnn_type': 'blstm', 'bidir_sum_fwd_bwd': True, 'last_proj_dim': 10, 'n_projs': 32}),
        # subsampling
        ({'rnn_type': 'blstm', 'subsample': "1_2_2_1_1", 'subsample_type': 'drop'}),
        ({'rnn_type': 'blstm', 'subsample': "1_2_2_1_1", 'subsample_type': 'concat'}),
        ({'rnn_type': 'blstm', 'subsample': "1_2_2_1_1", 'subsample_type': 'max_pool'}),
        ({'rnn_type': 'blstm', 'subsample': "1_2_2_1_1", 'subsample_type': '1dconv'}),
        ({'rnn_type': 'blstm', 'subsample': "1_2_2_1_1", 'subsample_type': 'drop',
          'bidir_sum_fwd_bwd': True}),
        ({'rnn_type': 'blstm', 'subsample': "1_2_2_1_1", 'subsample_type': 'concat',
          'bidir_sum_fwd_bwd': True}),
        ({'rnn_type': 'blstm', 'subsample': "1_2_2_1_1", 'subsample_type': 'max_pool',
          'bidir_sum_fwd_bwd': True}),
        ({'rnn_type': 'blstm', 'subsample': "1_2_2_1_1", 'subsample_type': '1dconv',
          'bidir_sum_fwd_bwd': True}),
        ({'rnn_type': 'blstm', 'subsample': "1_2_2_1_1", 'subsample_type': 'drop',
          'n_projs': 32}),
        ({'rnn_type': 'blstm', 'subsample': "1_2_2_1_1", 'subsample_type': 'concat',
          'n_projs': 32}),
        ({'rnn_type': 'blstm', 'subsample': "1_2_2_1_1", 'subsample_type': 'max_pool',
          'n_projs': 32}),
        ({'rnn_type': 'blstm', 'subsample': "1_2_2_1_1", 'subsample_type': '1dconv',
          'n_projs': 32}),
        # LC-BLSTM
        ({'rnn_type': 'blstm', 'chunk_size_right': 40}),  # for PT
        ({'rnn_type': 'blstm', 'chunk_size_left': 40, 'chunk_size_right': 40}),
        ({'rnn_type': 'blstm', 'bidir_sum_fwd_bwd': True,
          'chunk_size_right': 40}),  # for PT
        ({'rnn_type': 'blstm', 'bidir_sum_fwd_bwd': True,
          'chunk_size_left': 40, 'chunk_size_right': 40}),
        ({'rnn_type': 'blstm', 'bidir_sum_fwd_bwd': True,
          'chunk_size_left': 40, 'chunk_size_right': 40,
          'conv_poolings': "(2,1)_(2,1)"}),
        ({'rnn_type': 'blstm', 'bidir_sum_fwd_bwd': True,
          'chunk_size_left': 40, 'chunk_size_right': 40,
          'conv_poolings': "(1,2)_(1,2)"}),
        ({'rnn_type': 'blstm', 'bidir_sum_fwd_bwd': True,
          'chunk_size_left': 40, 'chunk_size_right': 40,
          'conv_poolings': "(1,1)_(1,1)"}),
        # Multi-task
        ({'rnn_type': 'blstm', 'n_layers_sub1': 3}),
        ({'rnn_type': 'blstm', 'n_layers_sub1': 3, 'n_layers_sub2': 2}),
        ({'rnn_type': 'blstm', 'n_layers_sub1': 3, 'n_layers_sub2': 2, 'task_specific_layer': True}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    xmaxs = [40, 45] if args['chunk_size_left'] == -1 else [400, 455]
    device = "cpu"

    module = importlib.import_module('neural_sp.models.seq2seq.encoders.rnn')
    enc = module.RNNEncoder(**args)
    enc = enc.to(device)

    for xmax in xmaxs:
        xs = np.random.randn(batch_size, xmax, args['input_dim']).astype(np.float32)
        xlens = torch.IntTensor([len(x) - i * enc.subsampling_factor for i, x in enumerate(xs)])
        xs = pad_list([np2tensor(x, device).float() for x in xs], 0.)
        enc_out_dict = enc(xs, xlens, task='all')

        assert enc_out_dict['ys']['xs'].size(0) == batch_size
        assert enc_out_dict['ys']['xs'].size(1) == enc_out_dict['ys']['xlens'].max()
        for b in range(batch_size):
            if 'conv' in args['rnn_type'] or args['subsample_type'] in ['max_pool', '1dconv']:
                assert enc_out_dict['ys']['xlens'][b].item() == math.ceil(xlens[b].item() / enc.subsampling_factor)
            else:
                assert enc_out_dict['ys']['xlens'][b].item() == math.floor(xlens[b].item() / enc.subsampling_factor)
        if args['n_layers_sub1'] > 0:
            # all outputs
            assert enc_out_dict['ys_sub1']['xs'].size(0) == batch_size
            assert enc_out_dict['ys_sub1']['xs'].size(1) == enc_out_dict['ys_sub1']['xlens'].max()
            for b in range(batch_size):
                if 'conv' in args['rnn_type'] or args['subsample_type'] in ['max_pool', '1dconv']:
                    assert enc_out_dict['ys_sub1']['xlens'][b].item() == math.ceil(
                        xlens[b].item() / enc.subsampling_factor)
                else:
                    assert enc_out_dict['ys_sub1']['xlens'][b].item() == math.floor(
                        xlens[b].item() / enc.subsampling_factor)
            # single output
            enc_out_dict_sub1 = enc(xs, xlens, task='ys_sub1')
            assert enc_out_dict_sub1['ys_sub1']['xs'].size(0) == batch_size
            assert enc_out_dict_sub1['ys_sub1']['xs'].size(1) == enc_out_dict['ys_sub1']['xlens'].max()

        if args['n_layers_sub2'] > 0:
            # all outputs
            assert enc_out_dict['ys_sub2']['xs'].size(0) == batch_size
            assert enc_out_dict['ys_sub2']['xs'].size(1) == enc_out_dict['ys_sub2']['xlens'].max()
            for b in range(batch_size):
                if 'conv' in args['rnn_type'] or args['subsample_type'] in ['max_pool', '1dconv']:
                    assert enc_out_dict['ys_sub2']['xlens'][b].item() == math.ceil(
                        xlens[b].item() / enc.subsampling_factor)
                else:
                    assert enc_out_dict['ys_sub2']['xlens'][b].item() == math.floor(
                        xlens[b].item() / enc.subsampling_factor)
            # single output
            enc_out_dict_sub12 = enc(xs, xlens, task='ys_sub2')
            assert enc_out_dict_sub12['ys_sub2']['xs'].size(0) == batch_size
            assert enc_out_dict_sub12['ys_sub2']['xs'].size(1) == enc_out_dict_sub12['ys_sub2']['xlens'].max()
