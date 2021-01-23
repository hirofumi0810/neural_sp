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
        enc_type='blstm',
        n_units=16,
        n_projs=0,
        last_proj_dim=0,
        n_layers=4,
        n_layers_sub1=0,
        n_layers_sub2=0,
        dropout_in=0.1,
        dropout=0.1,
        subsample="1_1_1_1",
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
        cnn_lookahead=True,
        rsp_prob=0,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        # RNN type
        ({'enc_type': 'blstm'}),
        ({'enc_type': 'bgru'}),
        ({'enc_type': 'lstm'}),
        ({'enc_type': 'lstm', 'rsp_prob': 0.5}),
        ({'enc_type': 'gru'}),
        ({'enc_type': 'gru', 'rsp_prob': 0.5}),
        # 2dCNN-RNN
        ({'enc_type': 'conv_blstm'}),
        ({'enc_type': 'conv_blstm', 'input_dim': 240, 'conv_in_channel': 3}),
        ({'enc_type': 'conv_bgru'}),
        ({'enc_type': 'conv_gru'}),
        # 1dCNN-RNN
        ({'enc_type': 'conv_blstm',
          'conv_kernel_sizes': "3_3", 'conv_strides': "1_1", 'conv_poolings': "2_2"}),
        ({'enc_type': 'conv_blstm',
          'conv_kernel_sizes': "3_3", 'conv_strides': "1_1", 'conv_poolings': "2_2",
          'input_dim': 240, 'conv_in_channel': 3}),
        ({'enc_type': 'conv_bgru',
          'conv_kernel_sizes': "3_3", 'conv_strides': "1_1", 'conv_poolings': "2_2"}),
        ({'enc_type': 'conv_gru',
          'conv_kernel_sizes': "3_3", 'conv_strides': "1_1", 'conv_poolings': "2_2"}),
        # normalization
        ({'enc_type': 'conv_blstm', 'conv_batch_norm': True}),
        ({'enc_type': 'conv_blstm', 'conv_layer_norm': True}),
        # projection
        ({'enc_type': 'blstm', 'n_projs': 8}),
        ({'enc_type': 'lstm', 'n_projs': 8}),
        ({'enc_type': 'blstm', 'bidir_sum_fwd_bwd': True}),
        ({'enc_type': 'blstm', 'bidir_sum_fwd_bwd': True, 'n_projs': 8}),
        ({'enc_type': 'blstm', 'last_proj_dim': 5}),
        ({'enc_type': 'blstm', 'last_proj_dim': 5, 'n_projs': 8}),
        ({'enc_type': 'lstm', 'last_proj_dim': 5, 'n_projs': 8}),
        ({'enc_type': 'blstm', 'bidir_sum_fwd_bwd': True, 'last_proj_dim': 5}),
        ({'enc_type': 'blstm', 'bidir_sum_fwd_bwd': True, 'last_proj_dim': 5, 'n_projs': 8}),
        # subsampling
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'drop'}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'concat'}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'max_pool'}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': '1dconv'}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'add'}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'drop',
          'bidir_sum_fwd_bwd': True}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'concat',
          'bidir_sum_fwd_bwd': True}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'max_pool',
          'bidir_sum_fwd_bwd': True}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': '1dconv',
          'bidir_sum_fwd_bwd': True}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'add',
          'bidir_sum_fwd_bwd': True}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'drop',
          'n_projs': 8}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'concat',
          'n_projs': 8}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'max_pool',
          'n_projs': 8}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': '1dconv',
          'n_projs': 8}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'add',
          'n_projs': 8}),
        # LC-BLSTM
        ({'enc_type': 'blstm', 'chunk_size_current': "0", 'chunk_size_right': "40"}),  # BLSTM for PT
        ({'enc_type': 'blstm', 'chunk_size_current': "40", 'chunk_size_right': "40"}),
        ({'enc_type': 'blstm', 'bidir_sum_fwd_bwd': True,
          'chunk_size_current': "0", 'chunk_size_right': "40"}),  # BLSTM for PT
        ({'enc_type': 'blstm', 'bidir_sum_fwd_bwd': True,
          'chunk_size_current': "40", 'chunk_size_right': "40"}),
        ({'enc_type': 'conv_blstm', 'bidir_sum_fwd_bwd': True,
          'chunk_size_current': "40", 'chunk_size_right': "40"}),
        ({'enc_type': 'conv_blstm', 'bidir_sum_fwd_bwd': True,
          'chunk_size_current': "40", 'chunk_size_right': "40", 'rsp_prob': 0.5}),
        # LC-BLSTM + subsampling
        ({'enc_type': 'blstm', 'subsample': "1_2_1_1",
          'chunk_size_right': "40"}),  # BLSTM for PT
        ({'enc_type': 'blstm', 'subsample': "1_2_1_1",
          'chunk_size_current': "40", 'chunk_size_right': "40"}),
        ({'enc_type': 'blstm', 'subsample': "1_2_1_1", 'bidir_sum_fwd_bwd': True,
          'chunk_size_right': "40"}),  # BLSTM for PT
        ({'enc_type': 'blstm', 'subsample': "1_2_1_1", 'bidir_sum_fwd_bwd': True,
          'chunk_size_current': "40", 'chunk_size_right': "40"}),
        ({'enc_type': 'blstm', 'subsample': "1_2_1_1", 'bidir_sum_fwd_bwd': True,
          'chunk_size_current': "40", 'chunk_size_right': "40", 'rsp_prob': 0.5}),
        # Multi-task
        ({'enc_type': 'blstm', 'n_layers_sub1': 2}),
        ({'enc_type': 'blstm', 'n_layers_sub1': 2, 'task_specific_layer': True}),
        ({'enc_type': 'blstm', 'n_layers_sub1': 2, 'task_specific_layer': True,
          'chunk_size_current': "0", 'chunk_size_right': "40"}),  # BLSTM for PT
        ({'enc_type': 'blstm', 'n_layers_sub1': 2, 'task_specific_layer': True,
          'chunk_size_current': "40", 'chunk_size_right': "40"}),  # LC-BLSTM
        ({'enc_type': 'blstm', 'n_layers_sub1': 2, 'task_specific_layer': True,
          'chunk_size_current': "0", 'chunk_size_right': "40",
          'rsp_prob': 0.5}),  # BLSTM for PT
        ({'enc_type': 'blstm', 'n_layers_sub1': 2, 'task_specific_layer': True,
          'chunk_size_current': "40", 'chunk_size_right': "40",
          'rsp_prob': 0.5}),  # LC-BLSTM
        ({'enc_type': 'blstm', 'n_layers_sub1': 2, 'n_layers_sub2': 1}),
        ({'enc_type': 'blstm', 'n_layers_sub1': 2, 'n_layers_sub2': 1,
          'task_specific_layer': True}),
        # Multi-task + subsampling
        ({'enc_type': 'blstm', 'subsample': "2_1_1_1", 'n_layers_sub1': 2,
          'chunk_size_current': "0", 'chunk_size_right': "40",
          'task_specific_layer': True}),  # BLSTM for PT
        ({'enc_type': 'blstm', 'subsample': "2_1_1_1", 'n_layers_sub1': 2,
          'chunk_size_current': "40", 'chunk_size_right': "40",
          'task_specific_layer': True}),  # LC-BLSTM
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    xmaxs = [40, 45] if int(args['chunk_size_current'].split('_')[0]) == -1 else [400, 455]
    device = "cpu"

    module = importlib.import_module('neural_sp.models.seq2seq.encoders.rnn')
    enc = module.RNNEncoder(**args).to(device)

    for xmax in xmaxs:
        xs = np.random.randn(batch_size, xmax, args['input_dim']).astype(np.float32)
        xlens = torch.IntTensor([len(x) - i * enc.subsampling_factor for i, x in enumerate(xs)])

        # shuffle
        perm_ids = torch.randperm(batch_size)
        xs = xs[perm_ids]
        xlens = xlens[perm_ids]

        xs = pad_list([np2tensor(x, device).float() for x in xs], 0.)
        enc_out_dict = enc(xs, xlens, task='all')

        assert enc_out_dict['ys']['xs'].size(0) == batch_size
        assert enc_out_dict['ys']['xs'].size(1) == enc_out_dict['ys']['xlens'].max()
        for b in range(batch_size):
            if 'conv' in args['enc_type'] or args['subsample_type'] in ['max_pool', '1dconv', 'drop', 'add']:
                assert enc_out_dict['ys']['xlens'][b].item() == math.ceil(xlens[b].item() / enc.subsampling_factor)
            else:
                assert enc_out_dict['ys']['xlens'][b].item() == xlens[b].item() // enc.subsampling_factor

        if args['n_layers_sub1'] > 0:
            # all outputs
            assert enc_out_dict['ys_sub1']['xs'].size(0) == batch_size
            assert enc_out_dict['ys_sub1']['xs'].size(1) == enc_out_dict['ys_sub1']['xlens'].max()
            for b in range(batch_size):
                if 'conv' in args['enc_type'] or args['subsample_type'] in ['max_pool', '1dconv', 'drop', 'add']:
                    assert enc_out_dict['ys_sub1']['xlens'][b].item() == math.ceil(
                        xlens[b].item() / enc.subsampling_factor_sub1)
                else:
                    assert enc_out_dict['ys_sub1']['xlens'][b].item() == xlens[b].item() // enc.subsampling_factor_sub1
            # single output
            enc_out_dict_sub1 = enc(xs, xlens, task='ys_sub1')
            assert enc_out_dict_sub1['ys_sub1']['xs'].size(0) == batch_size
            assert enc_out_dict_sub1['ys_sub1']['xs'].size(1) == enc_out_dict['ys_sub1']['xlens'].max()

        if args['n_layers_sub2'] > 0:
            # all outputs
            assert enc_out_dict['ys_sub2']['xs'].size(0) == batch_size
            assert enc_out_dict['ys_sub2']['xs'].size(1) == enc_out_dict['ys_sub2']['xlens'].max()
            for b in range(batch_size):
                if 'conv' in args['enc_type'] or args['subsample_type'] in ['max_pool', '1dconv', 'drop', 'add']:
                    assert enc_out_dict['ys_sub2']['xlens'][b].item() == math.ceil(
                        xlens[b].item() / enc.subsampling_factor_sub2)
                else:
                    assert enc_out_dict['ys_sub2']['xlens'][b].item() == xlens[b].item() // enc.subsampling_factor_sub2
            # single output
            enc_out_dict_sub2 = enc(xs, xlens, task='ys_sub2')
            assert enc_out_dict_sub2['ys_sub2']['xs'].size(0) == batch_size
            assert enc_out_dict_sub2['ys_sub2']['xs'].size(1) == enc_out_dict_sub2['ys_sub2']['xlens'].max()
