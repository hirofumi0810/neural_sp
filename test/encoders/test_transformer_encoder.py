#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for Transformer encoder."""

import importlib
import numpy as np
import pytest
import torch

from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list

np.random.seed(0)
torch.manual_seed(0)


def make_args(**kwargs):
    args = dict(
        input_dim=80,
        enc_type='conv_transformer',
        n_heads=4,
        n_layers=3,
        n_layers_sub1=0,
        n_layers_sub2=0,
        d_model=8,
        d_ff=16,
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


@pytest.mark.parametrize(
    "args",
    [
        ({'enc_type': 'transformer'}),
        # 2dCNN-Transformer
        ({'enc_type': 'conv_transformer'}),
        ({'input_dim': 240, 'conv_in_channel': 3}),
        # 1dCNN-Transformer
        ({'conv_kernel_sizes': "3_3", 'conv_strides': "1_1", 'conv_poolings': "2_2"}),
        ({'conv_kernel_sizes': "3_3", 'conv_strides': "1_1", 'conv_poolings': "2_2",
          'input_dim': 240, 'conv_in_channel': 3}),
        # positional encoding
        ({'pe_type': 'add'}),
        ({'pe_type': 'relative'}),
        ({'pe_type': 'relative_xl'}),
        ({'pe_type': 'relative', 'clamp_len': 10}),
        ({'pe_type': 'relative_xl', 'clamp_len': 10}),
        # normalization
        ({'conv_batch_norm': True}),
        ({'conv_layer_norm': True}),
        # projection
        ({'last_proj_dim': 10}),
        # unidirectional
        ({'enc_type': 'conv_uni_transformer'}),
        ({'enc_type': 'conv_uni_transformer', 'lookahead': "1_1_1"}),
        ({'enc_type': 'conv_uni_transformer', 'lookahead': "1_0_1"}),
        ({'enc_type': 'conv_uni_transformer', 'lookahead': "0_1_0"}),
        # LC-Transformer
        ({'streaming_type': 'reshape',
          'chunk_size_left': "64", 'chunk_size_current': "64", 'chunk_size_right': "32"}),
        ({'streaming_type': 'reshape',
          'chunk_size_left': "64", 'chunk_size_current': "128", 'chunk_size_right': "64"}),
        ({'streaming_type': 'reshape',
          'chunk_size_left': "64", 'chunk_size_current': "128", 'chunk_size_right': "64",
          'pe_type': 'relative'}),
        ({'streaming_type': 'mask',
          'chunk_size_left': "64", 'chunk_size_current': "64"}),
        ({'streaming_type': 'mask',
          'chunk_size_left': "64", 'chunk_size_current': "64",
          'pe_type': 'relative'}),
        # Multi-task
        ({'n_layers_sub1': 2}),
        ({'n_layers_sub1': 2, 'n_layers_sub2': 1}),
        ({'n_layers_sub1': 2, 'n_layers_sub2': 1, 'last_proj_dim': 10}),
        ({'n_layers_sub1': 2, 'n_layers_sub2': 1, 'task_specific_layer': True}),
        ({'n_layers_sub1': 2, 'n_layers_sub2': 1, 'task_specific_layer': True,
          'last_proj_dim': 10}),
        # bottleneck
        ({'ffn_bottleneck_dim': 16}),
        # subsampling
        ({'subsample': "1_2_1", 'subsample_type': 'drop'}),
        ({'subsample': "1_2_1", 'subsample_type': 'concat'}),
        ({'subsample': "1_2_1", 'subsample_type': 'max_pool'}),
        ({'subsample': "1_2_1", 'subsample_type': 'conv1d'}),
        ({'subsample': "1_2_1", 'subsample_type': 'add'}),
        ({'subsample': "1_2_1", 'subsample_type': 'max_pool', 'pe_type': 'relative'}),
        ({'subsample': "1_2_1", 'enc_type': 'conv_uni_transformer'}),
        ({'subsample': "1_2_1", 'streaming_type': 'reshape',
          'chunk_size_left': "64", 'chunk_size_current': "64", 'chunk_size_right': "32"}),
        ({'subsample': "1_2_1", 'streaming_type': 'mask',
          'chunk_size_left': "64", 'chunk_size_current': "64"}),
        ({'subsample': "1_2_1", 'streaming_type': 'reshape',
          'conv_poolings': "(1,1)_(2,2)",
          'chunk_size_left': "64", 'chunk_size_current': "64", 'chunk_size_right': "32"}),
        ({'subsample': "1_2_1", 'streaming_type': 'mask',
          'conv_poolings': "(1,1)_(2,2)",
          'chunk_size_left': "64", 'chunk_size_current': "64"}),
        ({'subsample': "2_2_1", 'streaming_type': 'reshape',
          'conv_poolings': "(1,1)_(2,2)",
          'chunk_size_left': "64", 'chunk_size_current': "64", 'chunk_size_right': "32"}),
        ({'subsample': "2_2_1", 'streaming_type': 'mask',
          'conv_poolings': "(1,1)_(2,2)",
          'chunk_size_left': "64", 'chunk_size_current': "64"}),
        ({'subsample': "2_2_1", 'streaming_type': 'reshape',
          'pe_type': "relative",
          'chunk_size_left': "64", 'chunk_size_current': "64", 'chunk_size_right': "32"}),
        ({'subsample': "2_2_1", 'streaming_type': 'mask',
          'pe_type': "relative",
          'chunk_size_left': "64", 'chunk_size_current': "64"}),
    ]
)
def test_forward(args):
    args = make_args(**args)

    batch_size = 4
    xmaxs = [40, 45] if int(args['chunk_size_left'].split('_')[0]) == -1 else [400, 455]
    device = "cpu"

    module = importlib.import_module('neural_sp.models.seq2seq.encoders.transformer')
    enc = module.TransformerEncoder(**args)
    enc = enc.to(device)

    for xmax in xmaxs:
        xs = np.random.randn(batch_size, xmax, args['input_dim']).astype(np.float32)
        xlens = torch.IntTensor([len(x) - i * enc.subsampling_factor for i, x in enumerate(xs)])
        xs = pad_list([np2tensor(x, device).float() for x in xs], 0.)

        # for mode in ['train', 'eval']:  # too slow
        for mode in ['train']:
            if mode == 'train':
                enc.train()
                enc_out_dict = enc(xs, xlens, task='all')
            elif mode == 'eval':
                enc.eval()
                with torch.no_grad():
                    enc_out_dict = enc(xs, xlens, task='all')
                    # enc._plot_attention()  # too slow

            assert enc_out_dict['ys']['xs'].size(0) == batch_size
            assert enc_out_dict['ys']['xs'].size(1) == enc_out_dict['ys']['xlens'][0]
            if args['n_layers_sub1'] > 0:
                assert enc_out_dict['ys_sub1']['xs'].size(0) == batch_size
                assert enc_out_dict['ys_sub1']['xs'].size(1) == enc_out_dict['ys_sub1']['xlens'][0]
            if args['n_layers_sub2'] > 0:
                assert enc_out_dict['ys_sub2']['xs'].size(0) == batch_size
                assert enc_out_dict['ys_sub2']['xs'].size(1) == enc_out_dict['ys_sub2']['xlens'][0]
