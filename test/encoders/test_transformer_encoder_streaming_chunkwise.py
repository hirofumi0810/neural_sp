#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for chunkwise encoding in streaming Transformer/Conformer encoder."""

import importlib
import math
import numpy as np
import pytest
import torch

from neural_sp.models.torch_utils import (
    np2tensor,
    pad_list
)

np.random.seed(0)
torch.manual_seed(0)


def make_args_transformer(**kwargs):
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
        pe_type='add',
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


def make_args_conformer(**kwargs):
    args = dict(
        input_dim=80,
        enc_type='conv_conformer',
        n_heads=4,
        kernel_size=3,
        normalization='batch_norm',
        n_layers=3,
        n_layers_sub1=0,
        n_layers_sub2=0,
        d_model=8,
        d_ff=16,
        ffn_bottleneck_dim=0,
        ffn_activation='swish',
        pe_type='relative',
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
        # no CNN
        ({'enc_type': 'uni_transformer', 'chunk_size_current': "1"}),
        ({'enc_type': 'uni_transformer', 'chunk_size_current': "4"}),
        ({'enc_type': 'uni_conformer', 'chunk_size_current': "1", 'kernel_size': 3}),
        ({'enc_type': 'uni_conformer', 'chunk_size_current': "1", 'kernel_size': 7}),
        ({'enc_type': 'uni_conformer', 'chunk_size_current': "4", 'kernel_size': 7}),
        ({'enc_type': 'uni_conformer_v2', 'chunk_size_current': "1", 'kernel_size': 3}),
        ({'enc_type': 'uni_conformer_v2', 'chunk_size_current': "1", 'kernel_size': 7}),
        ({'enc_type': 'uni_conformer_v2', 'chunk_size_current': "4", 'kernel_size': 7}),
        # no CNN, frame stacking
        ({'enc_type': 'uni_transformer', 'n_stacks': 3, 'chunk_size_current': "3"}),
        # LC-Transformer
        ({'enc_type': 'transformer', 'streaming_type': 'reshape',
          'chunk_size_left': "8", 'chunk_size_current': "16", 'chunk_size_right': "8"}),
        ({'enc_type': 'transformer', 'streaming_type': 'mask',
          'chunk_size_left': "16", 'chunk_size_current': "16"}),
        # LC-Conformer
        ({'enc_type': 'conformer', 'streaming_type': 'reshape',
          'chunk_size_left': "8", 'chunk_size_current': "16", 'chunk_size_right': "8"}),
        ({'enc_type': 'conformer_v2', 'streaming_type': 'reshape',
          'chunk_size_left': "8", 'chunk_size_current': "16", 'chunk_size_right': "8"}),
        ({'enc_type': 'conformer', 'streaming_type': 'mask',
          'chunk_size_left': "16", 'chunk_size_current': "16"}),
        ({'enc_type': 'conformer_v2', 'streaming_type': 'mask',
          'chunk_size_left': "16", 'chunk_size_current': "16"}),
        # subsample: 1 / 2
        ({'enc_type': 'conv',
          'conv_channels': "32", 'conv_kernel_sizes': "(3,3)",
          'conv_strides': "(1,1)", 'conv_poolings': "(2,2)",
          'chunk_size_current': "2"}),
        ({'enc_type': 'conv_transformer', 'streaming_type': 'mask',
          'conv_channels': "32", 'conv_kernel_sizes': "(3,3)",
          'conv_strides': "(1,1)", 'conv_poolings': "(2,2)",
          'chunk_size_left': "16", 'chunk_size_current': "16"}),
        ({'enc_type': 'conv_transformer', 'streaming_type': 'reshape',
          'conv_channels': "32", 'conv_kernel_sizes': "(3,3)",
          'conv_strides': "(1,1)", 'conv_poolings': "(2,2)",
          'chunk_size_left': "8", 'chunk_size_current': "16", 'chunk_size_right': "8"}),
        # subsample: 1/4
        ({'enc_type': 'conv', 'chunk_size_current': "8"}),
        ({'enc_type': 'conv_uni_transformer', 'chunk_size_current': "8"}),
        ({'enc_type': 'conv_uni_conformer', 'chunk_size_current': "8"}),
        ({'enc_type': 'conv_uni_conformer_v2', 'chunk_size_current': "8"}),
        ({'enc_type': 'conv_uni_conformer_v2', 'chunk_size_current': "16",
          'subsample': "1_2_1"}),
        ({'enc_type': 'conv_transformer', 'streaming_type': 'mask',
          'chunk_size_left': "16", 'chunk_size_current': "16"}),
        ({'enc_type': 'conv_conformer', 'streaming_type': 'mask',
          'chunk_size_left': "16", 'chunk_size_current': "16"}),
        ({'enc_type': 'conv_conformer_v2', 'streaming_type': 'mask',
          'chunk_size_left': "16", 'chunk_size_current': "16"}),
        ({'enc_type': 'conv_transformer', 'streaming_type': 'reshape',
          'chunk_size_left': "8", 'chunk_size_current': "16", 'chunk_size_right': "8"}),
        ({'enc_type': 'conv_conformer', 'streaming_type': 'reshape',
          'chunk_size_left': "8", 'chunk_size_current': "16", 'chunk_size_right': "8"}),
        ({'enc_type': 'conv_conformer_v2', 'streaming_type': 'reshape',
          'chunk_size_left': "8", 'chunk_size_current': "16", 'chunk_size_right': "8"}),
        # subsample: 1/8
        ({'enc_type': 'conv',
          'conv_channels': "32_32_32", 'conv_kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'conv_strides': "(1,1)_(1,1)_(1,1)", 'conv_poolings': "(2,2)_(2,2)_(2,2)",
          'chunk_size_current': "16"}),
        ({'enc_type': 'conv_transformer', 'streaming_type': 'mask',
          'conv_channels': "32_32_32", 'conv_kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'conv_strides': "(1,1)_(1,1)_(1,1)", 'conv_poolings': "(2,2)_(2,2)_(2,2)",
          'chunk_size_left': "16", 'chunk_size_current': "16"}),
        ({'enc_type': 'conv_transformer', 'streaming_type': 'reshape',
          'conv_channels': "32_32_32", 'conv_kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'conv_strides': "(1,1)_(1,1)_(1,1)", 'conv_poolings': "(2,2)_(2,2)_(2,2)",
          'chunk_size_left': "8", 'chunk_size_current': "16", 'chunk_size_right': "8"}),
    ]
)
def test_forward_streaming_chunkwise(args):
    is_conformer = 'conformer' in args['enc_type']
    if is_conformer:
        args = make_args_conformer(**args)
    else:
        args = make_args_transformer(**args)
    unidir = 'uni' in args['enc_type']

    batch_size = 1
    xmaxs = [t for t in range(164, 192, 3)]
    device = "cpu"
    atol = 1e-05

    N_l = max(0, int(args['chunk_size_left'].split('_')[0])) // args['n_stacks']
    N_c = max(0, int(args['chunk_size_current'].split('_')[0])) // args['n_stacks']
    N_r = max(0, int(args['chunk_size_right'].split('_')[0])) // args['n_stacks']
    if unidir:
        args['chunk_size_left'] = "0"
        args['chunk_size_current'] = "0"
        args['chunk_size_right'] = "0"
    if is_conformer:
        module = importlib.import_module('neural_sp.models.seq2seq.encoders.conformer')
        enc = module.ConformerEncoder(**args).to(device)
    else:
        module = importlib.import_module('neural_sp.models.seq2seq.encoders.transformer')
        enc = module.TransformerEncoder(**args).to(device)

    if args['streaming_type'] == 'mask':
        N_l = 0
        # NOTE: context in previous chunks are cached inside the encoder

    factor = enc.subsampling_factor
    conv_context = enc.conv.context_size if (enc.conv is not None and not enc.lc_bidir) else 0
    assert N_c % factor == 0

    module_stack = importlib.import_module('neural_sp.models.seq2seq.frontends.frame_stacking')

    enc.eval()
    for xmax_orig in xmaxs:
        xs = np.random.randn(batch_size, xmax_orig, args['input_dim']).astype(np.float32)
        if args['n_stacks'] > 1:
            xs = [module_stack.stack_frame(x, args['n_stacks'], args['n_stacks']) for x in xs]
        else:
            # zero padding for the last chunk (CNN)
            if enc.streaming_type == 'mask' and 'conv' in args['enc_type'] and xmax_orig % N_c != 0:
                zero_pad = np.zeros((batch_size, N_c - xmax_orig % N_c, args['input_dim'])).astype(np.float32)
                xs = np.concatenate([xs, zero_pad], axis=1)
        xlens = torch.IntTensor([len(x) for x in xs])

        # all encoding
        xs_pad = pad_list([np2tensor(x, device).float() for x in xs], 0.)
        enc.reset_cache()
        enc_out_dict = enc(xs_pad, xlens, task='all')
        eout_all = enc_out_dict['ys']['xs']
        elens_all = enc_out_dict['ys']['xlens']
        assert eout_all.size(0) == batch_size
        assert eout_all.size(1) == elens_all.max()

        # chunk by chunk encoding
        eouts_chunk_cat = []
        elens_chunk_cat = 0
        xmax = xlens.max().item()
        n_chunks = math.ceil(xmax / N_c)
        j = 0  # time offset for input
        j_out = 0  # time offset for encoder output
        enc.reset_cache()
        for chunk_idx in range(n_chunks):
            start = j - N_l - conv_context
            end = (j + N_c + N_r) + conv_context
            xs_pad_chunk = pad_list(
                [np2tensor(x[max(0, start):end], device).float() for x in xs], 0.)
            if enc.streaming_type == 'reshape':
                xlens_chunk = torch.IntTensor([max(factor, min(xmax - j, N_c)) for x in xs])
            else:
                xlens_chunk = torch.IntTensor([max(factor, xs_pad_chunk.size(1)) for x in xs])
            if enc.streaming_type == 'reshape':  # ???
                # left padding
                if start < 0:
                    zero_pad = xs_pad.new_zeros(batch_size, -start, args['input_dim'])
                    xs_pad_chunk = torch.cat([zero_pad, xs_pad_chunk], dim=1)
                # right padding
                if end >= xmax:
                    zero_pad = xs_pad.new_zeros(batch_size, end - xmax, args['input_dim'])
                    xs_pad_chunk = torch.cat([xs_pad_chunk, zero_pad], dim=1)

            lookback = start >= 0 and conv_context > 0
            lookahead = end < xmax and conv_context > 0

            with torch.no_grad():
                enc_out_dict_chunk = enc(xs_pad_chunk, xlens_chunk, task='all',
                                         streaming=True,
                                         lookback=lookback,
                                         lookahead=lookahead)

            eout_all_i = eout_all[:, j_out:]
            if lookahead or conv_context == 0 or not unidir:
                eout_all_i = eout_all_i[:, :(N_c // factor)]
            # NOTE: in the last chunk, the rest frames are fed at once
            if eout_all_i.size(1) == 0:
                break

            eout_chunk = enc_out_dict_chunk['ys']['xs']
            elens_chunk = enc_out_dict_chunk['ys']['xlens']

            diff = eout_chunk.size(1) - eout_all_i.size(1)

            eout_chunk = eout_chunk[:, :eout_all_i.size(1)]
            elens_chunk -= diff

            eouts_chunk_cat.append(eout_chunk)
            elens_chunk_cat += elens_chunk

            j += N_c
            j_out += (N_c // factor)
            if j > xmax:
                break
            if not lookahead and conv_context > 0 and unidir:
                break

        eouts_chunk_cat = torch.cat(eouts_chunk_cat, dim=1)
        assert eout_all.size() == eouts_chunk_cat.size()
        assert torch.allclose(eout_all, eouts_chunk_cat, atol=atol)
        assert torch.equal(elens_all, elens_chunk_cat), (elens_all, elens_chunk_cat)
