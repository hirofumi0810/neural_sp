#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for chunkwise encoding in streaming RNN encoder."""

import importlib
import math
import numpy as np
import pytest
import torch

from neural_sp.models.torch_utils import (
    np2tensor,
    pad_list
)


def make_args(**kwargs):
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
        chunk_size_left="0",
        chunk_size_right="0",
        rsp_prob=0,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        # no CNN
        ({'enc_type': 'blstm', 'chunk_size_left': "20", 'chunk_size_right': "20"}),
        ({'enc_type': 'blstm', 'chunk_size_left': "32", 'chunk_size_right': "16"}),
        ({'enc_type': 'lstm', 'chunk_size_left': "1"}),  # unidirectional
        ({'enc_type': 'lstm', 'chunk_size_left': "40"}),  # unidirectional
        # no CNN, frame stacking
        ({'enc_type': 'blstm', 'n_stacks': 2,
          'chunk_size_left': "20", 'chunk_size_right': "20"}),
        ({'enc_type': 'blstm', 'n_stacks': 2,
          'chunk_size_left': "32", 'chunk_size_right': "16"}),
        ({'enc_type': 'lstm', 'n_stacks': 2, 'chunk_size_left': "2"}),
        ({'enc_type': 'lstm', 'n_stacks': 3, 'chunk_size_left': "3"}),
        # subsample: 1/2
        ({'enc_type': 'conv',
          'conv_channels': "32", 'conv_kernel_sizes': "(3,3)",
          'conv_strides': "(1,1)", 'conv_poolings': "(2,2)",
          'chunk_size_left': "4"}),
        ({'enc_type': 'conv',
          'conv_channels': "32", 'conv_kernel_sizes': "(3,3)",
          'conv_strides': "(1,1)", 'conv_poolings': "(2,2)",
          'chunk_size_left': "32"}),
        # subsample: 1/4
        ({'enc_type': 'conv', 'chunk_size_left': "8"}),
        ({'enc_type': 'conv', 'chunk_size_left': "32"}),
        ({'enc_type': 'conv_lstm', 'chunk_size_left': "8"}),  # unidirectional
        ({'enc_type': 'conv_lstm', 'chunk_size_left': "40"}),  # unidirectional
        ({'enc_type': 'conv_blstm',
          'chunk_size_left': "20", 'chunk_size_right': "20"}),
        ({'enc_type': 'conv_blstm',
          'chunk_size_left': "32", 'chunk_size_right': "16"}),
        # subsample: 1/8
        ({'enc_type': 'conv',
          'conv_channels': "32_32_32", 'conv_kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'conv_strides': "(1,1)_(1,1)_(1,1)", 'conv_poolings': "(2,2)_(2,2)_(2,2)",
          'chunk_size_left': "16"}),
        ({'enc_type': 'conv',
          'conv_channels': "32_32_32", 'conv_kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'conv_strides': "(1,1)_(1,1)_(1,1)", 'conv_poolings': "(2,2)_(2,2)_(2,2)",
          'chunk_size_left': "32"}),
        ({'enc_type': 'conv_blstm',
          'conv_channels': "32_32_32", 'conv_kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'conv_strides': "(1,1)_(1,1)_(1,1)", 'conv_poolings': "(2,2)_(2,2)_(2,2)",
          'chunk_size_left': "32", 'chunk_size_right': "16"}),
        ({'enc_type': 'conv_blstm',
          'conv_channels': "32_32_32", 'conv_kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'conv_strides': "(1,1)_(1,1)_(1,1)", 'conv_poolings': "(2,2)_(2,2)_(2,2)",
          'chunk_size_left': "64", 'chunk_size_right': "32"}),
    ]
)
def test_forward_streaming_chunkwise(args):
    args = make_args(**args)
    unidir = args['enc_type'] in ['conv_lstm', 'conv_gru', 'lstm', 'gru']

    batch_size = 1
    xmaxs = [t for t in range(160, 192, 1)]
    device = "cpu"
    atol = 1e-05

    N_l = max(0, int(args['chunk_size_left'].split('_')[0])) // args['n_stacks']
    N_r = max(0, int(args['chunk_size_right'].split('_')[0])) // args['n_stacks']
    if unidir:
        args['chunk_size_left'] = "0"
        args['chunk_size_right'] = "0"
    module = importlib.import_module('neural_sp.models.seq2seq.encoders.rnn')
    enc = module.RNNEncoder(**args)
    enc = enc.to(device)
    if enc.lc_bidir:
        assert N_l > 0

    factor = enc.subsampling_factor
    conv_lookahead = enc.conv.context_size if enc.conv is not None else 0

    module_fs = importlib.import_module('neural_sp.models.seq2seq.frontends.frame_stacking')

    if enc.conv is not None:
        enc.turn_off_ceil_mode(enc)

    enc.eval()
    with torch.no_grad():
        for xmax in xmaxs:
            xs = np.random.randn(batch_size, xmax, args['input_dim']).astype(np.float32)

            if args['n_stacks'] > 1:
                xs = [module_fs.stack_frame(x, args['n_stacks'], args['n_stacks']) for x in xs]
            else:
                # zero padding for the last chunk
                if xmax % N_l != 0:
                    zero_pad = np.zeros((batch_size, N_l - xmax % N_l, args['input_dim'])).astype(np.float32)
                    xs = np.concatenate([xs, zero_pad], axis=1)

            xlens = torch.IntTensor([len(x) for x in xs])
            xmax = xlens.max().item()

            # all encoding
            xs_pad = pad_list([np2tensor(x, device).float() for x in xs], 0.)

            enc_out_dict = enc(xs_pad, xlens, task='all')
            assert enc_out_dict['ys']['xs'].size(0) == batch_size
            assert enc_out_dict['ys']['xs'].size(1) == enc_out_dict['ys']['xlens'][0]

            enc.reset_cache()

            # chunk by chunk encoding
            eouts_stream = []
            elens_stream = 0
            n_chunks = math.ceil(xmax / N_l)
            j = 0  # time offset for input
            j_out = 0  # time offset for encoder output
            for chunk_idx in range(n_chunks):
                start = j - conv_lookahead
                end = (j + N_l + N_r) + conv_lookahead
                xs_pad_stream = pad_list(
                    [np2tensor(x[max(0, start):end], device).float() for x in xs], 0.)
                xlens_stream = torch.IntTensor([xs_pad_stream.size(1) for x in xs])
                enc_out_dict_stream = enc(xs_pad_stream, xlens_stream, task='all',
                                          streaming=True,
                                          lookback=start >= 0,
                                          lookahead=end <= xmax - 1)

                eout_all_i = enc_out_dict['ys']['xs'][:, j_out:j_out + (N_l // factor)]
                if eout_all_i.size(1) == 0:
                    break
                eout_stream_i = enc_out_dict_stream['ys']['xs']
                elens_stream_i = enc_out_dict_stream['ys']['xlens']
                diff = eout_stream_i.size(1) - eout_all_i.size(1)
                eout_stream_i = eout_stream_i[:, :eout_all_i.size(1)]
                elens_stream_i -= diff
                for t in range(eout_stream_i.size(1)):
                    print(torch.allclose(eout_all_i[:, t], eout_stream_i[:, t], atol=atol))

                eouts_stream.append(eout_stream_i)
                elens_stream += elens_stream_i

                j += N_l
                j_out += (N_l // factor)
                if j > xmax:
                    break

            enc.reset_cache()

            eouts_stream = torch.cat(eouts_stream, dim=1)
            assert enc_out_dict['ys']['xs'].size() == eouts_stream.size()
            assert torch.allclose(enc_out_dict['ys']['xs'], eouts_stream, atol=atol)
            assert elens_stream.item() == eouts_stream.size(1)
            assert torch.equal(enc_out_dict['ys']['xlens'], elens_stream)
