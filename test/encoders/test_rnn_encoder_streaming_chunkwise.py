#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for chunkwise encoding in streaming RNN encoder."""

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
        n_units=128,
        n_projs=0,
        last_proj_dim=0,
        n_layers=5,
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
        chunk_size_left=-1,
        chunk_size_right=-1,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        # no CNN
        ({'rnn_type': 'blstm', 'chunk_size_left': 20, 'chunk_size_right': 20}),
        ({'rnn_type': 'blstm', 'chunk_size_left': 32, 'chunk_size_right': 16}),
        # no CNN, frame stacking
        ({'rnn_type': 'blstm', 'n_stacks': 2,
          'chunk_size_left': 20, 'chunk_size_right': 20}),
        ({'rnn_type': 'blstm', 'n_stacks': 2,
          'chunk_size_left': 32, 'chunk_size_right': 16}),
        # subsample: 1/2
        ({'rnn_type': 'conv',
          'conv_channels': "32", 'conv_kernel_sizes': "(3,3)",
          'conv_strides': "(1,1)", 'conv_poolings': "(2, 2)",
          'chunk_size_left': 20, 'chunk_size_right': 10}),
        ({'rnn_type': 'conv',
          'conv_channels': "32", 'conv_kernel_sizes': "(3,3)",
          'conv_strides': "(1,1)", 'conv_poolings': "(2, 2)",
          'chunk_size_left': 32, 'chunk_size_right': 16}),
        # subsample: 1/4
        ({'rnn_type': 'conv', 'chunk_size_left': 20, 'chunk_size_right': 10}),
        ({'rnn_type': 'conv', 'chunk_size_left': 32, 'chunk_size_right': 16}),
        ({'rnn_type': 'conv_blstm', 'chunk_size_left': 20, 'chunk_size_right': 20}),
        ({'rnn_type': 'conv_blstm', 'chunk_size_left': 32, 'chunk_size_right': 16}),
        # subsample: 1/8
        ({'rnn_type': 'conv',
          'conv_channels': "32_32_32", 'conv_kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'conv_strides': "(1,1)_(1,1)_(1,1)", 'conv_poolings': "(2, 2)_(2, 2)_(2, 2)",
          'chunk_size_left': 32, 'chunk_size_right': 16}),
        ({'rnn_type': 'conv',
          'conv_channels': "32_32_32", 'conv_kernel_sizes': "(3,3)_(3,3)_(3,3)",
          'conv_strides': "(1,1)_(1,1)_(1,1)", 'conv_poolings': "(2, 2)_(2, 2)_(2, 2)",
          'chunk_size_left': 64, 'chunk_size_right': 32}),
    ]
)
def test_forward_streaming_chunkwise(args):
    args = make_args(**args)
    assert args['chunk_size_left'] > 0

    batch_size = 1
    xmaxs = [t for t in range(160, 192)]
    device_id = -1
    module = importlib.import_module('neural_sp.models.seq2seq.encoders.rnn')
    N_l = args['chunk_size_left'] // args['n_stacks']
    N_r = args['chunk_size_right'] // args['n_stacks']
    enc = module.RNNEncoder(**args)

    factor = enc.subsampling_factor
    lookback = enc.conv.n_frames_context if enc.conv is not None else 0
    lookahead = enc.conv.n_frames_context if enc.conv is not None else 0

    module_fs = importlib.import_module('neural_sp.models.seq2seq.frontends.frame_stacking')

    if enc.conv is not None:
        enc.turn_off_ceil_mode(enc)

    enc.eval()
    with torch.no_grad():
        for xmax in xmaxs:
            xs = np.random.randn(batch_size, xmax, args['input_dim']).astype(np.float32)

            if args['n_stacks'] > 1:
                xs = [module_fs.stack_frame(x, args['n_stacks'], args['n_stacks']) for x in xs]

            xlens = torch.IntTensor([len(x) for x in xs])
            xmax = xlens.max().item()

            # all inputs
            xs_pad = pad_list([np2tensor(x, device_id).float() for x in xs], 0.)

            enc_out_dict = enc(xs_pad, xlens, task='all')
            assert enc_out_dict['ys']['xs'].size(0) == batch_size
            assert enc_out_dict['ys']['xs'].size(1) == enc_out_dict['ys']['xlens'][0]

            enc.reset_cache()

            # chunk by chunk encoding
            eouts_stream = []
            n_chunks = math.ceil(xmax / N_l)
            j = 0  # time offset for input
            j_out = 0  # time offset for encoder output
            for chunk_idx in range(n_chunks):
                start = j - lookback
                end = (j + N_l + N_r) + lookahead
                xs_pad_stream = pad_list(
                    [np2tensor(x[max(0, start):end], device_id).float() for x in xs], 0.)
                xlens_stream = torch.IntTensor([xs_pad_stream.size(1) for x in xs])
                enc_out_dict_stream = enc(xs_pad_stream, xlens_stream, task='all',
                                          streaming=True,
                                          lookback=start > 0,
                                          lookahead=end < xmax - 1)

                a = enc_out_dict['ys']['xs'][:, j_out:j_out + (N_l // factor)]
                b = enc_out_dict_stream['ys']['xs'][:, :a.size(1)]
                eouts_stream.append(b)

                j += N_l
                j_out += (N_l // factor)
                if j > xmax:
                    break

            eouts_stream = torch.cat(eouts_stream, dim=1)
            assert enc_out_dict['ys']['xs'].size() == eouts_stream.size()
            assert torch.equal(enc_out_dict['ys']['xs'], eouts_stream)
