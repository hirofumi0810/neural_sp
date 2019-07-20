#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Select an encoder network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from neural_sp.models.seq2seq.encoders.rnn import RNNEncoder
from neural_sp.models.seq2seq.encoders.transformer import TransformerEncoder


def build_encoder(args):

    if 'transformer' in args.enc_type:
        encoder = TransformerEncoder(
            input_dim=args.input_dim if args.input_type == 'speech' else args.emb_dim,
            attn_type=args.transformer_attn_type,
            attn_n_heads=args.transformer_attn_n_heads,
            n_layers=args.enc_n_layers,
            d_model=args.d_model,
            d_ff=args.d_ff,
            pe_type=args.pe_type,
            layer_norm_eps=args.layer_norm_eps,
            dropout_in=args.dropout_in,
            dropout=args.dropout_enc,
            dropout_att=args.dropout_att,
            last_proj_dim=args.d_model if 'transformer' in args.dec_type else args.dec_n_units,
            n_stacks=args.n_stacks,
            n_splices=args.n_splices,
            conv_in_channel=args.conv_in_channel,
            conv_channels=args.conv_channels,
            conv_kernel_sizes=args.conv_kernel_sizes,
            conv_strides=args.conv_strides,
            conv_poolings=args.conv_poolings,
            conv_batch_norm=args.conv_batch_norm,
            conv_residual=args.conv_residual,
            conv_bottleneck_dim=args.conv_bottleneck_dim,
            param_init=args.param_init)
    else:
        subsample = [1] * args.enc_n_layers
        for l, s in enumerate(list(map(int, args.subsample.split('_')[:args.enc_n_layers]))):
            subsample[l] = s
        encoder = RNNEncoder(
            input_dim=args.input_dim if args.input_type == 'speech' else args.emb_dim,
            rnn_type=args.enc_type,
            n_units=args.enc_n_units,
            n_projs=args.enc_n_projs,
            n_layers=args.enc_n_layers,
            n_layers_sub1=args.enc_n_layers_sub1,
            n_layers_sub2=args.enc_n_layers_sub2,
            dropout_in=args.dropout_in,
            dropout=args.dropout_enc,
            subsample=subsample,
            subsample_type=args.subsample_type,
            last_proj_dim=args.d_model if 'transformer' in args.dec_type else args.dec_n_units,
            n_stacks=args.n_stacks,
            n_splices=args.n_splices,
            conv_in_channel=args.conv_in_channel,
            conv_channels=args.conv_channels,
            conv_kernel_sizes=args.conv_kernel_sizes,
            conv_strides=args.conv_strides,
            conv_poolings=args.conv_poolings,
            conv_batch_norm=args.conv_batch_norm,
            conv_residual=args.conv_residual,
            conv_bottleneck_dim=args.conv_bottleneck_dim,
            residual=args.enc_residual,
            nin=args.enc_nin,
            task_specific_layer=args.task_specific_layer,
            param_init=args.param_init)
        # NOTE: pure Conv/TDS/GatedConv encoders are also included

    return encoder
