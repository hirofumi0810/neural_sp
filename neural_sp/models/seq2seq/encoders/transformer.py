#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Self-attention encoder for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

from neural_sp.models.linear import LinearND
from neural_sp.models.linear import SublayerConnection
from neural_sp.models.linear import PositionwiseFeedForward
from neural_sp.models.linear import PositionalEncoding
from neural_sp.models.seq2seq.encoders.cnn import CNNEncoder
from neural_sp.models.seq2seq.decoders.multihead_attention import TransformerMultiheadAttentionMechanism


class TransformerEncoder(nn.Module):
    """Transformer encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        nlayers (int): number of blocks
        d_model (int): dimension of keys/values/queries in
                   TransformerMultiheadAttentionMechanism, also the input size of
                   the first-layer of the PositionwiseFeedForward
        d_ff (int): dimension of the second layer of the PositionwiseFeedForward
        attn_type (str): type of attention
        nheads (int): number of heads for multi-head attention
        dropout_in (float): dropout probability for input-hidden connection
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        nstacks (int): number of frames to stack
        nsplices (int): frames to splice. Default is 1 frame.
        conv_in_channel (int): number of channels of input features
        conv_channels (int): number of channles in the CNN layers
        conv_kernel_sizes (list): size of kernels in the CNN layers
        conv_strides (list): number of strides in the CNN layers
        conv_poolings (list): size of poolings in the CNN layers
        conv_batch_norm (bool): apply batch normalization only in the CNN layers
        conv_bottleneck_dim (int): dimension of the bottleneck layer between CNN and self-attention layers

    """

    def __init__(self,
                 input_dim,
                 nlayers,
                 d_model,
                 d_ff,
                 attn_type,
                 nheads,
                 dropout_in,
                 dropout,
                 dropout_att,
                 nstacks,
                 nsplices,
                 conv_in_channel,
                 conv_channels,
                 conv_kernel_sizes,
                 conv_strides,
                 conv_poolings,
                 conv_batch_norm,
                 conv_bottleneck_dim):

        super(TransformerEncoder, self).__init__()

        # Setting for CNNs before RNNs
        if conv_poolings:
            channels = [int(c) for c in conv_channels.split('_')] if len(conv_channels) > 0 else []
            kernel_sizes = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                            for c in conv_kernel_sizes.split('_')] if len(conv_kernel_sizes) > 0 else []
            strides = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                       for c in conv_strides.split('_')] if len(conv_strides) > 0 else []
            poolings = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                        for c in conv_poolings.split('_')] if len(conv_poolings) > 0 else []
        else:
            channels = []
            kernel_sizes = []
            strides = []
            poolings = []

        if len(channels) > 0 and len(channels) == len(kernel_sizes) and len(kernel_sizes) == len(strides):
            # assert nstacks == 1 and nsplices == 1
            self.conv = CNNEncoder(input_dim * nstacks,
                                   in_channel=conv_in_channel * nstacks,
                                   channels=channels,
                                   kernel_sizes=kernel_sizes,
                                   strides=strides,
                                   poolings=poolings,
                                   dropout=dropout,
                                   activation='relu',
                                   batch_norm=conv_batch_norm,
                                   bottleneck_dim=d_model
                                   #    bottleneck_dim=conv_bottleneck_dim
                                   )
            self._output_dim = self.conv.output_dim
        else:
            self._output_dim = input_dim * nsplices * nstacks
            self.conv = None

            self.embed_in = LinearND(self._output_dim, d_model,
                                     dropout=dropout_in)

        self.layer_norm_in = nn.LayerNorm(d_model, eps=1e-6)
        self.position_in = PositionalEncoding(d_model, dropout_in)

        # Self-attention layers
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, d_ff,
                                     attn_type, nheads,
                                     dropout, dropout_att)
             for l in range(nlayers)])
        self._output_dim = d_model

        self.layer_norm_top = nn.LayerNorm(d_model, eps=1e-6)

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, xs, xlens, task):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim]`
            xlens (list): `[B]`
            task (str): not supported now
        Returns:
            eouts (dict):
                xs (FloatTensor): `[B, T, d_model]`
                xlens (list): `[B]`

        """
        eouts = {'ys': {'xs': None, 'xlens': None},
                 'ys.ctc': {'xs': None, 'xlens': None},
                 'ys_sub1': {'xs': None, 'xlens': None},
                 'ys_sub1.ctc': {'xs': None, 'xlens': None},
                 'ys_sub2': {'xs': None, 'xlens': None},
                 'ys_sub2.ctc': {'xs': None, 'xlens': None}}

        # Path through CNN layers before RNN layers
        if self.conv is None:
            # Transform to d_model dimension
            xs = self.embed_in(xs)
        else:
            xs, xlens = self.conv(xs, xlens)

        bs, max_time = xs.size()[:2]

        # Make source-source mask
        xx_mask = xs.new_ones(bs, max_time, max_time)
        for b in range(bs):
            if xlens[b] < max_time:
                xx_mask[b, xlens[b]:, xlens[b]:] = 0

        # Positional encoding & layer normalization
        xs = self.position_in(xs)
        xs = self.layer_norm_in(xs)

        for i in range(len(self.layers)):
            xs, xx_aw = self.layers[i](xs, xx_mask)

        xs = self.layer_norm_top(xs)

        eouts['ys']['xs'] = xs
        eouts['ys']['xlens'] = xlens

        return eouts


class TransformerEncoderBlock(nn.Module):
    """A single layer of the transformer encoder.

    Args:
        d_model (int): dimension of keys/values/queries in
                   TransformerMultiheadAttentionMechanism, also the input size of
                   the first-layer of the PositionwiseFeedForward
        d_ff (int): second-layer of the PositionwiseFeedForward
        attn_type (str):
        nheads (int): number of heads for multi-head attention
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions

    """

    def __init__(self, d_model, d_ff, attn_type, nheads,
                 dropout, dropout_att):
        super(TransformerEncoderBlock, self).__init__()

        # self-attention
        self.self_attn = TransformerMultiheadAttentionMechanism(nheads, d_model, dropout_att)
        self.add_norm1 = SublayerConnection(d_model, dropout, layer_norm=True)

        # feed-forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm2 = SublayerConnection(d_model, dropout, layer_norm=True)

    def forward(self, xs, mask):
        """Transformer encoder layer definition.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
            mask (LongTensor): `[B, T, T]`
                0: place to pad with -inf
                1: otherwise
        Returns:
            xs (FloatTensor): `[B, T, d_model]`
            xx_aw (FloatTensor):

        """
        # self-attention
        xs, xx_aw = self.add_norm1(xs, sublayer=lambda xs: self.self_attn(xs, xs, xs, mask))  # key/value/query

        # position-wise feed-forward
        xs = self.add_norm2(xs, sublayer=lambda xs: self.feed_forward(xs))
        return xs, xx_aw
