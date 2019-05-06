#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""TDS encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.modules.linear import LinearND


class TDSBlock(nn.Module):
    """TDS block.

    Args:
        channel (int):
        kernel_size (int):
        in_freq (int):
        dropout (float):

    """

    def __init__(self, channel, kernel_size, in_freq, dropout):
        super().__init__()

        self.channel = channel
        self.in_freq = in_freq

        self.conv2d = nn.Conv2d(in_channels=channel,
                                out_channels=channel,
                                kernel_size=(kernel_size, 1),
                                stride=(1, 1),
                                padding=(kernel_size // 2, 0))
        self.dropout_1 = nn.Dropout(p=dropout)
        self.layer_norm_1 = nn.LayerNorm(in_freq * channel, eps=1e-6)

        # second block
        self.conv1d_1 = nn.Conv2d(in_channels=in_freq * channel,
                                  out_channels=in_freq * channel,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.dropout_2_1 = nn.Dropout(p=dropout)
        self.conv1d_2 = nn.Conv2d(in_channels=in_freq * channel,
                                  out_channels=in_freq * channel,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.dropout_2_2 = nn.Dropout(p=dropout)
        self.layer_norm_2 = nn.LayerNorm(in_freq * channel, eps=1e-6)

    def forward(self, xs):
        """Forward computation.
        Args:
            xs (FloatTensor): `[B, in_ch, T, feat_dim]`
        Returns:
            out (FloatTensor): `[B, out_ch, T, feat_dim]`

        """
        bs, _, time, _ = xs.size()

        # first block
        residual = xs
        xs = self.conv2d(xs)
        xs = F.relu(xs)
        self.dropout_1(xs)

        xs = xs + residual  # `[B, out_ch, T, feat_dim]`

        # layer normalization
        bs, out_ch, time, feat_dim = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)  # `[B, T, out_ch * feat_dim]`
        xs = self.layer_norm_1(xs)
        xs = xs.contiguous().transpose(2, 1).unsqueeze(3)  # `[B, out_ch * feat_dim, T, 1]`

        # second block
        residual = xs
        xs = self.conv1d_1(xs)
        xs = F.relu(xs)
        self.dropout_2_1(xs)
        xs = self.conv1d_2(xs)
        self.dropout_2_2(xs)
        xs = xs + residual  # `[B, out_ch * feat_dim, T, 1]`

        # layer normalization
        xs = xs.unsqueeze(3)  # `[B, out_ch * feat_dim, T]`
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)  # `[B, T, out_ch * feat_dim]`
        xs = self.layer_norm_2(xs)
        xs = xs.view(bs, time, out_ch, feat_dim).contiguous().transpose(2, 1)

        return xs


class SubsampelBlock(nn.Module):
    def __init__(self, in_channel, out_channel, in_freq, dropout):
        super().__init__()

        self.conv1d = nn.Conv2d(in_channels=in_channel,
                                out_channels=out_channel,
                                kernel_size=(2, 1),
                                stride=(2, 1),
                                padding=(0, 0))
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(in_freq * out_channel, eps=1e-6)

    def forward(self, xs):
        """Forward computation.
        Args:
            xs (FloatTensor): `[B, in_ch, T, feat_dim]`
        Returns:
            out (FloatTensor): `[B, out_ch, T, feat_dim]`

        """
        bs, _, time, _ = xs.size()

        xs = self.conv1d(xs)
        xs = F.relu(xs)
        xs = self.dropout(xs)

        # layer normalization
        bs, out_ch, time, feat_dim = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)  # `[B, T, out_ch * feat_dim]`
        xs = self.layer_norm(xs)
        xs = xs.view(bs, time, out_ch, feat_dim).contiguous().transpose(2, 1)

        return xs


class TDSEncoder(nn.Module):
    """TDS (tim-depth separable convolutional) encoder.

    Args:
        input_dim (int) dimension of input features (freq * channel)
        in_channel (int) number of channels of input features
        channels (list) number of channles in TDS layers
        kernel_sizes (list) size of kernels in TDS layers
        strides (list): strides in TDS layers
        poolings (list) size of poolings in TDS layers
        dropout (float) probability to drop nodes in hidden-hidden connection
        batch_norm (bool): if True, apply batch normalization
        bottleneck_dim (int): dimension of the bottleneck layer after the last layer

    """

    def __init__(self,
                 input_dim,
                 in_channel,
                 channels,
                 kernel_sizes,
                 dropout,
                 bottleneck_dim=0):

        super(TDSEncoder, self).__init__()

        self.in_channel = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel
        self.bottleneck_dim = bottleneck_dim

        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes)

        layers = OrderedDict()
        in_ch = in_channel
        in_freq = self.input_freq
        for l in range(len(channels)):
            # subsample
            if in_ch != channels[l]:
                layers['subsample%d' % l] = SubsampelBlock(in_channel=in_ch,
                                                           out_channel=channels[l],
                                                           in_freq=in_freq,
                                                           dropout=dropout)

            # Conv
            layers['tds%d_block%d' % (channels[l], l)] = TDSBlock(channel=channels[l],
                                                                  kernel_size=kernel_sizes[l][0],
                                                                  in_freq=in_freq,
                                                                  dropout=dropout)

            in_ch = channels[l]

        self._output_dim = int(in_ch * in_freq)

        if bottleneck_dim > 0:
            self.bottleneck = LinearND(self._output_dim, bottleneck_dim)
            self._output_dim = bottleneck_dim

        self.layers = nn.Sequential(layers)

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim (+Δ, ΔΔ)]`
            xlens (list): A list of length `[B]`
        Returns:
            xs (FloatTensor): `[B, T', feat_dim]`
            xlens (list): A list of length `[B]`

        """
        bs, time, input_dim = xs.size()

        # Reshape to `[B, in_ch, T, input_dim // in_ch]`
        xs = xs.contiguous().view(bs, time, self.in_channel, input_dim // self.in_channel).transpose(2, 1)

        xs = self.layers(xs)  # `[B, out_ch, T, feat_dim]`

        # Collapse feature dimension
        bs, out_ch, time, freq = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)

        # Reduce dimension
        if self.bottleneck_dim > 0:
            xs = self.bottleneck(xs)

        # Update xlens
        xlens = [xlen // 8 for xlen in xlens]

        return xs, xlens
