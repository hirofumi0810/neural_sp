#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""CNN encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.modules.linear import LinearND


class Conv1LBlock(nn.Module):
    """1-layer CNN block without residual connection."""

    def __init__(self,
                 input_dim,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride,
                 pooling,
                 dropout,
                 batch_norm=False):

        super(Conv1LBlock, self).__init__()

        # Conv
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=tuple(kernel_size),
                              stride=tuple(stride),
                              padding=(1, 1))
        input_dim = update_lens([input_dim], self.conv, dim=1)[0]
        self.batch_norm = nn.BatchNorm2d(out_channel) if batch_norm else None
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None

        # Max Pooling
        self.pool = None
        if len(pooling) > 0 and np.prod(pooling) > 1:
            self.pool = nn.MaxPool2d(kernel_size=tuple(pooling),
                                     stride=tuple(pooling),
                                     padding=(0, 0),
                                     ceil_mode=True)
            # NOTE: If ceil_mode is False, remove last feature when the dimension of features are odd.
            input_dim = update_lens([input_dim], self.pool, dim=1)[0]

        self.input_dim = input_dim

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim (+Δ, ΔΔ)]`
            xlens (list): A list of length `[B]`
        Returns:
            xs (FloatTensor): `[B, T', feat_dim]`
            xlens (list): A list of length `[B]`

        """
        xs = self.conv(xs)
        if self.batch_norm is not None:
            xs = self.batch_norm(xs)
        xs = F.relu(xs)
        if self.dropout is not None:
            xs = self.dropout(xs)
        xlens = update_lens(xlens, self.conv, dim=0)

        if self.pool is not None:
            xs = self.pool(xs)
            xlens = update_lens(xlens, self.pool, dim=0)

        return xs, xlens


class Conv2LBlock(nn.Module):
    """2-layer CNN block."""

    def __init__(self,
                 input_dim,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride,
                 pooling,
                 dropout,
                 batch_norm=False,
                 residual=False):

        super(Conv2LBlock, self).__init__()

        self.batch_norm = batch_norm

        # 1st layer
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=tuple(kernel_size),
                               stride=tuple(stride),
                               padding=(1, 1))
        input_dim = update_lens([input_dim], self.conv1, dim=1)[0]
        if batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(out_channel)
        self.dropout1 = nn.Dropout2d(p=dropout) if dropout > 0 else None

        # 2nd layer
        self.conv2 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=tuple(kernel_size),
                               stride=tuple(stride),
                               padding=(1, 1))
        input_dim = update_lens([input_dim], self.conv2, dim=1)[0]
        if batch_norm:
            self.batch_norm2 = nn.BatchNorm2d(out_channel)
        self.dropout2 = nn.Dropout2d(p=dropout) if dropout > 0 else None

        # Max Pooling
        self.pool = None
        if len(pooling) > 0 and np.prod(pooling) > 1:
            self.pool = nn.MaxPool2d(kernel_size=tuple(pooling),
                                     stride=tuple(pooling),
                                     padding=(0, 0),
                                     ceil_mode=True)
            # NOTE: If ceil_mode is False, remove last feature when the dimension of features are odd.
            input_dim = update_lens([input_dim], self.pool, dim=1)[0]

        self.residual = residual
        self.input_dim = input_dim

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim (+Δ, ΔΔ)]`
            xlens (list): A list of length `[B]`
        Returns:
            xs (FloatTensor): `[B, T', feat_dim]`
            xlens (list): A list of length `[B]`

        """
        residual = xs
        xs = self.conv1(xs)
        if self.batch_norm:
            xs = self.batch_norm1(xs)
        if self.residual and xs.size() == residual.size():
            xs += residual
        xs = F.relu(xs)
        if self.dropout1 is not None:
            xs = self.dropout1(xs)
        xlens = update_lens(xlens, self.conv1, dim=0)

        residual = xs
        xs = self.conv2(xs)
        if self.batch_norm:
            xs = self.batch_norm2(xs)
        if self.residual:
            xs += residual
        xs = F.relu(xs)
        if self.dropout2 is not None:
            xs = self.dropout2(xs)
        xlens = update_lens(xlens, self.conv2, dim=0)

        if self.pool is not None:
            xs = self.pool(xs)
            xlens = update_lens(xlens, self.pool, dim=0)

        return xs, xlens


class ConvEncoder(nn.Module):
    """CNN encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        in_channel (int): number of channels of input features
        channels (list): number of channles in CNN blocks
        kernel_sizes (list): size of kernels in CNN blocks
        strides (list): strides in CNN blocks
        poolings (list): size of poolings in CNN blocks
        dropout (float): probability to drop nodes in hidden-hidden connection
        batch_norm (bool): apply batch normalization
        residual (bool): add residual connections
        bottleneck_dim (int): dimension of the bottleneck layer after the last layer

    """

    def __init__(self,
                 input_dim,
                 in_channel,
                 channels,
                 kernel_sizes,
                 strides,
                 poolings,
                 dropout,
                 batch_norm=False,
                 residual=False,
                 bottleneck_dim=0):

        super(ConvEncoder, self).__init__()

        self.in_channel = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel
        self.residual = residual
        self.bottleneck_dim = bottleneck_dim

        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes) == len(strides) == len(poolings)

        self.layers = nn.ModuleList()
        in_ch = in_channel
        in_freq = self.input_freq
        for l in range(len(channels)):
            block = Conv2LBlock(input_dim=in_freq,
                                in_channel=in_ch,
                                out_channel=channels[l],
                                kernel_size=kernel_sizes[l],
                                stride=strides[l],
                                pooling=poolings[l],
                                dropout=dropout,
                                batch_norm=batch_norm,
                                residual=residual)
            self.layers += [block]
            in_freq = block.input_dim
            in_ch = channels[l]

        self._output_dim = int(in_ch * in_freq)

        if bottleneck_dim > 0:
            self.bottleneck = LinearND(self._output_dim, bottleneck_dim)
            self._output_dim = bottleneck_dim

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
        xs = xs.view(bs, time, self.in_channel, input_dim // self.in_channel).contiguous().transpose(2, 1)

        for block in self.layers:
            xs, xlens = block(xs, xlens)

        # Collapse feature dimension
        bs, out_ch, time, freq = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)

        # Reduce dimension
        if self.bottleneck_dim > 0:
            xs = self.bottleneck(xs)

        return xs, xlens


def update_lens(xlens, layer, dim=0):
    assert type(layer) in [nn.Conv2d, nn.MaxPool2d]
    if type(layer) == nn.MaxPool2d and layer.ceil_mode:
        def update(xlen): return np.ceil(
            (xlen + 1 + 2 * layer.padding[dim] - (layer.kernel_size[dim] - 1) - 1) / layer.stride[dim] + 1)
    else:
        def update(xlen): return np.floor(
            (xlen + 2 * layer.padding[dim] - (layer.kernel_size[dim] - 1) - 1) / layer.stride[dim] + 1)
    xlens = [update(xlen) for xlen in xlens]
    xlens = torch.from_numpy(np.array(xlens, dtype=np.int64))
    return xlens
