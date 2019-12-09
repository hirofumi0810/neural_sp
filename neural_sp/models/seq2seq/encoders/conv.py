#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""CNN encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import numpy as np
import torch
import torch.nn as nn

from neural_sp.models.seq2seq.encoders.encoder_base import EncoderBase


class ConvEncoder(EncoderBase):
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
        bottleneck_dim (int): dimension of the bridge layer after the last layer
        param_init (float):

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
                 bottleneck_dim=0,
                 param_init=0.1):

        super(ConvEncoder, self).__init__()
        logger = logging.getLogger("training")

        channels, kernel_sizes, strides, poolings = parse_config(channels, kernel_sizes, strides, poolings)

        self.in_channel = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel
        self.residual = residual
        self.bridge = None

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

        self._odim = int(in_ch * in_freq)

        if bottleneck_dim > 0:
            self.bridge = nn.Linear(self._odim, bottleneck_dim)
            self._odim = bottleneck_dim

        # Initialize parameters
        self.reset_parameters(param_init)

    def reset_parameters(self, param_init):
        """Initialize parameters with lecun style."""
        logger = logging.getLogger('training')
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0.)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
            elif p.dim() == 2:
                fan_in = p.size(1)
                nn.init.normal_(p, mean=0., std=1. / math.sqrt(fan_in))  # linear weight
                logger.info('Initialize %s with %s / %.3f' % (n, 'lecun', param_init))
            elif p.dim() == 4:
                fan_in = p.size(1) * p[0][0].numel()
                nn.init.normal_(p, mean=0., std=1. / math.sqrt(fan_in))  # conv weight
                logger.info('Initialize %s with %s / %.3f' % (n, 'lecun', param_init))
            else:
                raise ValueError

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim (+Δ, ΔΔ)]`
            xlens (list): A list of length `[B]`
        Returns:
            xs (FloatTensor): `[B, T', out_ch * feat_dim]`
            xlens (list): A list of length `[B]`

        """
        bs, time, input_dim = xs.size()
        xs = xs.view(bs, time, self.in_channel, input_dim // self.in_channel).contiguous().transpose(2, 1)
        # `[B, in_ch, T, input_dim // in_ch]`

        for block in self.layers:
            xs, xlens = block(xs, xlens)
        bs, out_ch, time, freq = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)  # `[B, T', out_ch * feat_dim]`

        # Bridge layer
        if self.bridge is not None:
            xs = self.bridge(xs)

        return xs, xlens


class Conv1LBlock(EncoderBase):
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
        self.batch_norm = nn.BatchNorm2d(out_channel) if batch_norm else lambda x: x
        self.dropout = nn.Dropout2d(p=dropout)

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
        xs = self.batch_norm(xs)
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens(xlens, self.conv, dim=0)

        if self.pool is not None:
            xs = self.pool(xs)
            xlens = update_lens(xlens, self.pool, dim=0)

        return xs, xlens


class Conv2LBlock(EncoderBase):
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
        self.batch_norm1 = nn.BatchNorm2d(out_channel) if batch_norm else lambda x: x
        self.dropout1 = nn.Dropout2d(p=dropout)

        # 2nd layer
        self.conv2 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=tuple(kernel_size),
                               stride=tuple(stride),
                               padding=(1, 1))
        input_dim = update_lens([input_dim], self.conv2, dim=1)[0]
        self.batch_norm2 = nn.BatchNorm2d(out_channel) if batch_norm else lambda x: x
        self.dropout2 = nn.Dropout2d(p=dropout)

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
            xlens (IntTensor): `[B]`
        Returns:
            xs (FloatTensor): `[B, T', feat_dim]`
            xlens (IntTensor): `[B]`

        """
        residual = xs

        xs = self.conv1(xs)
        xs = self.batch_norm1(xs)
        xs = torch.relu(xs)
        xs = self.dropout1(xs)
        xlens = update_lens(xlens, self.conv1, dim=0)

        xs = self.conv2(xs)
        xs = self.batch_norm2(xs)
        if self.residual and xs.size() == residual.size():
            xs += residual
            # NOTE: this is based on ResNet
        xs = torch.relu(xs)
        xs = self.dropout2(xs)
        xlens = update_lens(xlens, self.conv2, dim=0)

        if self.pool is not None:
            xs = self.pool(xs)
            xlens = update_lens(xlens, self.pool, dim=0)

        return xs, xlens


def update_lens(seq_lens, layer, dim=0, device_id=-1):
    """Update lenghts (frequency or time).
    Args:
        seq_lens (list or IntTensor):
        layer (nn.Conv2d or nn.MaxPool2d):
        dim (int):
        device_id (int):
    Returns:
        seq_lens (IntTensor):
    """
    assert type(layer) in [nn.Conv2d, nn.MaxPool2d]
    if type(layer) == nn.MaxPool2d and layer.ceil_mode:
        def update(seq_len): return math.ceil(
            (seq_len + 1 + 2 * layer.padding[dim] - (layer.kernel_size[dim] - 1) - 1) / layer.stride[dim] + 1)
    else:
        def update(seq_len): return math.floor(
            (seq_len + 2 * layer.padding[dim] - (layer.kernel_size[dim] - 1) - 1) / layer.stride[dim] + 1)
    seq_lens = [update(seq_len) for seq_len in seq_lens]
    seq_lens = torch.IntTensor(seq_lens)
    if device_id >= 0:
        seq_lens = seq_lens.cuda(device_id)
    return seq_lens


def parse_config(conv_channels, conv_kernel_sizes, conv_strides, conv_poolings):
    channels, kernel_sizes, strides, poolings = [], [], [], []
    if len(conv_channels) > 0:
        channels = [int(c) for c in conv_channels.split('_')]
    if len(conv_kernel_sizes) > 0:
        kernel_sizes = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                        for c in conv_kernel_sizes.split('_')]
    if len(conv_strides) > 0:
        strides = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                   for c in conv_strides.split('_')]
    if len(conv_poolings) > 0:
        poolings = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                    for c in conv_poolings.split('_')]
    return channels, kernel_sizes, strides, poolings
