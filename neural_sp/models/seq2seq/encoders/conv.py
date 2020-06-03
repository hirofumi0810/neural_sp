#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""CNN encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.util import strtobool
import logging
import math
import numpy as np
import torch
import torch.nn as nn

from neural_sp.models.modules.initialization import init_with_lecun
from neural_sp.models.seq2seq.encoders.encoder_base import EncoderBase

logger = logging.getLogger(__name__)


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
        layer_norm (bool): apply layer normalization
        residual (bool): add residual connections
        bottleneck_dim (int): dimension of the bridge layer after the last layer
        param_init (float): model initialization parameter
        layer_norm_eps (float):

    """

    def __init__(self, input_dim, in_channel, channels,
                 kernel_sizes, strides, poolings,
                 dropout, batch_norm, layer_norm, residual,
                 bottleneck_dim, param_init, layer_norm_eps=1e-12):

        super(ConvEncoder, self).__init__()

        channels, kernel_sizes, strides, poolings = parse_config(channels, kernel_sizes, strides, poolings)

        self.in_channel = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel
        self.residual = residual
        self.bridge = None

        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes) == len(strides) == len(poolings)

        self.layers = nn.ModuleList()
        C_i = in_channel
        in_freq = self.input_freq
        for lth in range(len(channels)):
            block = Conv2LBlock(input_dim=in_freq,
                                in_channel=C_i,
                                out_channel=channels[lth],
                                kernel_size=kernel_sizes[lth],  # (T,F)
                                stride=strides[lth],  # (T,F)
                                pooling=poolings[lth],  # (T,F)
                                dropout=dropout,
                                batch_norm=batch_norm,
                                layer_norm=layer_norm,
                                layer_norm_eps=layer_norm_eps,
                                residual=residual)
            self.layers += [block]
            in_freq = block.output_dim
            C_i = channels[lth]

        self._odim = int(C_i * in_freq)

        if bottleneck_dim > 0:
            self.bridge = nn.Linear(self._odim, bottleneck_dim)
            self._odim = bottleneck_dim

        # calculate subsampling factor
        self._factor = 1
        if poolings:
            for p in poolings:
                self._factor *= p[0]

        self.reset_parameters(param_init)

    @staticmethod
    def add_args(parser, args):
        """Add arguments."""
        group = parser.add_argument_group("CNN encoder")
        group.add_argument('--conv_in_channel', type=int, default=1,
                           help='input dimension of the first CNN block')
        group.add_argument('--conv_channels', type=str, default="",
                           help='delimited list of channles in each CNN block')
        group.add_argument('--conv_kernel_sizes', type=str, default="",
                           help='delimited list of kernel sizes in each CNN block')
        group.add_argument('--conv_strides', type=str, default="",
                           help='delimited list of strides in each CNN block')
        group.add_argument('--conv_poolings', type=str, default="",
                           help='delimited list of poolings in each CNN block')
        group.add_argument('--conv_batch_norm', type=strtobool, default=False,
                           help='apply batch normalization in each CNN block')
        group.add_argument('--conv_layer_norm', type=strtobool, default=False,
                           help='apply layer normalization in each CNN block')
        group.add_argument('--conv_bottleneck_dim', type=int, default=0,
                           help='dimension of the bottleneck layer between CNN and the subsequent RNN/Transformer layers')
        return parser

    def reset_parameters(self, param_init):
        """Initialize parameters with lecun style."""
        logger.info('===== Initialize %s with lecun style =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_lecun(n, p, param_init)

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, F]`
            xlens (list): A list of length `[B]`
        Returns:
            xs (FloatTensor): `[B, T', F']`
            xlens (list): A list of length `[B]`

        """
        B, T, F = xs.size()
        C_i = self.in_channel
        xs = xs.view(B, T, C_i, F // C_i).contiguous().transpose(2, 1)  # `[B, C_i, T, F // C_i]`

        for block in self.layers:
            xs, xlens = block(xs, xlens)
        B, C_o, T, F = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(B, T, -1)  # `[B, T', C_o * F']`

        # Bridge layer
        if self.bridge is not None:
            xs = self.bridge(xs)

        return xs, xlens


class Conv1LBlock(EncoderBase):
    """1-layer CNN block without residual connection."""

    def __init__(self, input_dim, in_channel, out_channel,
                 kernel_size, stride, pooling,
                 dropout, batch_norm, layer_norm, layer_norm_eps):

        super(Conv1LBlock, self).__init__()

        # Conv
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=tuple(kernel_size),
                              stride=tuple(stride),
                              padding=(1, 1))
        self._odim = update_lens([input_dim], self.conv, dim=1)[0]
        self.batch_norm = nn.BatchNorm2d(out_channel) if batch_norm else lambda x: x
        self.layer_norm = LayerNorm2D(out_channel * self._odim.item(),
                                      eps=layer_norm_eps) if layer_norm else lambda x: x
        self.dropout = nn.Dropout2d(p=dropout)

        # Max Pooling
        self.pool = None
        if len(pooling) > 0 and np.prod(pooling) > 1:
            self.pool = nn.MaxPool2d(kernel_size=tuple(pooling),
                                     stride=tuple(pooling),
                                     padding=(0, 0),
                                     ceil_mode=True)
            # NOTE: If ceil_mode is False, remove last feature when the dimension of features are odd.
            self._odim = update_lens([self._odim], self.pool, dim=1)[0].item()

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, C_i, T, F]`
            xlens (list): A list of length `[B]`
        Returns:
            xs (FloatTensor): `[B, C_o, T', F']`
            xlens (list): A list of length `[B]`

        """
        xs = self.conv(xs)
        xs = self.batch_norm(xs)
        xs = self.layer_norm(xs)
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens(xlens, self.conv, dim=0)

        if self.pool is not None:
            xs = self.pool(xs)
            xlens = update_lens(xlens, self.pool, dim=0)

        return xs, xlens


class Conv2LBlock(EncoderBase):
    """2-layer CNN block."""

    def __init__(self, input_dim, in_channel, out_channel,
                 kernel_size, stride, pooling,
                 dropout, batch_norm, layer_norm, layer_norm_eps, residual):

        super(Conv2LBlock, self).__init__()

        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.residual = residual
        self.dropout = nn.Dropout2d(p=dropout)

        # 1st layer
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=tuple(kernel_size),
                               stride=tuple(stride),
                               padding=(1, 1))
        self._odim = update_lens([input_dim], self.conv1, dim=1)[0]
        self.batch_norm1 = nn.BatchNorm2d(out_channel) if batch_norm else lambda x: x
        self.layer_norm1 = LayerNorm2D(out_channel * self._odim.item(),
                                       eps=layer_norm_eps) if layer_norm else lambda x: x

        # 2nd layer
        self.conv2 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=tuple(kernel_size),
                               stride=tuple(stride),
                               padding=(1, 1))
        self._odim = update_lens([self._odim], self.conv2, dim=1)[0]
        self.batch_norm2 = nn.BatchNorm2d(out_channel) if batch_norm else lambda x: x
        self.layer_norm2 = LayerNorm2D(out_channel * self._odim.item(),
                                       eps=layer_norm_eps) if layer_norm else lambda x: x

        # Max Pooling
        self.pool = None
        if len(pooling) > 0 and np.prod(pooling) > 1:
            self.pool = nn.MaxPool2d(kernel_size=tuple(pooling),
                                     stride=tuple(pooling),
                                     padding=(0, 0),
                                     ceil_mode=True)
            # NOTE: If ceil_mode is False, remove last feature when the dimension of features are odd.
            self._odim = update_lens([self._odim], self.pool, dim=1)[0].item()
            if self._odim % 2 != 0:
                self._odim = (self._odim // 2) * 2
                # TODO(hirofumi0810): more efficient way?

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, C_i, T, F]`
            xlens (IntTensor): `[B]`
        Returns:
            xs (FloatTensor): `[B, C_o, T', F']`
            xlens (IntTensor): `[B]`

        """
        residual = xs

        xs = self.conv1(xs)
        xs = self.batch_norm1(xs)
        xs = self.layer_norm1(xs)
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens(xlens, self.conv1, dim=0)

        xs = self.conv2(xs)
        xs = self.batch_norm2(xs)
        xs = self.layer_norm2(xs)
        if self.residual and xs.size() == residual.size():
            xs += residual
            # NOTE: this is based on ResNet
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens(xlens, self.conv2, dim=0)

        if self.pool is not None:
            xs = self.pool(xs)
            xlens = update_lens(xlens, self.pool, dim=0)

        return xs, xlens


class LayerNorm2D(nn.Module):
    """Layer normalization for CNN outputs."""

    def __init__(self, dim, eps=1e-12):

        super(LayerNorm2D, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, xs):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, C, T, F]`
        Returns:
            xs (FloatTensor): `[B, C, T, F]`

        """
        B, C, T, F = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(B, T, C * F)
        xs = self.norm(xs)
        xs = xs.view(B, T, C, F).transpose(2, 1)
        return xs


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
    if seq_lens is None:
        return seq_lens
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
