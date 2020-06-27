#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""TDS encoder."""

from collections import OrderedDict
import logging
import math
import torch
import torch.nn as nn

from neural_sp.models.seq2seq.encoders.conv import parse_cnn_config
from neural_sp.models.seq2seq.encoders.encoder_base import EncoderBase

logger = logging.getLogger(__name__)


class TDSEncoder(EncoderBase):
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

        (channels, kernel_sizes, _, _), _ = parse_cnn_config(channels, kernel_sizes, '', '')

        self.in_channel = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel
        self.bridge = None

        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes)

        layers = OrderedDict()
        C_i = in_channel
        in_freq = self.input_freq
        for lth in range(len(channels)):
            # subsample
            if C_i != channels[lth]:
                layers['subsample%d' % lth] = SubsampelBlock(in_channel=C_i,
                                                             out_channel=channels[lth],
                                                             in_freq=in_freq,
                                                             dropout=dropout)

            # Conv
            layers['tds%d_block%d' % (channels[lth], lth)] = TDSBlock(channel=channels[lth],
                                                                      kernel_size=kernel_sizes[lth][0],
                                                                      in_freq=in_freq,
                                                                      dropout=dropout)

            C_i = channels[lth]

        self._odim = int(C_i * in_freq)

        if bottleneck_dim > 0:
            self.bridge = nn.Linear(self._odim, bottleneck_dim)
            self._odim = bottleneck_dim

        self.layers = nn.Sequential(layers)

        self._factor = 8

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0.)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
            elif p.dim() == 2:
                fan_in = p.size(1)
                nn.init.uniform_(p, a=-math.sqrt(4 / fan_in), b=math.sqrt(4 / fan_in))  # linear weight
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', math.sqrt(4 / fan_in)))
            elif p.dim() == 4:
                fan_in = p.size(1) * p[0][0].numel()
                nn.init.uniform_(p, a=-math.sqrt(4 / fan_in), b=math.sqrt(4 / fan_in))  # conv weight
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', math.sqrt(4 / fan_in)))
            else:
                raise ValueError(n)

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, F]`
            xlens (list): A list of length `[B]`
        Returns:
            xs (FloatTensor): `[B, T', C_o * F]`
            xlens (list): A list of length `[B]`

        """
        B, T, F = xs.size()
        xs = xs.contiguous().view(B, T, self.in_channel, F // self.in_channel).transpose(2, 1)
        # `[B, C_i, T, F // C_i]`

        xs = self.layers(xs)  # `[B, C_o, T, F]`
        B, C_o, T, F = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(B, T, -1)  # `[B, T, C_o * F]`

        # Bridge layer
        if self.bridge is not None:
            xs = self.bridge(xs)

        # Update xlens
        xlens //= 8

        return xs, xlens


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

        self.dropout = nn.Dropout(p=dropout)

        self.conv2d = nn.Conv2d(in_channels=channel,
                                out_channels=channel,
                                kernel_size=(kernel_size, 1),
                                stride=(1, 1),
                                padding=(kernel_size // 2, 0),
                                groups=channel)  # depthwise
        self.norm1 = nn.LayerNorm(in_freq * channel, eps=1e-12)

        # second block
        self.conv1d_1 = nn.Conv2d(in_channels=in_freq * channel,
                                  out_channels=in_freq * channel,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.conv1d_2 = nn.Conv2d(in_channels=in_freq * channel,
                                  out_channels=in_freq * channel,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.norm2 = nn.LayerNorm(in_freq * channel, eps=1e-12)

    def forward(self, xs):
        """Forward computation.
        Args:
            xs (FloatTensor): `[B, C_i, T, F]`
        Returns:
            out (FloatTensor): `[B, C_o, T, F]`

        """
        B, C_i, T, F = xs.size()

        # first block
        residual = xs
        xs = self.dropout(torch.relu(self.conv2d(xs)))
        raise ValueError(xs.size())
        xs = xs + residual  # `[B, C_o, T, F]`

        # layer normalization
        B, C_o, T, F = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(B, T, -1)  # `[B, T, C_o * F]`
        xs = self.norm1(xs)
        xs = xs.contiguous().transpose(2, 1).unsqueeze(3)  # `[B, C_o * F, T, 1]`

        # second block
        residual = xs
        self.dropout(torch.relu(self.conv1d_1(xs)))
        xs = self.dropout(self.conv1d_2(xs)) + residual  # `[B, C_o * F, T, 1]`

        # layer normalization
        xs = xs.unsqueeze(3)  # `[B, C_o * F, T]`
        xs = xs.transpose(2, 1).contiguous().view(B, T, -1)  # `[B, T, C_o * F]`
        xs = self.norm2(xs)
        xs = xs.view(B, T, C_o, F).contiguous().transpose(2, 1)

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
        self.norm = nn.LayerNorm(in_freq * out_channel, eps=1e-12)

    def forward(self, xs):
        """Forward computation.
        Args:
            xs (FloatTensor): `[B, C_i, T, F]`
        Returns:
            out (FloatTensor): `[B, C_o, T, F]`

        """
        bs, _, time, _ = xs.size()

        xs = self.dropout(torch.relu(self.conv1d(xs)))

        # layer normalization
        bs, C_o, time, F = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)  # `[B, T, C_o * F]`
        xs = self.norm(xs)
        xs = xs.view(bs, time, C_o, F).contiguous().transpose(2, 1)

        return xs
