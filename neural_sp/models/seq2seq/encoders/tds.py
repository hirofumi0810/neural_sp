#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Time-depth separable convolution (TDS) encoder."""

import logging
import math
import torch
import torch.nn as nn

from neural_sp.models.seq2seq.encoders.conv import ConvEncoder
from neural_sp.models.seq2seq.encoders.conv import LayerNorm2D
from neural_sp.models.seq2seq.encoders.conv import parse_cnn_config
from neural_sp.models.seq2seq.encoders.conv import update_lens_1d
from neural_sp.models.seq2seq.encoders.encoder_base import EncoderBase

logger = logging.getLogger(__name__)


class TDSEncoder(EncoderBase):
    """Time-depth separable convolution (TDS) encoder.

    Args:
        input_dim (int) dimension of input features (freq * channel)
        in_channel (int) number of channels of input features
        channels (list) number of channles in TDS layers
        kernel_sizes (list) size of kernels in TDS layers
        dropout (float) probability to drop nodes in hidden-hidden connection
        last_proj_dim (int): dimension of the last projection layer
        layer_norm_eps (float): epsilon value for layer normalization

    """

    def __init__(self,
                 input_dim,
                 in_channel,
                 channels,
                 kernel_sizes,
                 dropout,
                 last_proj_dim=0,
                 layer_norm_eps=1e-12):

        super(TDSEncoder, self).__init__()

        (channels, kernel_sizes, _, _), _ = parse_cnn_config(channels, kernel_sizes, '', '')

        self.C_in = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel
        self.bridge = None

        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes)

        C_i = in_channel
        in_freq = self.input_freq
        n_subsampling = 0
        self.layers = nn.ModuleList()
        for lth in range(len(channels)):
            # subsample
            if C_i != channels[lth]:
                self.layers += [SubsampelBlock(in_channel=C_i,
                                               out_channel=channels[lth],
                                               kernel_size=kernel_sizes[lth][0],
                                               stride=2 if n_subsampling < 3 else 1,
                                               in_freq=in_freq,
                                               dropout=dropout)]
                n_subsampling += 1

            # Conv
            self.layers += [TDSBlock(channel=channels[lth],
                                     kernel_size=kernel_sizes[lth][0],
                                     in_freq=in_freq,
                                     dropout=dropout,
                                     layer_norm_eps=layer_norm_eps)]

            C_i = channels[lth]

        self._odim = int(C_i * in_freq)

        if last_proj_dim > 0:
            self.bridge = nn.Linear(self._odim, last_proj_dim)
            self._odim = last_proj_dim

        self._factor = 8

        self.reset_parameters()

    @staticmethod
    def add_args(parser, args):
        # group = parser.add_argument_group("TDS encoder")
        parser = ConvEncoder.add_args(parser, args)
        return parser

    @staticmethod
    def define_name(parser, args):
        raise NotImplementedError

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
            elif p.dim() == 3:
                fan_in = p.size(1) * p[0][0].numel()
                nn.init.uniform_(p, a=-math.sqrt(4 / fan_in), b=math.sqrt(4 / fan_in))  # 1dconv weight
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', math.sqrt(4 / fan_in)))
            elif p.dim() == 4:
                fan_in = p.size(1) * p[0][0].numel()
                nn.init.uniform_(p, a=-math.sqrt(4 / fan_in), b=math.sqrt(4 / fan_in))  # 1dconv weight
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', math.sqrt(4 / fan_in)))
            else:
                raise ValueError(n)

    def forward(self, xs, xlens, task, use_cache=False, streaming=False,):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]`
            xlens (IntTensor): `[B]`
        Returns:
            eouts (dict):
                xs (FloatTensor): `[B, T', C_o * F]`
                xlens (IntTensor): `[B]`

        """
        eouts = {'ys': {'xs': None, 'xlens': None},
                 'ys_sub1': {'xs': None, 'xlens': None},
                 'ys_sub2': {'xs': None, 'xlens': None}}

        B, T, F = xs.size()
        xs = xs.contiguous().view(B, T, self.C_in, F // self.C_in).transpose(2, 1)
        # `[B, C_i, T, F // C_i]`

        for layer in self.layers:
            xs, xlens = layer(xs, xlens)
        B, C_o, T, F = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(B, T, -1)  # `[B, T, C_o * F]`

        # Bridge layer
        if self.bridge is not None:
            xs = self.bridge(xs)

        if task in ['all', 'ys']:
            eouts['ys']['xs'], eouts['ys']['xlens'] = xs, xlens
        else:
            raise NotImplementedError
        return eouts


class TDSBlock(nn.Module):
    """TDS block.

    Args:
        channel (int): input/output channle size
        kernel_size (int): kernel size
        in_freq (int): frequency width
        dropout (float): dropout probability

    """

    def __init__(self, channel, kernel_size, in_freq, dropout,
                 layer_norm_eps=1e-12):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # 2D conv over time
        self.conv1d = nn.Conv1d(in_channels=in_freq * channel,
                                out_channels=in_freq * channel,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=(kernel_size - 1) // 2,  # TODO(hirofumi0810): assymetric
                                groups=in_freq)  # depthwise
        self.norm1 = LayerNorm2D(channel, in_freq, eps=layer_norm_eps)

        # fully connected block
        self.pointwise_conv1 = nn.Conv1d(in_channels=in_freq * channel,
                                         out_channels=in_freq * channel,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.pointwise_conv2 = nn.Conv1d(in_channels=in_freq * channel,
                                         out_channels=in_freq * channel,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        # self.feed_forward = nn.Linear(in_freq * channel, in_freq * channel)
        self.norm2 = LayerNorm2D(channel, in_freq, eps=layer_norm_eps)

    def forward(self, xs, xlens):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, C, T, F]`
            xlens (IntTensor): `[B]`
        Returns:
            xs (FloatTensor): `[B, C, T, F]`
            xlens (IntTensor): `[B]`

        """
        B, C, T, F = xs.size()

        # 1d conv
        residual = xs
        xs = xs.transpose(3, 2).view(B, C * F, T)
        xs = self.dropout(torch.relu(self.conv1d(xs)))
        xs = xs.view(B, -1, F, T).transpose(3, 2)
        xs = xs + residual  # `[B, C, T, F]`
        xs = self.norm1(xs)  # not depends on time-axis based on https://arxiv.org/abs/2001.09727

        # fully connected block
        B, C, T, F = xs.size()
        residual = xs

        # v1
        xs = xs.transpose(3, 2).view(B, C * F, T)
        xs = self.dropout(torch.relu(self.pointwise_conv1(xs)))
        xs = self.dropout(self.pointwise_conv2(xs))
        xs = xs.view(B, -1, F, T).transpose(3, 2)  # `[B, C, T, F]`

        # v2
        # xs = xs.transpose(2, 1).view(B, T, -1)
        # xs = self.dropout(torch.relu(self.feed_forward(xs)))
        # xs = xs.view(B, T, C, F).transpose(2, 1)

        xs = xs + residual
        xs = self.norm2(xs)  # not depends on time-axis based on https://arxiv.org/abs/2001.09727
        return xs, xlens


class SubsampelBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, in_freq,
                 dropout, layer_norm_eps=1e-12):
        super().__init__()

        self.C_in = in_channel
        self.C_out = out_channel
        self.in_freq = in_freq

        self.conv1d = nn.Conv1d(in_channels=in_freq * in_channel,
                                out_channels=in_freq * out_channel,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=(kernel_size - 1) // 2,
                                groups=in_freq)  # TODO(hirofumi0810): Is this correct?
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm2D(out_channel, in_freq, eps=layer_norm_eps)

    def forward(self, xs, xlens):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, C_i, T, F]`
            xlens (IntTensor): `[B]`
        Returns:
            xs (FloatTensor): `[B, C_o, T, F]`
            xlens (IntTensor): `[B]`

        """
        B, C, T, F = xs.size()
        xs = xs.transpose(3, 2).view(B, C * F, T)
        xs = self.dropout(torch.relu(self.conv1d(xs)))
        xs = xs.view(B, self.C_out, F, -1).transpose(3, 2)
        xs = self.norm(xs)

        xlens = update_lens_1d(xlens, self.conv1d)
        return xs, xlens
