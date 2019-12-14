#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Gated convolutional neural netwrok encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.modules.glu import ConvGLUBlock
from neural_sp.models.seq2seq.encoders.conv import parse_config
from neural_sp.models.seq2seq.encoders.encoder_base import EncoderBase

logger = logging.getLogger(__name__)


class GatedConvEncoder(EncoderBase):
    """Gated convolutional neural netwrok encoder.

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
        param_init (float):

    """

    def __init__(self,
                 input_dim,
                 in_channel,
                 channels,
                 kernel_sizes,
                 dropout,
                 bottleneck_dim=0,
                 param_init=0.1):

        super(GatedConvEncoder, self).__init__()

        channels, kernel_sizes, _, _ = parse_config(channels, kernel_sizes, '', '')

        self.in_channel = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel
        self.bridge = None

        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes)

        layers = OrderedDict()
        for l in range(len(channels)):
            layers['conv%d' % l] = ConvGLUBlock(kernel_sizes[l][0], input_dim, channels[l],
                                                weight_norm=True,
                                                dropout=0.2)
            input_dim = channels[l]

        # weight normalization + GLU for the last fully-connected layer
        self.fc_glu = nn.utils.weight_norm(nn.Linear(input_dim, input_dim * 2),
                                           name='weight', dim=0)

        self._odim = int(input_dim)

        if bottleneck_dim > 0:
            self.bridge = nn.Linear(self._odim, bottleneck_dim)
            self._odim = bottleneck_dim

        self.layers = nn.Sequential(layers)

        self.reset_parameters(param_init)

    def reset_parameters(self, param_init):
        """Initialize parameters with kaiming_uniform style."""
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0.)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
            elif p.dim() in [2, 4]:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
                # nn.init.kaiming_normal_(p, mode='fan_in', nonlinearity='relu')
                logger.info('Initialize %s with %s / %.3f' % (n, 'kaiming_uniform', param_init))
            else:
                raise ValueError(n)

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
        xs = xs.transpose(2, 1).unsqueeze(3)  # `[B, in_ch (input_dim), T, 1]`

        xs = self.layers(xs)  # `[B, out_ch, T, 1]`
        bs, out_ch, time, freq = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)  # `[B, T, out_ch * feat_dim]`

        # weight normalization + GLU for the last fully-connected layer
        xs = F.glu(self.fc_glu(xs), dim=2)

        # Bridge layer
        if self.bridge is not None:
            xs = self.bridge(xs)

        # NOTE: no subsampling is conducted

        return xs, xlens
