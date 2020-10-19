#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Gated convolutional encoder."""

from collections import OrderedDict
import logging
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.modules.glu import ConvGLUBlock
from neural_sp.models.seq2seq.encoders.conv import parse_cnn_config
from neural_sp.models.seq2seq.encoders.encoder_base import EncoderBase

logger = logging.getLogger(__name__)


class GatedConvEncoder(EncoderBase):
    """Gated convolutional encoder.

    Args:
        input_dim (int) dimension of input features (freq * channel)
        in_channel (int) number of channels of input features
        channels (list) number of channles in TDS layers
        kernel_sizes (list) size of kernels in TDS layers
        dropout (float) dropout probability
        batch_norm (bool): apply batch normalization
        last_proj_dim (int): dimension of the last projection layer
        param_init (float): model initialization parameter

    """

    def __init__(self, input_dim, in_channel, channels, kernel_sizes,
                 dropout, last_proj_dim, param_init):

        super(GatedConvEncoder, self).__init__()

        (channels, kernel_sizes, _, _), _ = parse_cnn_config(channels, kernel_sizes, '', '')

        self.in_channel = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel
        self.bridge = None

        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes)

        layers = OrderedDict()
        for lth in range(len(channels)):
            layers['conv%d' % lth] = ConvGLUBlock(kernel_sizes[lth][0], input_dim, channels[lth],
                                                  weight_norm=True,
                                                  dropout=0.2)
            input_dim = channels[lth]

        # weight normalization + GLU for the last fully-connected layer
        self.fc_glu = nn.utils.weight_norm(nn.Linear(input_dim, input_dim * 2),
                                           name='weight', dim=0)

        self._odim = int(input_dim)

        if last_proj_dim > 0:
            self.bridge = nn.Linear(self._odim, last_proj_dim)
            self._odim = last_proj_dim

        self.layers = nn.Sequential(layers)

        self._factor = 1

        self.reset_parameters(param_init)

    @staticmethod
    def define_name(dir_name, args):
        return dir_name

    def reset_parameters(self, param_init):
        """Initialize parameters with kaiming_uniform style."""
        logger.info('===== Initialize %s with kaiming_uniform style =====' % self.__class__.__name__)
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

    def forward(self, xs, xlens, task, streaming=False, lookback=False, lookahead=False):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]`
            xlens (IntTensor): `[B]`
            streaming (bool): streaming encoding
            lookback (bool): truncate leftmost frames for lookback in CNN context
            lookahead (bool): truncate rightmost frames for lookahead in CNN context
        Returns:
            eouts (dict):
                xs (FloatTensor): `[B, T', C_o * F]`
                xlens (IntTensor): `[B]`

        """
        eouts = {'ys': {'xs': None, 'xlens': None},
                 'ys_sub1': {'xs': None, 'xlens': None},
                 'ys_sub2': {'xs': None, 'xlens': None}}

        bs, xmax, input_dim = xs.size()
        xs = xs.transpose(2, 1).unsqueeze(3)  # `[B, in_ch (input_dim), T, 1]`

        xs = self.layers(xs)  # `[B, out_ch, T, 1]`
        bs, out_ch, xmax, freq = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, xmax, -1)  # `[B, T, out_ch * feat_dim]`

        # weight normalization + GLU for the last fully-connected layer
        xs = F.glu(self.fc_glu(xs), dim=2)

        # Bridge layer
        if self.bridge is not None:
            xs = self.bridge(xs)

        # NOTE: no subsampling is conducted

        if task in ['all', 'ys']:
            eouts['ys']['xs'], eouts['ys']['xlens'] = xs, xlens
        else:
            raise NotImplementedError
        return eouts
