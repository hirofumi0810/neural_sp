#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""CNN encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import math
import torch.nn as nn

from neural_sp.models.model_utils import LinearND
from neural_sp.models.seq2seq.encoders.cnn_utils import ConvOutSize
from neural_sp.models.seq2seq.encoders.cnn_utils import Maxout


class CNNEncoder(nn.Module):
    """CNN encoder.

    Args:
        input_dim (int) dimension of input features (freq * channel)
        in_channel (int) number of channels of input features
        channels (list) number of channles in CNN layers
        kernel_sizes (list) size of kernels in CNN layers
        strides (list): strides in CNN layers
        poolings (list) size of poolings in CNN layers
        dropout (float) probability to drop nodes in hidden-hidden connection
        activation (str): relu or prelu or hard_tanh or maxout
        batch_norm (bool): if True, apply batch normalization
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
                 activation='relu',
                 batch_norm=False,
                 bottleneck_dim=0):

        super(CNNEncoder, self).__init__()

        self.in_channel = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel
        self.bottleneck_dim = bottleneck_dim

        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes)
        assert len(kernel_sizes) == len(strides)
        assert len(strides) == len(poolings)

        layers = OrderedDict()
        in_ch = self.in_channel
        in_freq = self.input_freq
        for l in range(len(channels)):

            # Conv
            conv = nn.Conv2d(in_channels=in_ch,
                             out_channels=channels[l],
                             kernel_size=tuple(kernel_sizes[l]),
                             stride=tuple(strides[l]),
                             #  padding=tuple(strides[l]),
                             padding=0,
                             bias=not batch_norm)
            layers['conv' + str(channels[l]) + '_l' + str(l)] = conv
            in_freq = int(math.floor((in_freq + 2 * conv.padding[0] - conv.kernel_size[0]) / conv.stride[0] + 1))

            # Activation
            if activation == 'relu':
                act = nn.ReLU()
            elif activation == 'prelu':
                act = nn.PReLU(num_parameters=1, init=0.2)
            elif activation == 'hard_tanh':
                act = nn.Hardtanh(min_val=0, max_val=20, inplace=True)
            elif activation == 'maxout':
                raise NotImplementedError(activation)
                # act = Maxout(1, 1, 2)
            else:
                raise NotImplementedError(activation)
            layers[activation + '_' + str(l)] = act

            # Max Pooling
            if len(poolings[l]) > 0 and poolings[l][0] * poolings[l][1] > 1:
                pool = nn.MaxPool2d(kernel_size=tuple(poolings[l]),
                                    stride=tuple(poolings[l]),
                                    padding=(0, 0),  # default
                                    ceil_mode=True)
                layers['pool_' + str(l)] = pool
                # NOTE: If ceil_mode is False, remove last feature when the
                # dimension of features are odd.

                in_freq = int(math.ceil(
                    (in_freq + 2 * pool.padding[0] - pool.kernel_size[0]) / pool.stride[0] + 1))

            # Batch Normalization
            if batch_norm:
                layers['bn_' + str(l)] = nn.BatchNorm2d(channels[l])

            # Dropout for hidden-hidden connection
            layers['dropout_' + str(l)] = nn.Dropout(p=dropout)
            # TODO(hirofumi): compare BN before and after ReLU

            in_ch = channels[l]

        self._output_dim = int(in_ch * in_freq)

        if bottleneck_dim > 0:
            self.bottleneck = LinearND(self._output_dim, bottleneck_dim)
            self._output_dim = bottleneck_dim

        self.layers = nn.Sequential(layers)
        self.get_conv_out_size = ConvOutSize(self.layers)

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim (+Δ, ΔΔ)]`
            xlens (list): A list of length `[B]`
        Returns:
            xs (FloatTensor): `[B, T', feature_dim]`
            xlens (list): A list of length `[B]`

        """
        bs, max_xlen, input_dim = xs.size()

        # Reshape to 4D tensor `[B, in_ch, freq // in_ch, max_xlen]`
        xs = xs.contiguous().view(bs, max_xlen, input_dim // self.in_channel, self.in_channel)
        xs = xs.transpose(1, 3)

        xs = self.layers(xs)
        # NOTE: xs: `[B, out_ch, new_freq, new_time]`

        # Collapse feature dimension
        bs, out_ch, freq, time = xs.size()
        xs = xs.transpose(1, 2).contiguous().view(bs, time, -1)

        # Reduce dimension
        if self.bottleneck_dim > 0:
            xs = self.bottleneck(xs)

        # Update xlens
        xlens = [self.get_conv_out_size(xlen, 1) for xlen in xlens]

        return xs, xlens
