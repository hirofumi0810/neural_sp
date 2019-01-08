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

from neural_sp.models.seq2seq.encoders.cnn_utils import ConvOutSize
from neural_sp.models.seq2seq.encoders.cnn_utils import Maxout


class CNNEncoder(nn.Module):
    """CNN encoder.

    Args:
        input_dim (int): the dimension of input features (freq * channel)
        in_channel (int): the number of channels of input features
        channels (list): the number of channles in CNN layers
        kernel_sizes (list): the size of kernels in CNN layers
        strides (list): strides in CNN layers
        poolings (list): the size of poolings in CNN layers
        dropout (float): the probability to drop nodes in hidden-hidden connection
        activation (str): relu or prelu or hard_tanh or maxout
        batch_norm (bool): if True, apply batch normalization

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
                 batch_norm=False):

        super(CNNEncoder, self).__init__()

        self.in_channel = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel

        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes)
        assert len(kernel_sizes) == len(strides)
        assert len(strides) == len(poolings)

        layers = OrderedDict()
        in_ch = self.in_channel
        in_freq = self.input_freq
        first_max_pool = True
        for l in range(len(channels)):

            # Conv
            conv = nn.Conv2d(in_channels=in_ch,
                             out_channels=channels[l],
                             kernel_size=tuple(kernel_sizes[l]),
                             stride=tuple(strides[l]),
                             padding=tuple(strides[l]),
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
                act = Maxout(1, 1, 2)
            else:
                raise NotImplementedError(activation)
            layers[activation + '_' + str(l)] = act

            # Max Pooling
            if len(poolings[l]) > 0 and poolings[l][0] * poolings[l][1] > 1:
                pool = nn.MaxPool2d(kernel_size=tuple(poolings[l]),
                                    stride=tuple(poolings[l]),
                                    padding=(0, 0),  # default
                                    ceil_mode=not first_max_pool)
                layers['pool_' + str(l)] = pool
                # NOTE: If ceil_mode is False, remove last feature when the
                # dimension of features are odd.

                first_max_pool = False
                # NOTE: This is important for having the same frames as RNN models

                if pool.ceil_mode:
                    in_freq = int(math.ceil(
                        (in_freq + 2 * pool.padding[0] - pool.kernel_size[0]) / pool.stride[0] + 1))
                else:
                    in_freq = int(math.floor(
                        (in_freq + 2 * pool.padding[0] - pool.kernel_size[0]) / pool.stride[0] + 1))

            # Batch Normalization
            if batch_norm:
                layers['bn_' + str(l)] = nn.BatchNorm2d(channels[l])

            # Dropout for hidden-hidden connection
            layers['dropout_' + str(l)] = nn.Dropout(p=dropout)
            # TODO(hirofumi): compare BN before and after ReLU

            in_ch = channels[l]

        self.layers = nn.Sequential(layers)

        self.get_conv_out_size = ConvOutSize(self.layers)
        self.output_dim = int(in_ch * in_freq)

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim (+Δ, ΔΔ)]`
            xlens (list): A list of length `[B]`
        Returns:
            xs (FloatTensor): `[B, T', feature_dim]`
            xlens (list): A list of length `[B]`

        """
        bs, max_time, input_dim = xs.size()
        # assert input_dim == self.input_freq * self.in_channel

        # Reshape to 4D tensor `[B, in_ch, max_time, freq // in_ch]`
        # xs = xs.view(bs, max_time, self.in_channel, input_dim // self.in_channel)
        # xs = xs.transpose(1, 2).contiguous()

        xs = xs.view(bs, max_time, input_dim // self.in_channel, self.in_channel)
        xs = xs.transpose(2, 3).contiguous()
        xs = xs.transpose(1, 2).contiguous()

        xs = self.layers(xs)
        # NOTE: xs: `[B, out_ch, new_time, new_freq]`

        # Collapse feature dimension
        bs, out_ch, time, freq = xs.size()
        xs = xs.transpose(1, 2).contiguous()
        xs = xs.view(bs, time, freq * out_ch)
        # NOTE: xs: `[B, new_time, new_freq * out_ch]`

        # Update xlens
        xlens = [self.get_conv_out_size(x_len, 1) for x_len in xlens]

        return xs, xlens
