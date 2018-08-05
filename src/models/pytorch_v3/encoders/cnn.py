#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""CNN encoder (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import math

import torch

from src.models.pytorch_v3.encoders.cnn_utils import ConvOutSize
from src.models.pytorch_v3.encoders.cnn_utils import Maxout


class CNNEncoder(torch.nn.Module):
    """CNN encoder.

    Args:
        input_size (int): the dimension of input features (freq * channel)
        in_channel (int): the number of channels of input features
        channels (list): the number of channles in CNN layers
        kernel_sizes (list): the size of kernels in CNN layers
        strides (list): strides in CNN layers
        poolings (list): the size of poolings in CNN layers
        dropout_in (float): the probability to drop nodes in input-hidden connection
        dropout_hidden (float): the probability to drop nodes in hidden-hidden connection
        activation (string): relu or prelu or hard_tanh or maxout
        batch_norm (bool): if True, apply batch normalization

    """

    def __init__(self,
                 input_size,
                 in_channel,
                 channels,
                 kernel_sizes,
                 strides,
                 poolings,
                 dropout_in,
                 dropout_hidden,
                 activation='relu',
                 batch_norm=False):

        super(CNNEncoder, self).__init__()

        self.in_channel = in_channel
        assert input_size % in_channel == 0
        self.input_freq = input_size // in_channel

        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes)
        assert len(kernel_sizes) == len(strides)
        assert len(strides) == len(poolings)

        # Dropout for input-hidden connection
        self.dropout_in = torch.nn.Dropout(p=dropout_in)

        layers = OrderedDict()
        in_ch = self.in_channel
        in_freq = self.input_freq
        first_max_pool = True
        for l in range(len(channels)):

            # Conv
            conv = torch.nn.Conv2d(in_channels=in_ch,
                                   out_channels=channels[l],
                                   kernel_size=tuple(kernel_sizes[l]),
                                   stride=tuple(strides[l]),
                                   padding=tuple(strides[l]),
                                   bias=not batch_norm)
            layers['conv_' + str(l)] = conv
            in_freq = math.floor(
                (in_freq + 2 * conv.padding[0] - conv.kernel_size[0]) / conv.stride[0] + 1)

            # Activation
            if activation == 'relu':
                act = torch.nn.ReLU()
            elif activation == 'prelu':
                act = torch.nn.PReLU(num_parameters=1, init=0.2)
            elif activation == 'hard_tanh':
                act = torch.nn.Hardtanh(min_val=0, max_val=20, inplace=True)
            elif activation == 'maxout':
                act = Maxout(1, 1, 2)
            else:
                raise NotImplementedError
            layers[activation + '_' + str(l)] = act

            # Max Pooling
            if len(poolings[l]) > 0:
                pool = torch.nn.MaxPool2d(kernel_size=tuple(poolings[l]),
                                          stride=tuple(poolings[l]),
                                          # padding=(1, 1),
                                          padding=(0, 0),  # default
                                          ceil_mode=not first_max_pool)
                layers['pool_' + str(l)] = pool
                # NOTE: If ceil_mode is False, remove last feature when the
                # dimension of features are odd.

                first_max_pool = False
                # NOTE: This is important for having the same frames as RNN models

                if pool.ceil_mode:
                    in_freq = math.ceil(
                        (in_freq + 2 * pool.padding[0] - pool.kernel_size[0]) / pool.stride[0] + 1)
                else:
                    in_freq = math.floor(
                        (in_freq + 2 * pool.padding[0] - pool.kernel_size[0]) / pool.stride[0] + 1)

            # Batch Normalization
            if batch_norm:
                layers['bn_' + str(l)] = torch.nn.BatchNorm2d(channels[l])

            # Dropout for hidden-hidden connection
            layers['dropout_' + str(l)] = torch.nn.Dropout(p=dropout_hidden)
            # TODO(hirofumi): compare BN before ReLU and after ReLU

            # TODO(hirofumi): try this
            # layers.append(torch.nn.Dropout2d(p=dropout_hidden))

            in_ch = channels[l]

        self.layers = torch.nn.Sequential(layers)

        self.get_conv_out_size = ConvOutSize(self.layers)
        self.output_size = channels[-1] * in_freq

    def forward(self, xs, x_lens):
        """Forward computation.
        Args:
            xs (torch.autograd.Variable, float): A tensor of size
                `[B, T, input_size (+Δ, ΔΔ)]`
            x_lens (list): A list of length `[B]`
        Returns:
            xs (torch.autograd.Variable, float): A tensor of size
                `[B, T', feature_dim]`
            x_lens (list): A list of length `[B]`
        """
        batch, max_time, input_size = xs.size()
        # assert input_size == self.input_freq * self.in_channel

        # Dropout for inputs-hidden connection
        xs = self.dropout_in(xs)

        # Reshape to 4D tensor
        xs = xs.transpose(1, 2).contiguous()
        if self.in_channel > 1:
            xs = xs.view(batch, self.in_channel,
                         input_size // self.in_channel, max_time)
            # NOTE: xs: `[B, in_ch (3), freq // in_ch, max_time]`
        else:
            xs = xs.unsqueeze(1)
            # NOTE: xs: `[B, in_ch (1), freq, max_time]`

        xs = self.layers(xs)
        # NOTE: xs: `[B, out_ch, new_freq, new_time]`

        # Collapse feature dimension
        output_channels, freq, time = xs.size()[1:]
        xs = xs.transpose(1, 3).contiguous()
        xs = xs.view(batch, time, freq * output_channels)

        # Update x_lens
        x_lens = [self.get_conv_out_size(x_len, 1) for x_len in x_lens]

        return xs, x_lens
