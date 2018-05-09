#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""CNN encoder (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from models.pytorch.encoders.cnn_utils import ConvOutSize, Maxout


class CNNEncoder(nn.Module):
    """CNN encoder.
    Args:
        input_size (int): the dimension of input features (freq * channel)
        input_channel (int, optional): the number of channels of input features
        conv_channels (list, optional): the number of channles in CNN layers
        conv_kernel_sizes (list, optional): the size of kernels in CNN layers
        conv_strides (list, optional): strides in CNN layers
        poolings (list, optional): the size of poolings in CNN layers
        dropout_input (float): the probability to drop nodes in input-hidden connection
        dropout_hidden (float): the probability to drop nodes in hidden-hidden connection
        activation (string, optional): relu or prelu or hard_tanh or maxout
        batch_norm (bool, optional): if True, apply batch normalization
    """

    def __init__(self,
                 input_size,
                 input_channel,
                 conv_channels,
                 conv_kernel_sizes,
                 conv_strides,
                 poolings,
                 dropout_input,
                 dropout_hidden,
                 activation='relu',
                 batch_norm=False):

        super(CNNEncoder, self).__init__()

        self.input_channel = input_channel
        assert input_size % input_channel == 0
        self.input_freq = input_size // input_channel

        assert len(conv_channels) > 0
        assert len(conv_channels) == len(conv_kernel_sizes)
        assert len(conv_kernel_sizes) == len(conv_strides)
        assert len(conv_strides) == len(poolings)

        # Dropout for input-hidden connection
        self.dropout_input = nn.Dropout(p=dropout_input)

        layers = []
        in_c = self.input_channel
        in_freq = self.input_freq
        first_max_pool = True
        for l in range(len(conv_channels)):

            # Conv
            conv = nn.Conv2d(in_channels=in_c,
                             out_channels=conv_channels[l],
                             kernel_size=tuple(conv_kernel_sizes[l]),
                             stride=tuple(conv_strides[l]),
                             padding=tuple(conv_strides[l]),
                             bias=not batch_norm)
            layers.append(conv)
            in_freq = math.floor(
                (in_freq + 2 * conv.padding[0] - conv.kernel_size[0]) / conv.stride[0] + 1)

            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'prelu':
                layers.append(nn.PReLU(num_parameters=1, init=0.2))
            elif activation == 'hard_tanh':
                layers.append(nn.Hardtanh(min_val=0, max_val=20, inplace=True))
            elif activation == 'maxout':
                layers.append(Maxout(1, 1, 2))
            else:
                raise NotImplementedError

            # Max Pooling
            if len(poolings[l]) > 0:
                pool = nn.MaxPool2d(kernel_size=tuple(poolings[l]),
                                    stride=tuple(poolings[l]),
                                    # padding=(1, 1),
                                    padding=(0, 0),  # default
                                    ceil_mode=False if first_max_pool else True)
                # NOTE: If ceil_mode is False, remove last feature when the
                # dimension of features are odd.
                first_max_pool = False
                # NOTE: This is important for having the same frames as RNN models

                layers.append(pool)
                in_freq = math.floor(
                    (in_freq + 2 * pool.padding[0] - pool.kernel_size[0]) / pool.stride[0] + 1)

            # Batch Normalization
            if batch_norm:
                layers.append(nn.BatchNorm2d(conv_channels[l]))

            # Dropout for hidden-hidden connection
            layers.append(nn.Dropout(p=dropout_hidden))
            # TODO: compare BN before ReLU and after ReLU

            # TODO: try this
            # layers.append(nn.Dropout2d(p=dropout_hidden))

            in_c = conv_channels[l]

        self.layers = nn.Sequential(*layers)

        self.get_conv_out_size = ConvOutSize(self.layers)
        self.output_size = conv_channels[-1] * in_freq

    def forward(self, xs, x_lens):
        """Forward computation.
        Args:
            xs (torch.autograd.Variable, float): A tensor of size
                `[B, T, input_size (+Δ, ΔΔ)]`
            x_lens (torch.autograd.Variable, int): A tensor of size `[B]`
        Returns:
            xs (torch.autograd.Variable, float): A tensor of size
                `[B, T', feature_dim]`
            x_lens (torch.autograd.Variable, int): A tensor of size `[B]`
        """
        batch_size, max_time, input_size = xs.size()

        # assert input_size == self.input_freq * self.input_channel

        # Dropout for inputs-hidden connection
        xs = self.dropout_input(xs)

        # Reshape to 4D tensor
        xs = xs.transpose(1, 2).contiguous()
        if self.input_channel == 3:
            xs = xs.view(batch_size, 3, input_size // 3, max_time)
            # NOTE: xs: `[B, in_ch (3), freq // 3, max_time]`
        else:
            xs = xs.unsqueeze(1)
            # NOTE: xs: `[B, in_ch (1), freq, max_time]`

        xs = self.layers(xs)
        # NOTE: xs: `[B, out_ch, new_freq, new_time]`

        # Collapse feature dimension
        output_channels, freq, time = xs.size()[1:]
        xs = xs.transpose(1, 3).contiguous()
        xs = xs.view(batch_size, time, freq * output_channels)

        # Update x_lens
        x_lens = np.array([self.get_conv_out_size(x, 1) for x in x_lens])
        x_lens = Variable(torch.from_numpy(x_lens).int(), requires_grad=False)
        if xs.is_cuda:
            x_lens = x_lens.cuda()

        return xs, x_lens
