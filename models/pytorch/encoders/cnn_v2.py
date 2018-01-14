#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""CNN encoder (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch.nn as nn

from models.pytorch.encoders.cnn_utils import ConvOutSize, Maxout


class CNNEncoder(nn.Module):
    """CNN encoder.
    Args:
        input_size (int): the dimension of input features
        conv_channels (list, optional): the number of channles in CNN layers
        conv_kernel_sizes (list, optional): the size of kernels in CNN layers
        conv_strides (list, optional): strides in CNN layers
        poolings (list, optional): the size of poolings in CNN layers
        dropout (float): the probability to drop nodes
        activation (string, optional): relu or prelu or hard_tanh or maxout
        batch_norm (bool, optional):
    """

    def __init__(self,
                 input_size,
                 conv_channels,
                 conv_kernel_sizes,
                 conv_strides,
                 poolings,
                 dropout,
                 activation='relu',
                 batch_norm=False):

        super(CNNEncoder, self).__init__()

        self.input_size = input_size
        self.input_channels = 3
        self.input_freq = input_size // self.input_channels

        assert input_size % self.input_channels == 0
        assert len(conv_channels) > 0
        assert len(conv_channels) == len(conv_kernel_sizes)
        assert len(conv_kernel_sizes) == len(conv_strides)
        assert len(conv_strides) == len(poolings)

        convs = []
        in_c = 1
        in_freq = input_size
        for i in range(len(conv_channels)):

            # Conv
            conv = nn.Conv2d(in_channels=in_c,
                             out_channels=conv_channels[i],
                             kernel_size=tuple(conv_kernel_sizes[i]),
                             stride=tuple(conv_strides[i]),
                             padding=tuple(conv_strides[i]),
                             bias=not batch_norm)
            convs.append(conv)
            in_freq = math.floor(
                (in_freq + 2 * conv.padding[0] - conv.kernel_size[0]) / conv.stride[0] + 1)

            # Activation
            if activation == 'relu':
                convs.append(nn.ReLU())
            elif activation == 'prelu':
                convs.append(nn.PReLU(num_parameters=1, init=0.2))
            elif activation == 'hard_tanh':
                convs.append(nn.Hardtanh(min_val=0, max_val=20, inplace=True))
            elif activation == 'maxout':
                convs.append(Maxout(1, 1, 2))
            else:
                raise NotImplementedError

            # Max Pooling
            if len(poolings[i]) > 0:
                pool = nn.MaxPool2d(
                    kernel_size=(poolings[i][0], poolings[i][0]),
                    stride=(poolings[i][0], poolings[i][1]),
                    # padding=(1, 1),
                    padding=(0, 0),  # default
                    ceil_mode=True)
                convs.append(pool)
                in_freq = math.floor(
                    (in_freq + 2 * pool.padding[0] - pool.kernel_size[0]) / pool.stride[0] + 1)

            # Batch Normalization
            if batch_norm:
                convs.append(nn.BatchNorm2d(conv_channels[i]))
                # TODO: compare BN before ReLU and after ReLU

            convs.append(nn.Dropout(p=dropout))
            in_c = conv_channels[i]

        self.conv = nn.Sequential(*convs)

        self.get_conv_out_size = ConvOutSize(self.conv)
        self.output_size = conv_channels[-1] * in_freq

    def forward(self, xs):
        """Forward computation.
        Args:
            xs (FloatTensor): A tensor of size `[B, T, input_size]`
        Returns:
            xs (FloatTensor): A tensor of size `[B, T', feature_dim]`
        """
        batch_size, max_time, input_size = xs.size()

        assert input_size == self.input_freq * self.input_channels

        # Reshape to 4D tensor
        xs = xs.transpose(1, 2).contiguous()
        xs = xs.unsqueeze(dim=1)
        # NOTE: xs: `[B, in_ch, freq, time]`

        xs = self.conv(xs)
        # print(xs.size())
        # NOTE: xs: `[B, out_ch, new_freq, new_time]`

        # Collapse feature dimension
        output_channels, freq, time = xs.size()[1:]
        xs = xs.transpose(1, 3).contiguous()
        xs = xs.view(batch_size, time, freq * output_channels)

        return xs
