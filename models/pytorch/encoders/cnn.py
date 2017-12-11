#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""CNN encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch.nn as nn


class CNNEncoder(nn.Module):
    """CNN encoder.
    Args:
        input_size (int): the dimension of input features
        num_stack (int, optional): the number of frames to stack
        splice (int, optional): frames to splice. Default is 1 frame.
        conv_channels (list, optional):
        conv_kernel_sizes (list, optional):
        conv_strides (list, optional):
        poolings (list, optional):
        dropout (float): the probability to drop nodes
        parameter_init (float): the range of uniform distribution to
            initialize weight parameters (>= 0)
        use_cuda (bool, optional): if True, use GPUs
        batch_norm (bool, optional):
    """

    def __init__(self,
                 input_size,
                 num_stack,
                 splice,
                 conv_channels,
                 conv_kernel_sizes,
                 conv_strides,
                 poolings,
                 dropout,
                 parameter_init,
                 use_cuda=False,
                 batch_norm=False):

        super(CNNEncoder, self).__init__()

        self.input_size = input_size
        self.splice = splice
        self.num_stack = num_stack
        self.input_channels = 3
        self.input_freq = input_size // self.input_channels

        assert input_size % self.input_channels == 0
        assert splice % 2 == 1, 'splice must be the odd number'
        assert len(conv_channels) > 0
        assert len(conv_channels) == len(conv_kernel_sizes)
        assert len(conv_kernel_sizes) == len(conv_strides)
        assert len(conv_strides) == len(poolings)

        convs = []
        in_c = self.input_channels
        in_freq = self.input_freq
        in_time = splice * num_stack
        for i in range(len(conv_channels)):
            assert conv_kernel_sizes[i][0] % 2 == 1
            assert conv_kernel_sizes[i][1] % 2 == 1

            # conv
            conv = nn.Conv2d(
                in_channels=in_c,
                out_channels=conv_channels[i],
                kernel_size=tuple(conv_kernel_sizes[i]),
                stride=tuple(conv_strides[i]),
                padding=(conv_kernel_sizes[i][0], conv_kernel_sizes[i][1]),
                bias=not batch_norm)
            convs.append(conv)
            in_freq = math.floor(
                (in_freq + 2 * conv.padding[0] - conv.kernel_size[0]) / conv.stride[0] + 1)
            in_time = math.floor(
                (in_time + 2 * conv.padding[1] - conv.kernel_size[1]) / conv.stride[1] + 1)

            # relu
            convs.append(nn.ReLU())

            # pooling
            if len(poolings[i]) > 0:
                pool = nn.MaxPool2d(
                    kernel_size=(poolings[i][0], poolings[i][0]),
                    stride=(poolings[i][0], poolings[i][1]),
                    padding=(1, 1))
                convs.append(pool)
                in_freq = math.floor(
                    (in_freq + 2 * pool.padding[0] - pool.kernel_size[0]) / pool.stride[0] + 1)
                in_time = math.floor(
                    (in_time + 2 * pool.padding[1] - pool.kernel_size[1]) / pool.stride[1] + 1)

            # batch normalization
            if batch_norm:
                convs.append(nn.BatchNorm2d(conv_channels[i]))
                # TODO: compare BN before ReLU and after ReLU

            convs.append(nn.Dropout(p=dropout))
            in_c = conv_channels[i]

        self.conv = nn.Sequential(*convs)
        # self.conv = convs

        self.output_size = conv_channels[-1] * in_freq * in_time

    def forward(self, inputs):
        """Forward computation.
        Args:
            inputs (FloatTensor): A tensor of size `[B, T, input_size]`
        Returns:
            outputs (FloatTensor): A tensor of size `[B, T, feature_dim]`
        """
        batch_size, max_time, input_size = inputs.size()

        # for debug
        # print('input_size: %d' % input_size)
        # print('input_freq: %d' % self.input_freq)
        # print('input_channels %d' % self.input_channels)
        # print('splice: %d' % self.splice)
        # print('num_stack: %d' % self.num_stack)

        assert input_size == self.input_freq * \
            self.input_channels * self.splice * self.num_stack

        # Reshape to 4D tensor
        inputs = inputs.view(
            batch_size * max_time, self.input_channels,
            self.input_freq, self.splice * self.num_stack)

        # print(inputs.size())
        outputs = self.conv(inputs)
        # print(outputs.size())

        # for debug
        # print(inputs.size())
        # outputs = inputs
        # for layer in self.conv:
        #     print(layer)
        #     outputs = layer(outputs)
        #     print(outputs.size())

        output_channels, freq, time = outputs.size()[1:]

        # Collapse feature dimension
        outputs = outputs.view(
            batch_size, -1, output_channels * freq * time)
        # print(outputs.size())

        return outputs
