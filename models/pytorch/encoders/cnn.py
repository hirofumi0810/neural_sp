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
        channels (list, optional):
        kernel_sizes (list, optional):
        strides (list, optional):
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
                 channels,
                 kernel_sizes,
                 strides,
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
        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes)
        assert len(kernel_sizes) == len(strides)

        convs = []
        in_c = self.input_channels
        for i in range(len(channels)):
            assert kernel_sizes[i][0] % 2 == 1
            assert kernel_sizes[i][1] % 2 == 1

            convs.append(nn.Conv2d(
                in_channels=in_c,
                out_channels=channels[i],
                kernel_size=tuple(kernel_sizes[i]),
                stride=tuple(strides[i]),
                padding=(kernel_sizes[i][0] // 2, kernel_sizes[i][1] // 2),
                bias=not batch_norm))
            convs.append(nn.ReLU())
            if batch_norm:
                convs.append(nn.BatchNorm2d(channels[i]))
                # TODO: compare BN before ReLU and after ReLU
            convs.append(nn.Dropout(p=dropout))
            in_c = channels[i]
        self.conv = nn.Sequential(*convs)

        out_freq = self.input_freq
        out_time = splice * num_stack
        for f, t in strides:
            out_freq = math.ceil(out_freq / f)
            out_time = math.ceil(out_time / t)
        self.output_size = channels[-1] * out_freq * out_time

    def forward(self, inputs):
        """Forward computation.
        Args:
            inputs (FloatTensor): A tensor of size `[B, T, input_size]`
        Returns:
            outputs (FloatTensor): A tensor of size `[B]`
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
        # print(inputs.size())

        output_channels, freq, time = outputs.size()[1:]

        # Collapse feature dimension
        outputs = outputs.view(
            batch_size, -1, output_channels * freq * time)
        # print(outputs.size())

        return outputs
