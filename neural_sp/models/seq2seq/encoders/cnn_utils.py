#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utilitiely functions for CNN encoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch.nn as nn


class ConvOutSize(object):
    """TODO."""

    def __init__(self, conv):
        super(ConvOutSize, self).__init__()
        self.conv = conv

        if self.conv is None:
            raise ValueError

    def __call__(self, size, dim):
        """

        Args:
            size (int):
            dim (int): dim == 0 means frequency dimension, dim == 1 means
                time dimension.
        Returns:
            size (int):

        """
        for m in self.conv._modules.values():
            if type(m) in [nn.Conv2d, nn.MaxPool2d]:
                if type(m) == nn.MaxPool2d and not m.ceil_mode:
                    # first max pool layer
                    size = int(math.floor(
                        (size + 2 * m.padding[dim] - m.kernel_size[dim]) / m.stride[dim] + 1))
                else:
                    size = int(math.ceil(
                        (size + 2 * m.padding[dim] - m.kernel_size[dim]) / m.stride[dim] + 1))
                # assert size >= 1
        return size


class Maxout(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super(Maxout, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.pool_size = pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        # print(inputs.size())
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        # print(out.size())
        m, i = out.view(*shape).max(max_dim)
        return m

    def __repr__(self):
        return 'maxout'
