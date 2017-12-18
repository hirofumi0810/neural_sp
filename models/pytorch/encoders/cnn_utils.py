#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilitiels for CNN encoders."""

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

    def __call__(self, size, dim):
        """
        Args:
            size (int):
            dim (int): dim == 0 means frequency dimension, dim == 1 means
                time dimension.
        Returns:
            size (int):
        """
        if self.conv is None:
            return size

        for m in self.conv._modules.values():
            if type(m) in [nn.Conv2d, nn.MaxPool2d]:
                size = math.floor(
                    (size + 2 * m.padding[dim] - m.kernel_size[dim]) / m.stride[dim] + 1)

        assert size >= 1

        return size
