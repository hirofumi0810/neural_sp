#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utilitiely functions for CNN encoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.nn as nn


class ConvOutSize(object):
    """Return the size of outputs for CNN layers."""

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
                if type(m) == nn.MaxPool2d:
                    if m.ceil_mode:
                        size = int(np.ceil(
                            (size + 2 * m.padding[dim] - m.kernel_size[dim]) / m.stride[dim] + 1))
                    else:
                        size = int(np.floor(
                            (size + 2 * m.padding[dim] - m.kernel_size[dim]) / m.stride[dim] + 1))
                else:
                    size = int(np.floor(
                        (size + 2 * m.padding[dim] - m.kernel_size[dim]) / m.stride[dim] + 1))
                # assert size >= 1
        return size
