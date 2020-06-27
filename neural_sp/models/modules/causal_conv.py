#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Dilated causal convolution."""

import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class CausalConv1d(nn.Module):
    """1D dilated causal convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
                                padding=self.padding, dilation=dilation)

    def forward(self, xs):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, in_channels]`
        Returns:
            xs (FloatTensor): `[B, T, out_channels]`

        """
        xs = xs.transpose(2, 1)
        xs = self.conv1d(xs)
        if self.padding != 0:
            xs = xs[:, :, :-self.padding]
        xs = xs.transpose(2, 1).contiguous()
        return xs
