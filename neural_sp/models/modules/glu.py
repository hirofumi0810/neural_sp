#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Gated Linear Units (GLU) block."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLUBlock(nn.Module):
    """GLU block.

    Args:
        kernel_size (int): kernel size
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        bottlececk_dim (int): dimension of the bottleneck layers for computational efficiency
        weight_norm (bool): weight normalization
        dropout (float): dropout probability

    """

    def __init__(self, kernel_size, in_ch, out_ch, bottlececk_dim=0,
                 weight_norm=True, dropout=0.0):
        super().__init__()

        self.conv_residual = None
        if in_ch != out_ch:
            self.conv_residual = nn.Conv2d(in_channels=in_ch,
                                           out_channels=out_ch,
                                           kernel_size=(1, 1))
            if weight_norm:
                self.conv_residual = nn.utils.weight_norm(self.conv_residual,
                                                          name='weight', dim=0)

        self.pad_left = nn.ConstantPad2d((0, 0, kernel_size - 1, 0), 0)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_gate = nn.Dropout(p=dropout)

        if bottlececk_dim == 0:
            self.conv_in = lambda x: x
            self.conv = nn.Conv2d(in_channels=in_ch,
                                  out_channels=out_ch,
                                  kernel_size=(kernel_size, 1))
            self.conv_out = lambda x: x

            self.conv_gate_in = lambda x: x
            self.conv_gate = nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch,
                                       kernel_size=(kernel_size, 1))
            self.conv_gate_out = lambda x: x

            if weight_norm:
                self.conv = nn.utils.weight_norm(self.conv,
                                                 name='weight', dim=0)
                self.conv_gate = nn.utils.weight_norm(self.conv_gate,
                                                      name='weight', dim=0)

        elif bottlececk_dim > 0:
            self.conv_in = nn.Conv2d(in_channels=in_ch,
                                     out_channels=bottlececk_dim,
                                     kernel_size=(1, 1))
            self.conv = nn.Conv2d(in_channels=bottlececk_dim,
                                  out_channels=bottlececk_dim,
                                  kernel_size=(kernel_size, 1))
            self.conv_out = nn.Conv2d(in_channels=bottlececk_dim,
                                      out_channels=out_ch,
                                      kernel_size=(1, 1))

            self.conv_gate_in = nn.Conv2d(in_channels=in_ch,
                                          out_channels=bottlececk_dim,
                                          kernel_size=(1, 1))
            self.conv_gate = nn.Conv2d(in_channels=bottlececk_dim,
                                       out_channels=bottlececk_dim,
                                       kernel_size=(kernel_size, 1))
            self.conv_gate_out = nn.Conv2d(in_channels=bottlececk_dim,
                                           out_channels=out_ch,
                                           kernel_size=(1, 1))

            if weight_norm:
                self.conv_in = nn.utils.weight_norm(self.conv_in,
                                                    name='weight', dim=0)
                self.conv = nn.utils.weight_norm(self.conv,
                                                 name='weight', dim=0)
                self.conv_out = nn.utils.weight_norm(self.conv_out,
                                                     name='weight', dim=0)

                self.conv_gate_in = nn.utils.weight_norm(self.conv_gate_in,
                                                         name='weight', dim=0)
                self.conv_gate = nn.utils.weight_norm(self.conv_gate,
                                                      name='weight', dim=0)
                self.conv_gate_out = nn.utils.weight_norm(self.conv_gate_out,
                                                          name='weight', dim=0)

    def forward(self, x):
        """Forward computation.
        Args:
            x (FloatTensor): `[B, in_ch, T]`
        Returns:
            out (FloatTensor): `[B, out_ch, T]`
        """
        residual = x
        if self.conv_residual is not None:
            residual = self.conv_residual(residual)
        x = self.pad_left(x)  # `[B, embed_dim, T+kernel-1, 1]`
        a = self.conv_out(self.conv(self.conv_in(x)))  # `[B, out_ch, T ,1]`
        a = self.dropout(a)
        b = self.conv_gate_out(self.conv_gate(self.conv_gate_in(x)))  # `[B, out_ch, T, 1]`
        b = self.dropout_gate(b)

        x = torch.mul(a, F.sigmoid(b))
        x += residual
        return x
