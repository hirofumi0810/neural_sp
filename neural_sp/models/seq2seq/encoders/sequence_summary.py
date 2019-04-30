#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Sequence summayr network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from neural_sp.models.model_utils import LinearND


class SequenceSummaryNetwork(nn.Module):
    """Sequence summayr network.

    Args:
        input_dim (int): dimension of input features
        n_units (int):
        n_layers (int):
        bottleneck_dim (int): dimension of the last bottleneck layer
        dropout (float):

    """

    def __init__(self,
                 input_dim,
                 n_units,
                 n_layers,
                 bottleneck_dim,
                 dropout):

        super(SequenceSummaryNetwork, self).__init__()

        self.n_layers = n_layers

        self.ssn = nn.ModuleList()
        self.ssn += [LinearND(input_dim, n_units, bias=False, dropout=dropout)]
        for l in range(1, n_layers - 1, 1):
            self.ssn += [LinearND(n_units, bottleneck_dim if l == n_layers - 2 else n_units,
                                  bias=False, dropout=dropout)]
        self.ssn += [LinearND(bottleneck_dim, input_dim, bias=False, dropout=dropout)]

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim (+Δ, ΔΔ)]`
            xlens (list): A list of length `[B]`
        Returns:
            xs (FloatTensor): `[B, T', input_dim]`

        """
        bs, time = xs.size()[:2]

        aw_ave = xs.new_zeros((bs, 1, time))
        for b in range(bs):
            aw_ave[b, :, :xlens[b]] = 1 / xlens[b]

        s = xs.clone()
        for l in range(self.n_layers - 1):
            s = torch.tanh(self.ssn[l](s))  # `[B, T, bottleneck_dim]`

        # time average
        s = torch.matmul(aw_ave, s)  # `[B, 1, bottleneck_dim]`
        xs += torch.tanh(self.ssn[self.n_layers - 1](s))

        return xs
