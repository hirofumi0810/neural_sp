#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Sequence summary network."""

import logging
import torch.nn as nn

from neural_sp.models.modules.initialization import init_with_uniform
from neural_sp.models.torch_utils import make_pad_mask

logger = logging.getLogger(__name__)


class SequenceSummaryNetwork(nn.Module):
    """Sequence summary network.

    Args:
        input_dim (int): dimension of input features
        n_units (int):
        n_layers (int):
        bottleneck_dim (int): dimension of the last bottleneck layer
        dropout (float): dropout probability
        param_init (str): parameter initialization method

    """

    def __init__(self, input_dim, n_units, n_layers, bottleneck_dim,
                 dropout, param_init=0.1):

        super(SequenceSummaryNetwork, self).__init__()

        self.n_layers = n_layers

        layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        odim = input_dim
        for lth in range(n_layers - 1):
            layers += [nn.Linear(odim, n_units)]
            layers += [nn.Tanh()]
            layers += [nn.Dropout(p=dropout)]
            odim = n_units
        layers += [nn.Linear(odim, bottleneck_dim if bottleneck_dim > 0 else n_units)]
        layers += [nn.Tanh()]
        layers += [nn.Dropout(p=dropout)]
        self.layers = nn.Sequential(*layers)
        self.proj = nn.Linear(bottleneck_dim if bottleneck_dim > 0 else n_units, input_dim)

        self.reset_parameters(param_init)

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_uniform(n, p, param_init)

    def forward(self, xs, xlens):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, input_dim (+Δ, ΔΔ)]`
            xlens (IntTensor): `[B]`
        Returns:
            xs (FloatTensor): `[B, T, input_dim]`

        """
        residual = xs
        xs = self.layers(xs)  # `[B, T, input_dim]`

        # padding
        xlens = xlens.to(xs.device)
        mask = make_pad_mask(xlens).unsqueeze(2)  # `[B, T, 1]`
        xs = xs.clone().masked_fill_(mask == 0, 0)

        # time average
        denom = xlens.float().unsqueeze(1)
        xs = xs.sum(1) / denom
        xs = residual + self.proj(xs).unsqueeze(1)
        return xs
