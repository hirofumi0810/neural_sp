#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Sequence summary network."""

import logging
import torch
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
        param_init (float):

    """

    def __init__(self,
                 input_dim,
                 n_units,
                 n_layers,
                 bottleneck_dim,
                 dropout,
                 param_init=0.1):

        super(SequenceSummaryNetwork, self).__init__()

        self.n_layers = n_layers

        self.ssn = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        odim = input_dim
        for lth in range(n_layers - 1):
            self.ssn += [nn.Linear(odim, n_units)]
            odim = n_units
        self.ssn += [nn.Linear(odim, bottleneck_dim if bottleneck_dim > 0 else n_units)]
        self.proj = nn.Linear(bottleneck_dim if bottleneck_dim > 0 else n_units, input_dim)

        self.reset_parameters(param_init)

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_uniform(n, p, param_init)

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim (+Δ, ΔΔ)]`
            xlens (IntTensor): `[B]`
        Returns:
            xs (FloatTensor): `[B, T', input_dim]`

        """
        s = xs.clone()
        for lth in range(self.n_layers):
            s = self.dropout(torch.tanh(self.ssn[lth](s)))
        # `[B, T, input_dim]`

        # padding
        device_id = torch.cuda.device_of(next(self.parameters())).idx
        mask = make_pad_mask(xlens, device_id).unsqueeze(2)  # `[B, T, 1]`
        s = s.masked_fill_(mask == 0, 0)

        # time average
        denom = xlens.float().unsqueeze(1)
        if device_id >= 0:
            denom = denom.cuda(device_id)
        s = s.sum(1) / denom
        xs = xs + self.proj(s).unsqueeze(1)
        return xs
