#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utilities for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer.

    Args:
        d_model (int):
        dropout (float):
        pe_type (str):
        max_len (int):

    """

    def __init__(self, d_model, dropout, pe_type, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.pe_type = pe_type

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # for batch dimension
        # self.pe = pe
        self.register_buffer('pe', pe)

        # TODO(hirofumi): add concat option

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, xs):
        if self.pe_type == 'add':
            xs = xs + self.pe[:, :xs.size(1)]
        elif self.pe_type == 'concat':
            xs = torch.cat([xs, self.pe[:, :xs.size(1)]], dim=-1)
        else:
            raise NotImplementedError
        return self.dropout(xs)


class SublayerConnection(nn.Module):
    """A residual connection with dropout and layer normalization.

        input -> layer norm -> sublayer -> dropout -> output ->
          |                                         |
          -------------------------------------------
    Args:
        layer_norm_eps (float): epsilon parameter for layer normalization

    """

    def __init__(self, d_model, dropout, layer_norm_eps=1e-6):
        super(SublayerConnection, self).__init__()

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # self.layer_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xs, sublayer):
        xs_norm = self.layer_norm(xs)
        output = sublayer(xs_norm)

        # NOTE: output may be tuple paired with attention weights
        if isinstance(output, tuple):
            xs_norm, aw = output
        else:
            xs_norm = output
        xs_norm = self.dropout(xs_norm)
        xs_norm = xs_norm + xs

        if isinstance(output, tuple):
            return xs_norm, aw
        else:
            return xs_norm


class PositionwiseFeedForward(nn.Module):
    """Positionwise fully-connected feed-forward neural network.

    Args:
        d_model (int):
        d_ff (int):
        dropout (float):

    """

    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xs):
        return self.w_2(self.dropout(F.relu(self.w_1(xs))))


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
