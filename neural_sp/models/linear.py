#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""MLP & embedding layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearND(nn.Module):

    def __init__(self, in_size, out_size, bias=True, dropout=0):
        """Linear layer.

        A torch.nn.Linear layer modified to accept ND arrays.
            The function treats the last dimension of the input
            as the hidden dimension.
        Args:
            in_size (int):
            out_size (int):
            bias (bool): if False, remove a bias term
            dropout (float):

        """
        super(LinearND, self).__init__()

        self.fc = nn.Linear(in_size, out_size, bias=bias)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, xs):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim]`
        Returns:
            xs (FloatTensor): `[B, T, size[-1]]`

        """
        size = list(xs.size())
        xs = xs.contiguous().view((int(np.prod(size[:-1])), int(size[-1])))
        # print(self.fc.weight.data.sum())
        xs = self.fc(xs)
        if hasattr(self, 'dropout'):
            xs = self.dropout(xs)
        size[-1] = xs.size()[-1]
        return xs.view(size)


class Embedding(nn.Module):

    def __init__(self, vocab, emb_dim, dropout=0, ignore_index=-1, scale=False):
        """Embedding layer.

        Args:
            vocab (int): the number of nodes in softmax layer
                (including <sos> and <eos> classes)
            emb_dim (int): the dimension of the embedding in target spaces
            dropout (float): the probability to dropout nodes of the embedding
            ignore_index (int):
            scale (bool): if True, scale outputs of the embedding layer by square root of emb_dim

        """
        super(Embedding, self).__init__()

        self.emb_dim = emb_dim
        self.embed = nn.Embedding(vocab, emb_dim, padding_idx=ignore_index)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.scale = scale

    def forward(self, y):
        """Forward computation.

        Args:
            y (LongTensor): `[B, L]`
        Returns:
            y (FloatTensor): `[B, L, emb_dim]`

        """
        y = self.embed(y)
        if hasattr(self, 'dropout'):
            y = self.dropout(y)
        if self.scale:
            y *= math.sqrt(self.emb_dim)
        return y


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer.

    Args:
        d_model (int):
        dropout (float):
        max_len (int):

    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # for batch dimension
        self.pe = pe

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, xs):
        """Forward computation.

        Args:
            xs (FloatTensor):
        Returns:
            (FloatTensor):

        """
        device_id = xs.get_device()
        xs = xs + self.pe[:, :xs.size(1)].cuda(device_id)
        return self.dropout(xs)


class SublayerConnection(nn.Module):
    """A residual connection with dropout and layer normalization.

        input -> layer norm -> (residual) -> sublayer -> dropout -> add -> output
                                   |                                 |
                                   -----------------------------------
    Args:
        epsilon (float): epsilon parameter for layer normalization

    """

    def __init__(self, d_model, dropout, layer_norm=True, epsilon=1e-6):
        super(SublayerConnection, self).__init__()

        self.layer_norm = layer_norm
        if layer_norm:
            self.norm = nn.LayerNorm(d_model, eps=epsilon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xs, sublayer):
        """Apply residual connection to any sublayer with the same size.

        Args:
            xs (FloatTensor):
        Returns:
            xs (FloatTensor):

        """
        if self.layer_norm:
            xs = self.norm(xs)
        residual = xs

        out = sublayer(xs)
        # NOTE: out may be tuple

        rest_out = None
        if isinstance(out, tuple):
            xs = out[0]
            rest_out = out[1:]
        else:
            xs = out

        xs = self.dropout(xs)
        xs += residual

        if rest_out is not None:
            xs = (xs, rest_out)
        return xs


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
        """Forward computation.

        Args:
            xs (FloatTensor):
        Returns:
            (FloatTensor):

        """
        return self.w_2(self.dropout(F.relu(self.w_1(xs))))


class ResidualFeedForward(nn.Module):
    """Wrapper for the combination of SublayerConnection and PositionwiseFeedForward

        input -> layer norm -> (residual) -> PositionwiseFeedForward -> dropout -> add -> output
                                   |                                                |
                                   --------------------------------------------------

    Args:
        d_model (int):
        d_ff (int):
        dropout (float):
        layer_norm (bool):
        epsilon (float):

    """

    def __init__(self, d_model, d_ff, dropout, layer_norm=False, epsilon=1e-6):
        super(ResidualFeedForward, self).__init__()

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm = SublayerConnection(d_model, dropout, layer_norm, epsilon)

    def forward(self, xs):
        """Forward computation.

        Args:
            xs (FloatTensor):
        Returns:
            (FloatTensor):

        """
        return self.add_norm(xs, self.feed_forward)
