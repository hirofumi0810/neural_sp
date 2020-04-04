#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Relative multi-head attention layer for TransformerXL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import numpy as np
import random
import torch
import torch.nn as nn

random.seed(1)

NEG_INF = float(np.finfo(np.float32).min)

logger = logging.getLogger(__name__)


class RelativeMultiheadAttentionMechanism(nn.Module):
    """Relative multi-head attention layer for TransformerXL.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of the attention space
        n_heads (int): number of heads
        dropout (float): dropout probability for attenion weights
        bias (bool): use bias term in linear layers
        param_init (str): parameter initialization method

    """

    def __init__(self, kdim, qdim, adim, n_heads, dropout,
                 bias=True, param_init=''):
        super(RelativeMultiheadAttentionMechanism, self).__init__()

        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(self.d_k)

        # attention dropout applied AFTER the softmax layer
        self.dropout = nn.Dropout(p=dropout)

        self.w_key = nn.Linear(kdim, adim, bias=bias)
        self.w_value = nn.Linear(kdim, adim, bias=bias)
        self.w_query = nn.Linear(qdim, adim, bias=bias)
        self.w_position = nn.Linear(qdim, adim, bias=bias)
        # TODO: fix later
        self.w_out = nn.Linear(adim, kdim, bias=bias)

        if param_init == 'xavier_uniform':
            self.reset_parameters(bias)

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        # NOTE: see https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_value.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.)
            nn.init.constant_(self.w_value.bias, 0.)
            nn.init.constant_(self.w_query.bias, 0.)

        nn.init.xavier_uniform_(self.w_out.weight)
        if bias:
            nn.init.constant_(self.w_out.bias, 0.)

    def _rel_shift(self, xs):
        """Calculate relative positional attention efficiently.

        Args:
            xs (FloatTensor): `[B, qlen, klen, H]`
        Returns:
            xs_shifted (FloatTensor): `[B, qlen, klen, H]`

        """
        bs, qlen, klen, n_heads = xs.size()
        # `[qlen, klen, B, H]` -> `[B, qlen, klen, H]`
        xs = xs.permute(1, 2, 0, 3).contiguous().view(qlen, klen, bs * n_heads)

        zero_pad = xs.new_zeros((qlen, 1, bs * n_heads))
        xs_shifted = (torch.cat([zero_pad, xs], dim=1)
                      .view(klen + 1, qlen, bs * n_heads)[1:]
                      .view_as(xs))
        return xs_shifted.view(qlen, klen, bs, n_heads).permute(2, 0, 1, 3)

    def forward(self, key, query, memory, pos_embs, mask, u, v):
        """Forward computation.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            pos_embs (LongTensor): `[qlen, 1, d_model]`
            memory (FloatTensor): `[B, mlen, d_model]`
            mask (ByteTensor): `[B, qlen, klen]`
            u (nn.Parameter): `[H, d_k]`
            v (nn.Parameter): `[H, d_k]`
        Returns:
            cv (FloatTensor): `[B, qlen, vdim]`
            aw (FloatTensor): `[B, H, qlen, klen]`

        """
        bs, qlen = query.size()[: 2]
        klen = key.size(1)
        mlen = memory.size(1) if memory.dim() > 1 else 0
        if mlen > 0:
            key = torch.cat([memory, key], dim=1)

        value = self.w_value(key).view(bs, -1, self.n_heads, self.d_k)  # `[B, klen+mlen, H, d_k]`
        key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)  # `[B, klen+mlen, H, d_k]`
        if mask is not None:
            mask = mask.unsqueeze(3).repeat([1, 1, 1, self.n_heads])
            assert mask.size() == (bs, qlen, mlen + klen, self.n_heads), \
                (mask.size(), (bs, qlen, klen + mlen, self.n_heads))

        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)  # `[B, qlen, H, d_k]`
        pos_embs = self.w_position(pos_embs)
        pos_embs = pos_embs.view(-1, self.n_heads, self.d_k)  # `[qlen, H, d_k]`

        # content-based attention term: (a) + (c)
        AC = torch.einsum("bihd,bjhd->bijh", ((query + u[None, None]), key))  # `[B, qlen, klen+mlen, H]`

        # position-based attention term: (b) + (d)
        BD = torch.einsum("bihd,jhd->bijh", ((query + v[None, None]), pos_embs))  # `[B, qlen, klen+mlen, H]`

        # Compute positional attention efficiently
        BD = self._rel_shift(BD)

        # the attention is the sum of content-based and position-based attention
        e = (AC + BD) / self.scale  # `[B, qlen, klen, H]`

        # Compute attention weights
        if mask is not None:
            e = e.masked_fill_(mask == 0, NEG_INF)  # `[B, qlen, klen, H]`
        aw = torch.softmax(e, dim=2)
        aw = self.dropout(aw)
        cv = torch.einsum("bijh,bjhd->bihd", (aw, value))  # `[B, qlen, H, d_k]`
        cv = cv.contiguous().view(bs, -1, self.n_heads * self.d_k)  # `[B, qlen, H * d_k]`
        cv = self.w_out(cv)
        aw = aw.permute(0, 3, 1, 2)  # `[B, H, qlen, klen]`

        return cv, aw
