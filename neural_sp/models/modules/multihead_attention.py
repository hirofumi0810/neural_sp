#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-head attention layer."""

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


class MultiheadAttentionMechanism(nn.Module):
    """Multi-headed attention layer.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of the attention space
        n_heads (int): number of heads
        dropout (float): dropout probability for attenion weights
        atype (str): type of attention mechanism
        bias (bool): use bias term in linear layers
        param_init (str): parameter initialization method

    """

    def __init__(self, kdim, qdim, adim, n_heads, dropout,
                 atype='scaled_dot', bias=True, param_init=''):
        super(MultiheadAttentionMechanism, self).__init__()

        self.atype = atype
        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(self.d_k)
        self.reset()

        # attention dropout applied AFTER the softmax layer
        self.dropout = nn.Dropout(p=dropout)

        if atype == 'scaled_dot':
            # for Transformer
            self.w_key = nn.Linear(kdim, adim, bias=bias)
            self.w_value = nn.Linear(kdim, adim, bias=bias)
            self.w_query = nn.Linear(qdim, adim, bias=bias)
        elif atype == 'add':
            # for LAS
            self.w_key = nn.Linear(kdim, adim, bias=bias)
            self.w_value = nn.Linear(kdim, adim, bias=bias)
            self.w_query = nn.Linear(qdim, adim, bias=bias)
            self.v = nn.Linear(adim, n_heads, bias=bias)
        else:
            raise NotImplementedError(atype)

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

    def reset(self):
        self.key = None
        self.value = None
        self.mask = None

    def forward(self, key, value, query, mask, aw_prev=None,
                cache=False, mode='', trigger_point=None,
                eps_wait=-1, boundary_rightmost=None):
        """Forward pass.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            value (FloatTensor): `[B, klen, vdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            aw_prev: dummy interface
            cache (bool): cache key, value, and mask
            mode: dummy interface for MoChA
            trigger_point: dummy interface for MoChA
            eps_wait: dummy interface for MMA
        Returns:
            cv (FloatTensor): `[B, qlen, vdim]`
            aw (FloatTensor): `[B, H, qlen, klen]`
            beta: dummy interface for MoChA

        """
        bs, klen = key.size()[: 2]
        qlen = query.size(1)

        if self.key is None or not cache:
            self.key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)  # `[B, klen, H, d_k]`
            self.value = self.w_value(value).view(bs, -1, self.n_heads, self.d_k)  # `[B, klen, H, d_k]`
            self.mask = mask
            if self.mask is not None:
                self.mask = self.mask.unsqueeze(3).repeat([1, 1, 1, self.n_heads])
                assert self.mask.size() == (bs, qlen, klen, self.n_heads), \
                    (self.mask.size(), (bs, qlen, klen, self.n_heads))

        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)  # `[B, qlen, H, d_k]`

        if self.atype == 'scaled_dot':
            e = torch.einsum("bihd,bjhd->bijh", (query, self.key)) / self.scale  # `[B, qlen, klen, H]`
        elif self.atype == 'add':
            key = self.key.unsqueeze(1)  # `[B, 1, klen, H, d_k]`
            query = query.unsqueeze(2)  # `[B, qlen, 1, H, d_k]`
            e = self.v(torch.tanh(key + query)).squeeze(4)  # `[B, qlen, klen, H]`

        # Compute attention weights
        if self.mask is not None:
            e = e.masked_fill_(self.mask == 0, NEG_INF)  # `[B, qlen, klen, H]`
        aw = torch.softmax(e, dim=2)
        aw = self.dropout(aw)
        cv = torch.einsum("bijh,bjhd->bihd", (aw, self.value))  # `[B, qlen, H, d_k]`
        cv = cv.contiguous().view(bs, -1, self.n_heads * self.d_k)  # `[B, qlen, H * d_k]`
        cv = self.w_out(cv)
        aw = aw.permute(0, 3, 1, 2)  # `[B, H, qlen, klen]`

        return cv, aw, None
