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
import torch
import torch.nn as nn

NEG_INF = float(np.finfo(np.float32).min)

logger = logging.getLogger(__name__)


class MultiheadAttentionMechanism(nn.Module):
    """Multi-headed attention layer.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of the attention space
        atype (str): type of attention mechanisms
        dropout (float): dropout probability
        n_heads (int): number of heads
        bias (bool): use bias term in linear layers
        param_init (str):

    """

    def __init__(self, kdim, qdim, adim, atype, dropout=0., n_heads=4, bias=True,
                 param_init=''):
        super(MultiheadAttentionMechanism, self).__init__()

        self.atype = atype
        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.reset()

        # attention dropout applied AFTER the softmax layer
        self.attn_dropout = nn.Dropout(p=dropout)

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
            # newly introduced
            nn.init.constant_(self.w_key.bias, 0.)
            nn.init.constant_(self.w_value.bias, 0.)
            nn.init.constant_(self.w_query.bias, 0.)

        nn.init.xavier_uniform_(self.w_out.weight)
        if bias:
            nn.init.constant_(self.w_out.bias, 0.)

    def reset(self):
        self.key_fwd = None
        self.key_bwd = None
        self.value_fwd = None
        self.value_bwd = None
        self.mask = None

    def forward(self, key_fwd, value_fwd, query_fwd, mask, aw_prev=None,
                mode='', cache=True, trigger_point=None):
        """Forward computation.

        Args:
            key_fwd (FloatTensor): `[B, klen, kdim]`
            klens (IntTensor): `[B]`
            value_fwd (FloatTensor): `[B, klen, vdim]`
            query_fwd (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            aw_prev: dummy interface for single-head attention
            mode: dummy interface for MoChA
            cache (bool): cache key_fwd and mask
            trigger_point (IntTensor): dummy
        Returns:
            cv (FloatTensor): `[B, qlen, vdim]`
            aw (FloatTensor): `[B, n_heads, qlen, klen]`

        """
        bs, klen = key_fwd.size()[: 2]
        qlen = query_fwd.size(1)

        if self.key_fwd is None or not cache:
            key_fwd = self.w_key(key_fwd).view(bs, -1, self.n_heads, self.d_k)
            value_fwd = self.w_value(value_fwd).view(bs, -1, self.n_heads, self.d_k)
            self.key_fwd = key_fwd.transpose(2, 1).contiguous()      # `[B, n_heads, klen, d_k]`
            self.value_fwd = value_fwd.transpose(2, 1).contiguous()  # `[B, n_heads, klen, d_k]`
            self.mask = mask.unsqueeze(1).repeat(
                [1, self.n_heads, 1, 1]) if mask is not None else None  # `[B, n_heads, qlen, klen]`
            if self.mask is not None:
                assert self.mask.size() == (bs, self.n_heads, qlen, klen)

        query_fwd = self.w_query(query_fwd).view(bs, -1, self.n_heads, self.d_k)
        query_fwd = query_fwd.transpose(2, 1).contiguous()  # `[B, n_heads, qlen, d_k]`

        if self.atype == 'scaled_dot':
            e = torch.matmul(query_fwd, self.key_fwd.transpose(3, 2)) / math.sqrt(self.d_k)
        elif self.atype == 'add':
            e = torch.tanh(self.key_fwd.unsqueeze(2) + query_fwd.unsqueeze(3))
            e = e.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)
            e = self.v(e).permute(0, 3, 1, 2)

        # Compute attention weights
        if self.mask is not None:
            e = e.masked_fill_(self.mask == 0, NEG_INF)  # `[B, n_heads, qlen, klen]`
        aw = torch.softmax(e, dim=-1)
        aw = self.attn_dropout(aw)
        cv = torch.matmul(aw, self.value_fwd)  # `[B, n_heads, qlen, d_k]`
        cv = cv.transpose(2, 1).contiguous().view(bs, -1,  self.n_heads * self.d_k)
        cv = self.w_out(cv)

        return cv, aw
