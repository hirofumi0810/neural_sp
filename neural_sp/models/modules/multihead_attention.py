# ! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-head attention layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.modules.linear import Linear

NEG_INF = float(np.finfo(np.float32).min)


class MultiheadAttentionMechanism(nn.Module):
    """Multi-headed attention layer.

    Args:
        key_dim (int): dimensions of key
        query_dim (int): dimensions of query
        attn_type (str): type of attention mechanisms
        attn_dim: (int) dimension of the attention layer
        sharpening_factor (float): sharpening factor in the softmax layer
            for attention weights
        sigmoid_smoothing (bool): replace the softmax layer for attention weights
            with the sigmoid function
        conv_out_channels (int): number of channles of conv outputs.
            This is used for location-based attention.
        conv_kernel_size (int): size of kernel.
            This must be the odd number.
        dropout (float): dropout probability
        n_heads (int): number of heads in the multi-head attention

    """

    def __init__(self,
                 key_dim,
                 query_dim,
                 attn_type,
                 attn_dim,
                 dropout=0,
                 n_heads=4):

        super(MultiheadAttentionMechanism, self).__init__()

        self.attn_type = attn_type
        assert attn_dim % n_heads == 0
        self.d_k = attn_dim // n_heads
        self.n_heads = n_heads
        self.key = None
        self.value = None
        self.mask = None

        # attention dropout applied AFTER the softmax layer
        self.attn_dropout = nn.Dropout(p=dropout)

        if attn_type == 'scaled_dot':
            self.w_key = Linear(key_dim, attn_dim, bias=False)
            self.w_value = Linear(key_dim, attn_dim, bias=False)
            self.w_query = Linear(query_dim, attn_dim, bias=False)
        elif attn_type == 'add':
            self.w_key = Linear(key_dim, attn_dim, bias=True)
            self.w_value = Linear(key_dim, attn_dim, bias=False)
            self.w_query = Linear(query_dim, attn_dim, bias=False)
            self.v = Linear(attn_dim, n_heads, bias=False)
        else:
            raise NotImplementedError(attn_type)

        self.w_out = Linear(attn_dim, key_dim)

    def reset(self):
        self.key = None
        self.value = None
        self.mask = None

    def forward(self, key, value, query, mask, aw_prev=None, mode=''):
        """Forward computation.

        Args:
            key (FloatTensor): `[B, klen, key_dim]`
            klens (IntTensor): `[B]`
            value (FloatTensor): `[B, klen, value_dim]`
            query (FloatTensor): `[B, qlen, query_dim]`
            mask (ByteTensor): `[B, n_heads, klen, qlen]`
            aw_prev: dummy
            mode: dummy
        Returns:
            cv (FloatTensor): `[B, qlen, value_dim]`
            aw (FloatTensor): `[B, n_heads, qlen, klen]`

        """
        bs, klen = key.size()[: 2]
        qlen = query.size(1)

        if self.key is None:
            key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)
            value = self.w_value(value).view(bs, -1, self.n_heads, self.d_k)
            self.key = key.transpose(2, 1).contiguous()      # `[B, n_heads, klen, d_k]`
            self.value = value.transpose(2, 1).contiguous()  # `[B, n_heads, klen, d_k]`
            self.mask = mask

        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)
        query = query.transpose(2, 1).contiguous()  # `[B, n_heads, qlen, d_k]`

        if self.attn_type == 'scaled_dot':
            e = torch.matmul(query, self.key.transpose(3, 2)) / math.sqrt(self.d_k)
        elif self.attn_type == 'add':
            e = torch.tanh(self.key.unsqueeze(2) + query.unsqueeze(3))
            e = e.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)
            e = self.v(e).permute(0, 3, 1, 2)

        # Compute attention weights
        if self.mask is not None:
            e = e.masked_fill_(self.mask == 0, NEG_INF)  # `[B, n_heads, qlen, klen]`
            # e = e.transpose(3, 2).masked_fill_(self.mask.transpose(3, 2) == 0, NEG_INF).transpose(3, 2)
        aw = F.softmax(e, dim=-1)
        aw = self.attn_dropout(aw)
        cv = torch.matmul(aw, self.value)  # `[B, n_heads, qlen, d_k]`
        cv = cv.transpose(2, 1).contiguous().view(bs, -1,  self.n_heads * self.d_k)
        cv = self.w_out(cv)

        return cv, aw
