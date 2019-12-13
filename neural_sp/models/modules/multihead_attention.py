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

NEG_INF = float(np.finfo(np.float32).min)


class MultiheadAttentionMechanism(nn.Module):
    """Multi-headed attention layer.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        atype (str): type of attention mechanisms
        adim: (int) dimension of the attention space
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
                 kdim,
                 qdim,
                 adim,
                 atype,
                 dropout=0.,
                 n_heads=4,
                 bias=True):

        super(MultiheadAttentionMechanism, self).__init__()

        self.atype = atype
        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.key = None
        self.value = None
        self.mask = None

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

    def reset(self):
        self.key = None
        self.value = None
        self.mask = None

    def forward(self, key, value, query, mask, aw_prev=None, mode=''):
        """Forward computation.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            klens (IntTensor): `[B]`
            value (FloatTensor): `[B, klen, vdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, klen, qlen]`
            aw_prev: dummy
            mode: dummy
        Returns:
            cv (FloatTensor): `[B, qlen, vdim]`
            aw (FloatTensor): `[B, n_heads, qlen, klen]`

        """
        bs, klen = key.size()[: 2]
        qlen = query.size(1)

        if self.key is None:
            key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)
            value = self.w_value(value).view(bs, -1, self.n_heads, self.d_k)
            self.key = key.transpose(2, 1).contiguous()      # `[B, n_heads, klen, d_k]`
            self.value = value.transpose(2, 1).contiguous()  # `[B, n_heads, klen, d_k]`
            self.mask = mask.unsqueeze(1) if mask is not None else None  # `[B, 1, klen, qlen]`

        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)
        query = query.transpose(2, 1).contiguous()  # `[B, n_heads, qlen, d_k]`

        if self.atype == 'scaled_dot':
            e = torch.matmul(query, self.key.transpose(3, 2)) / math.sqrt(self.d_k)
        elif self.atype == 'add':
            e = torch.tanh(self.key.unsqueeze(2) + query.unsqueeze(3))
            e = e.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)
            e = self.v(e).permute(0, 3, 1, 2)

        # Compute attention weights
        if self.mask is not None:
            e = e.masked_fill_(self.mask == 0, NEG_INF)  # `[B, n_heads, qlen, klen]`
            # e = e.transpose(3, 2).masked_fill_(self.mask.transpose(3, 2) == 0, NEG_INF).transpose(3, 2)
        aw = torch.softmax(e, dim=-1)
        aw = self.attn_dropout(aw)
        cv = torch.matmul(aw, self.value)  # `[B, n_heads, qlen, d_k]`
        cv = cv.transpose(2, 1).contiguous().view(bs, -1,  self.n_heads * self.d_k)
        cv = self.w_out(cv)

        return cv, aw
