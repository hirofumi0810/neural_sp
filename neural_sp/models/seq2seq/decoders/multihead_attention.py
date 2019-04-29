# ! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-head attention layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.model_utils import LinearND


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
        dropout (float):
        n_heads (int): number of heads in the multi-head attention

    """

    def __init__(self,
                 key_dim,
                 query_dim,
                 attn_type,
                 attn_dim,
                 dropout=0,
                 n_heads=4,
                 scale=1):

        super(MultiheadAttentionMechanism, self).__init__()

        self.attn_type = attn_type
        self.attn_dim = attn_dim
        self.n_heads = n_heads
        # self.scale = attn_dim ** -0.5
        self.scale = scale
        self.key = None
        self.value = None
        self.mask = None

        # attention dropout applied AFTER the softmax layer
        self.attn_dropout = nn.Dropout(p=dropout)

        assert attn_type == 'dot'

        self.w_key = LinearND(key_dim, attn_dim * n_heads, bias=False)
        self.w_value = LinearND(key_dim, attn_dim * n_heads, bias=False)
        self.w_query = LinearND(query_dim, attn_dim * n_heads, bias=False)
        self.w_out = LinearND(attn_dim * n_heads, key_dim)

    def reset(self):
        self.key = None
        self.value = None
        self.mask = None

    def forward(self, key, key_lens, value, query, aw=None):
        """Forward computation.

        Args:
            key (FloatTensor): `[B, key_len, key_dim]`
            key_lens (list): A list of length `[B]`
            value (FloatTensor): `[B, key_len, value_dim]`
            query (FloatTensor): `[B, 1, query_dim]`
            aw (FloatTensor): not used
        Returns:
            cv (FloatTensor): `[B, 1, value_dim]`
            aw (FloatTensor): `[B, key_len, n_heads]`

        """
        bs, key_len = key.size()[:2]

        # Pre-computation of encoder-side features for computing scores
        if self.key is None:
            key = self.w_key(key).view(bs, key_len, self.n_heads, self.attn_dim)
            self.key = key.permute(2, 0, 3, 1).contiguous().view(self.n_heads * bs, self.attn_dim, key_len)
            # `[n_heads * B, key_len, attn_dim]`

            value = self.w_value(value).view(bs, key_len, self.n_heads, self.attn_dim)
            self.value = value.permute(2, 0, 1, 3).contiguous().view(self.n_heads * bs, key_len, self.attn_dim)
            # `[n_heads * B, key_len, attn_dim]`

        # Mask attention distribution
        if self.mask is None:
            mask = key.new_ones(bs, 1, key_len)
            for b in range(bs):
                if key_lens[b] < key_len:
                    mask[b, :, key_lens[b]:] = 0
            self.mask = mask.repeat(self.n_heads, 1, 1)

        query = self.w_query(query).view(bs, 1, self.n_heads, self.attn_dim)
        query = query.transpose(2, 1).contiguous().view(self.n_heads * bs, 1, self.attn_dim)
        e = torch.bmm(query, self.key) * self.scale

        # Compute attention weights
        e = e.masked_fill_(self.mask == 0, -1024)  # `[n_heads * B, 1, key_len]`
        aw = F.softmax(e, dim=1)

        # attention dropout
        aw = self.attn_dropout(aw)

        # Compute context vector (weighted sum of encoder outputs)
        cv = torch.bmm(aw, self.value)  # `[n_heads * B, 1, attn_dim]`
        cv = cv.view(self.n_heads, bs, 1, self.attn_dim).permute(
            1, 2, 0, 3).contiguous().view(bs, 1, self.n_heads * self.attn_dim)
        cv = self.w_out(cv)

        return cv, aw


class TransformerMultiheadAttentionMechanism(nn.Module):
    def __init__(self, n_heads, d_model, dropout):
        """Multi-headed attention layer impelemented in Transformer.

        Args:
            n_heads (int):
            d_model (int):
            dropout (float):

        """
        super(TransformerMultiheadAttentionMechanism, self).__init__()

        assert d_model % n_heads == 0
        # We assume d_v always equals d_k
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.scale = self.d_k ** -0.5

        self.w_key = LinearND(d_model, d_model, bias=False)
        self.w_value = LinearND(d_model, d_model, bias=False)
        self.w_query = LinearND(d_model, d_model, bias=False)
        self.w_out = LinearND(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)  # for probabilities

    def reset(self):
        self.mask = None

    def forward(self, key, value, query, mask):
        """Forward computation.

        Args:
            key (FloatTensor): `[B, key_len, d_model]`
            value (FloatTensor): `[B, key_len, d_model]`
            query (FloatTensor): `[B, query_len, d_model]`
            mask (): `[B, query_len, key_len]`
                0: place to pad with -1024
                1: otherwise
        Returns:
            cv (FloatTensor): `[B, query_len, key_len]`
            aw (FloatTensor): `[B, n_heads, query_len, key_len]`

        """
        bs = query.size(0)

        # 1) Do all the linear projections in batch from d_model => head x d_k
        key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k).transpose(2, 1).transpose(3, 2)
        value = self.w_value(value).view(bs, -1, self.n_heads, self.d_k).transpose(2, 1)
        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k).transpose(2, 1)

        # 2) Apply attention on all the projected vectors in batch.
        e = torch.bmm(query, key) * self.scale
        if mask is not None:
            mask = mask.unsqueeze(1)  # Same mask applied to all heads.
            e = e.masked_fill(mask == 0, -10e9)  # this is ok
            # e = e.masked_fill(mask == 0, -1024)
        aw = self.dropout(F.softmax(e, dim=-1))
        cv = torch.bmm(aw, value)

        # 3) "Concat" using a view and apply a final linear.
        cv = cv.transpose(2, 1).contiguous().view(bs, -1, self.n_heads * self.d_k)
        cv = self.w_out(cv)

        return cv, aw
