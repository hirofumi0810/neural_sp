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

from neural_sp.models.modules.linear import LinearND


class MultiheadAttentionMechanism(nn.Module):
    """Multi-headed attention layer.

    Args:
        key_dim (int): dimensions of key
        query_dim (int): dimensions of query
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
                 attn_dim,
                 dropout=0,
                 n_heads=4):

        super(MultiheadAttentionMechanism, self).__init__()

        self.d_k = attn_dim // n_heads
        self.n_heads = n_heads
        self.key = None
        self.value = None
        self.mask = None

        # attention dropout applied AFTER the softmax layer
        self.attn_dropout = nn.Dropout(p=dropout)

        self.w_key = LinearND(key_dim, attn_dim, bias=False)
        self.w_value = LinearND(key_dim, attn_dim, bias=False)
        self.w_query = LinearND(query_dim, attn_dim, bias=False)
        self.w_out = LinearND(attn_dim, key_dim)

    def reset(self):
        self.key = None
        self.value = None
        self.mask = None

    def forward(self, key, key_lens, value, query, aw=None, diagonal=False):
        """Forward computation.

        Args:
            key (FloatTensor): `[B, key_len, key_dim]`
            key_lens (list): A list of length `[B]`
            value (FloatTensor): `[B, key_len, value_dim]`
            query (FloatTensor): `[B, query_len, query_dim]`
            aw (FloatTensor): dummy (not used)
            diagonal (bool): for Transformer decoder to hide future information
        Returns:
            cv (FloatTensor): `[B, query_len, value_dim]`
            aw (FloatTensor): `[B, key_len, n_heads]`

        """
        bs, key_len = key.size()[: 2]
        query_len = query.size(1)

        # Pre-computation of encoder-side features for computing scores
        if self.key is None:
            key = self.w_key(key).view(bs, key_len, self.n_heads, self.d_k)
            self.key = key.permute(0, 2, 3, 1).contiguous()
            value = self.w_value(value).view(bs, key_len, self.n_heads, self.d_k)
            self.value = value.permute(0, 2, 1, 3).contiguous()

        # Mask attention distribution
        if self.mask is None:
            self.mask = key.new_ones(bs, self.n_heads, query_len, key_len).byte()
            for b in range(bs):
                if key_lens[b] < key_len:
                    self.mask[b, :, :, key_lens[b]:] = 0

            # hide future information for transformer decoder
            if diagonal:
                assert query_len == key_len
                subsequent_mask = torch.tril(key.new_ones((query_len, key_len)).byte(), diagonal=0)
                subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1).expand(
                    bs, self.n_heads, -1, -1)  # `[B, n_heads, query_len, key_len]`
                self.mask = self.mask & subsequent_mask

        query = self.w_query(query).view(bs, query_len, self.n_heads, self.d_k)
        query = query.permute(0, 2, 1, 3).contiguous()  # `[B, n_heads, query_len, d_k]`
        e = torch.matmul(query, self.key) * (self.d_k ** -0.5)

        # Compute attention weights
        e = e.masked_fill_(self.mask == 0, -1024)  # `[B, n_heads, query_len, key_len]`
        aw = F.softmax(e, dim=-1)
        aw = self.attn_dropout(aw)
        cv = torch.matmul(aw, self.value)  # `[B, n_heads, query_len, d_k]`
        cv = cv.permute(0, 2, 3, 1).contiguous().view(bs, query_len, self.d_k * self.n_heads)
        cv = self.w_out(cv)

        aw = aw.permute(0, 2, 3, 1)[:, 0, :, :]
        # TODO(hiroufmi): fix for Transformer

        return cv, aw
