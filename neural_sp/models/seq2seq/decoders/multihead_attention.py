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

        self.head_dim = attn_dim // n_heads
        self.n_heads = n_heads
        self.scale = self.head_dim ** -0.5
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
            diagonal (bool):
        Returns:
            cv (FloatTensor): `[B, query_len, value_dim]`
            aw (FloatTensor): `[B, key_len, n_heads]`

        """
        bs, key_len = key.size()[: 2]
        query_len = query.size(1)

        # Pre-computation of encoder-side features for computing scores
        if self.key is None:
            key = self.w_key(key).view(bs, key_len, self.n_heads, self.head_dim)
            self.key = key.permute(0, 2, 3, 1).contiguous().view(bs * self.n_heads, self.head_dim, key_len)
            # `[B * n_heads, key_len, head_dim]`

            value = self.w_value(value).view(bs, key_len, self.n_heads, self.head_dim)
            self.value = value.permute(0, 2, 1, 3).contiguous().view(bs * self.n_heads, key_len, self.head_dim)
            # `[B * n_heads, key_len, head_dim]`

        # Mask attention distribution
        if self.mask is None:
            mask = key.new_ones(bs, query_len, key_len).byte()
            for b in range(bs):
                if key_lens[b] < key_len:
                    mask[b, :, key_lens[b]:] = 0
            self.mask = mask.repeat(self.n_heads, 1, 1)

            # hide future information for transformer decoder
            if diagonal:
                assert query_len == key_len
                subsequent_mask = torch.tril(key.new_ones((query_len, key_len)).byte(), diagonal=0)
                subsequent_mask = subsequent_mask.unsqueeze(0).expand(
                    bs * self.n_heads, -1, -1)  # `[B, query_len, key_len]`
                self.mask = self.mask & subsequent_mask

        query = self.w_query(query).view(bs, query_len, self.n_heads, self.head_dim)
        query = query.transpose(2, 1).contiguous().view(bs * self.n_heads, query_len, self.head_dim)
        e = torch.bmm(query, self.key) * self.scale

        # Compute attention weights
        e = e.masked_fill_(self.mask == 0, -1024)  # `[B * n_heads, query_len, key_len]`
        aw = F.softmax(e, dim=-1)

        # attention dropout
        aw = self.attn_dropout(aw)

        # Compute context vector (weighted sum of encoder outputs)
        cv = torch.bmm(aw, self.value)  # `[B * n_heads, query_len, head_dim]`
        cv = cv.view(bs, self.n_heads, query_len, self.head_dim).permute(
            0, 2, 1, 3).contiguous().view(bs, query_len, self.n_heads * self.head_dim)
        cv = self.w_out(cv)

        aw = aw.view(bs, self.n_heads, query_len, key_len)
        aw = aw.permute(0, 2, 3, 1)[:, 0, :, :]
        # TODO(hiroufmi): fix for Transformer

        return cv, aw
