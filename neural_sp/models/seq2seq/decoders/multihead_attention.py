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
                 sharpening_factor=1,
                 sigmoid_smoothing=False,
                 conv_out_channels=10,
                 conv_kernel_size=100,
                 dropout=0,
                 n_heads=4):

        super(MultiheadAttentionMechanism, self).__init__()

        self.attn_type = attn_type
        self.attn_dim = attn_dim
        self.sharpening_factor = sharpening_factor
        self.sigmoid_smoothing = sigmoid_smoothing
        self.n_heads = n_heads
        self.key_a = None
        self.mask = None

        # attention dropout applied AFTER the softmax layer
        if dropout > 0:
            self.dropout = nn.ModuleList([nn.Dropout(p=dropout)] * n_heads)
        else:
            self.dropout = None

        if attn_type == 'add':
            self.w_enc = nn.ModuleList([LinearND(key_dim, attn_dim)] * n_heads)
            self.w_dec = nn.ModuleList([LinearND(query_dim, attn_dim, bias=False)] * n_heads)
            self.v = nn.ModuleList([LinearND(attn_dim, 1, bias=False)] * n_heads)

        elif attn_type == 'location':
            self.w_enc = nn.ModuleList([LinearND(key_dim, attn_dim)] * n_heads)
            self.w_dec = nn.ModuleList([LinearND(query_dim, attn_dim, bias=False)] * n_heads)
            self.w_conv = nn.ModuleList([LinearND(conv_out_channels, attn_dim, bias=False)] * n_heads)
            # self.conv = nn.ModuleList([nn.Conv1d(in_channels=1,
            #                                       out_channels=conv_out_channels,
            #                                       kernel_size=conv_kernel_size * 2 + 1,
            #                                       stride=1,
            #                                       padding=conv_kernel_size,
            #                                       bias=False) for _ in range(n_heads)] * n_heads)
            self.conv = nn.ModuleList([nn.Conv2d(in_channels=1,
                                                 out_channels=conv_out_channels,
                                                 kernel_size=(1, conv_kernel_size * 2 + 1),
                                                 stride=1,
                                                 padding=(0, conv_kernel_size),
                                                 bias=False) for _ in range(n_heads)] * n_heads)
            self.v = nn.ModuleList([LinearND(attn_dim, 1, bias=False)] * n_heads)

        elif attn_type == 'dot':
            self.w_enc = nn.ModuleList([LinearND(key_dim, attn_dim, bias=False)] * n_heads)
            self.w_dec = nn.ModuleList([LinearND(query_dim, attn_dim, bias=False)] * n_heads)

        elif attn_type == 'luong_dot':
            pass
            # NOTE: no additional parameters

        elif attn_type == 'luong_general':
            self.w_enc = nn.ModuleList([LinearND(key_dim, query_dim, bias=False)] * n_heads)

        elif attn_type == 'luong_concat':
            self.w = nn.ModuleList([LinearND(key_dim + query_dim, attn_dim, bias=False)] * n_heads)
            self.v = nn.ModuleList([LinearND(attn_dim, 1, bias=False)] * n_heads)

        else:
            raise ValueError(attn_type)

        self.w_out = LinearND(key_dim * n_heads, key_dim)

    def reset(self):
        self.key_a = None
        self.mask = None

    def forward(self, key, key_lens, value, query, aw):
        """Forward computation.

        Args:
            key (FloatTensor): `[B, T, key_dim]`
            key_lens (list): A list of length `[B]`
            value (FloatTensor): `[B, T, value_dim]`
            query (FloatTensor): `[B, 1, query_dim]`
            aw (FloatTensor): `[B, T, n_heads]`
        Returns:
            context (FloatTensor): `[B, 1, value_dim]`
            aw (FloatTensor): `[n_heads, B, T]`

        """
        bs, key_len = key.size()[:2]

        if aw is None:
            aw = key.new_ones(self.n_heads, bs, key_len)

        # Pre-computation of encoder-side features for computing scores
        if self.key_a is None:
            if self.attn_type in ['add', 'location', 'dot', 'luong_general']:
                self.key_a = [self.w_enc[h](key) for h in range(self.n_heads)]

        # Mask attention distribution
        if self.mask is None:
            self.mask = key.new_ones(bs, key_len)
            for b in range(bs):
                if key_lens[b] < key_len:
                    self.mask[b, key_lens[b]:] = 0
            # TODO(hirofumi): prepare mask per attention

        # Compute per head
        cvs = []
        aws = []
        for h in range(self.n_heads):
            if self.attn_type == 'add':
                query_h = query.expand_as(torch.zeros((bs, key_len, query.size(2))))
                e_h = self.v[h](F.tanh(self.key_a[h] + self.w_dec[h](query_h))).squeeze(2)

            elif self.attn_type == 'location':
                query_h = query.expand_as(torch.zeros((bs, key_len, query.size(2))))
                # For 1D conv
                # conv_feat = self.conv[h](aw[h][:, :].contiguous().unsqueeze(1))
                # For 2D conv
                conv_feat_h = self.conv[h](aw[h].view(bs, 1, 1, key_len)
                                           ).squeeze(2)  # `[B, conv_out_channels, T]`
                conv_feat_h = conv_feat_h.transpose(2, 1).contiguous()  # `[B, T, conv_out_channels]`
                e_h = self.v[h](F.tanh(self.key_a[h] + self.w_dec[h]
                                       (query_h) + self.w_conv[h](conv_feat_h))).squeeze(2)

            elif self.attn_type == 'dot':
                e_h = torch.matmul(self.key_a[h], self.w_dec[h](query).transpose(-1, -2)).squeeze(2)

            elif self.attn_type == 'luong_dot':
                e_h = torch.matmul(key, query.transpose(-1, -2)).squeeze(2)

            elif self.attn_type == 'luong_general':
                e_h = torch.matmul(self.key_a[h], query.transpose(-1, -2)).squeeze(2)

            elif self.attn_type == 'luong_concat':
                query_h = query.expand_as(torch.zeros((bs, key_len, query.size(2))))
                e_h = self.v[h](F.tanh(self.w[h](torch.cat([key, query_h], dim=-1)))).squeeze(2)

            # Compute attention weights
            e_h = e_h.masked_fill_(self.mask == 0, -float('inf'))  # `[B, T]`
            if self.sigmoid_smoothing:
                aw_h = F.sigmoid(e_h) / F.sigmoid(e_h).sum(-1).unsqueeze(-1)
            else:
                aw_h = F.softmax(e_h * self.sharpening_factor, dim=-1)  # `[B, T]`
            # attention dropout
            if self.dropout is not None:
                aw_h = self.dropout[h](aw_h)
            aws.append(aw_h)

            # Compute context vector (weighted sum of encoder outputs)
            cvs.append(torch.matmul(aw[h].unsqueeze(1), value))

        return self.w_out(torch.cat(cvs, dim=-1)), torch.stack(aws, dim=0)


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
            key (FloatTensor): `[B, key_time, d_model]`
            value (FloatTensor): `[B, key_time, d_model]`
            query (FloatTensor): `[B, query_time, d_model]`
            mask (): `[B, query_time, key_time]`
                0: place to pad with -1024
                1: otherwise
        Returns:
            cv (FloatTensor): `[B, query_time, key_time]`
            aw (FloatTensor): `[B, n_heads, query_time, key_time]`

        """
        bs = query.size(0)

        # 1) Do all the linear projections in batch from d_model => head x d_k
        key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k).transpose(2, 1)
        value = self.w_value(value).view(bs, -1, self.n_heads, self.d_k).transpose(2, 1)
        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k).transpose(2, 1)

        # 2) Apply attention on all the projected vectors in batch.
        e = torch.matmul(query, key.transpose(3, 2)) * (self.d_k ** -0.5)
        if mask is not None:
            mask = mask.unsqueeze(1)  # Same mask applied to all heads.
            # e = e.masked_fill(mask == 0, -float('inf'))  # this is buggy
            e = e.masked_fill(mask == 0, -10e9)  # this is ok
            # e = e.masked_fill(mask == 0, -1024)
        aw = self.dropout(F.softmax(e, dim=-1))
        cv = torch.matmul(aw, value)

        # 3) "Concat" using a view and apply a final linear.
        cv = cv.transpose(2, 1).contiguous().view(bs, -1, self.n_heads * self.d_k)
        cv = self.w_out(cv)

        return cv, aw
