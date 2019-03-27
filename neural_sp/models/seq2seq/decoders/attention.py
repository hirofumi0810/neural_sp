# ! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Single-head attention layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.model_utils import LinearND


class AttentionMechanism(nn.Module):
    """Single-head attention layer.

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
                 dropout=0):

        super(AttentionMechanism, self).__init__()

        self.attn_type = attn_type
        self.attn_dim = attn_dim
        self.sharpening_factor = sharpening_factor
        self.sigmoid_smoothing = sigmoid_smoothing
        self.n_heads = 1
        self.key_a = None
        self.mask = None

        # attention dropout applied AFTER the softmax layer
        self.attn_dropout = nn.Dropout(p=dropout)

        if attn_type == 'add':
            self.w_enc = LinearND(key_dim, attn_dim)
            self.w_dec = LinearND(query_dim, attn_dim, bias=False)
            self.v = LinearND(attn_dim, 1, bias=False)

        elif attn_type == 'location':
            self.w_enc = LinearND(key_dim, attn_dim)
            self.w_dec = LinearND(query_dim, attn_dim, bias=False)
            self.w_conv = LinearND(conv_out_channels, attn_dim, bias=False)
            # self.conv = nn.Conv1d(in_channels=1,
            #                       out_channels=conv_out_channels,
            #                       kernel_size=conv_kernel_size * 2 + 1,
            #                       stride=1,
            #                       padding=conv_kernel_size,
            #                       bias=False)
            self.conv = nn.Conv2d(in_channels=1,
                                  out_channels=conv_out_channels,
                                  kernel_size=(1, conv_kernel_size * 2 + 1),
                                  stride=1,
                                  padding=(0, conv_kernel_size),
                                  bias=False)
            self.v = LinearND(attn_dim, 1, bias=False)

        elif attn_type == 'dot':
            self.w_enc = LinearND(key_dim, attn_dim, bias=False)
            self.w_dec = LinearND(query_dim, attn_dim, bias=False)

        elif attn_type == 'luong_dot':
            pass
            # NOTE: no additional parameters

        elif attn_type == 'luong_general':
            self.w_enc = LinearND(key_dim, query_dim, bias=False)

        elif attn_type == 'luong_concat':
            self.w = LinearND(key_dim + query_dim, attn_dim, bias=False)
            self.v = LinearND(attn_dim, 1, bias=False)

        else:
            raise ValueError(attn_type)

    def reset(self):
        self.key_a = None
        self.mask = None

    def forward(self, key, key_lens, value, query, aw=None):
        """Forward computation.

        Args:
            key (FloatTensor): `[B, T, key_dim]`
            key_lens (list): A list of length `[B]`
            value (FloatTensor): `[B, T, value_dim]`
            query (FloatTensor): `[B, 1, query_dim]`
            aw (FloatTensor): `[B, T]`
        Returns:
            cv (FloatTensor): `[B, 1, value_dim]`
            aw (FloatTensor): `[B, T]`

        """
        bs, key_len = key.size()[:2]

        if aw is None:
            aw = key.new_zeros(bs, key_len)

        # Pre-computation of encoder-side features for computing scores
        if self.key_a is None:
            if self.attn_type in ['add', 'location', 'dot', 'luong_general']:
                self.key_a = self.w_enc(key)

        # Mask attention distribution
        if self.mask is None:
            self.mask = key.new_ones(bs, key_len)
            for b in range(bs):
                if key_lens[b] < key_len:
                    self.mask[b, key_lens[b]:] = 0

        if self.attn_type == 'add':
            query = query.expand_as(torch.zeros((bs, key_len, query.size(2))))
            e = self.v(F.tanh(self.key_a + self.w_dec(query))).squeeze(2)

        elif self.attn_type == 'location':
            query = query.expand_as(torch.zeros((bs, key_len, query.size(2))))
            # For 1D conv
            # conv_feat = self.conv(aw[:, :].contiguous().unsqueeze(1))
            # For 2D conv
            conv_feat = self.conv(aw.view(bs, 1, 1, key_len)).squeeze(2)  # `[B, conv_out_channels, T]`
            conv_feat = conv_feat.transpose(2, 1).contiguous()  # `[B, T, conv_out_channels]`
            e = self.v(F.tanh(self.key_a + self.w_dec(query) + self.w_conv(conv_feat))).squeeze(2)

        elif self.attn_type == 'dot':
            e = torch.matmul(self.key_a, self.w_dec(query).transpose(-1, -2)).squeeze(2)

        elif self.attn_type == 'luong_dot':
            e = torch.matmul(key, query.transpose(-1, -2)).squeeze(2)

        elif self.attn_type == 'luong_general':
            e = torch.matmul(self.key_a, query.transpose(-1, -2)).squeeze(2)

        elif self.attn_type == 'luong_concat':
            query = query.expand_as(torch.zeros((bs, key_len, query.size(2))))
            e = self.v(F.tanh(self.w(torch.cat([key, query], dim=-1)))).squeeze(2)

        # Compute attention weights
        e = e.masked_fill_(self.mask == 0, -float('inf'))  # `[B, T]`
        if self.sigmoid_smoothing:
            aw = F.sigmoid(e) / F.sigmoid(e).sum(-1).unsqueeze(-1)
        else:
            aw = F.softmax(e * self.sharpening_factor, dim=-1)  # `[B, T]`

        # attention dropout
        aw = self.attn_dropout(aw)

        # Compute context vector (weighted sum of encoder outputs)
        cv = torch.matmul(aw.unsqueeze(1), value)

        return cv, aw
