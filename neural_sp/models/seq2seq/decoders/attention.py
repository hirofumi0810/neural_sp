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

from neural_sp.models.modules.linear import LinearND
from neural_sp.models.torch_utils import make_pad_mask


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
        self.key = None
        self.mask = None

        # attention dropout applied AFTER the softmax layer
        self.attn_dropout = nn.Dropout(p=dropout)

        if attn_type == 'no':
            pass
            # NOTE: sequence-to-sequence without attetnion (use the last state as a context vector)

        elif attn_type == 'add':
            self.w_key = LinearND(key_dim, attn_dim, bias=True)
            self.w_query = LinearND(query_dim, attn_dim, bias=False)
            self.v = LinearND(attn_dim, 1, bias=False)

        elif attn_type == 'location':
            self.w_key = LinearND(key_dim, attn_dim, bias=True)
            self.w_query = LinearND(query_dim, attn_dim, bias=False)
            self.w_conv = LinearND(conv_out_channels, attn_dim, bias=False)
            self.conv = nn.Conv2d(in_channels=1,
                                  out_channels=conv_out_channels,
                                  kernel_size=(1, conv_kernel_size * 2 + 1),
                                  stride=1,
                                  padding=(0, conv_kernel_size),
                                  bias=False)
            self.v = LinearND(attn_dim, 1, bias=False)

        elif attn_type == 'dot':
            self.w_key = LinearND(key_dim, attn_dim, bias=False)
            self.w_query = LinearND(query_dim, attn_dim, bias=False)

        elif attn_type == 'luong_dot':
            pass
            # NOTE: no additional parameters

        elif attn_type == 'luong_general':
            self.w_key = LinearND(key_dim, query_dim, bias=False)

        elif attn_type == 'luong_concat':
            self.w = LinearND(key_dim + query_dim, attn_dim, bias=False)
            self.v = LinearND(attn_dim, 1, bias=False)

        else:
            raise ValueError(attn_type)

    def reset(self):
        self.key = None
        self.mask = None

    def forward(self, key, klens, value, query, aw=None):
        """Forward computation.

        Args:
            key (FloatTensor): `[B, klen, key_dim]`
            klens (list): A list of length `[B]`
            value (FloatTensor): `[B, klen, value_dim]`
            query (FloatTensor): `[B, 1, query_dim]`
            aw (FloatTensor): `[B, klen, 1 (n_heads)]`
        Returns:
            cv (FloatTensor): `[B, 1, value_dim]`
            aw (FloatTensor): `[B, klen, 1 (n_heads)]`

        """
        bs, klen = key.size()[:2]

        if aw is None:
            aw = key.new_zeros(bs, klen, 1)

        # Mask attention distribution
        if self.mask is None:
            device_id = torch.cuda.device_of(key.data).idx
            self.mask = make_pad_mask(klens, device_id).expand(bs, klen)  # `[B, qlen, klen]`

        # Pre-computation of encoder-side features for computing scores
        if self.key is None:
            if self.attn_type in ['add', 'location', 'dot', 'luong_general']:
                self.key = self.w_key(key)
            else:
                self.key = key

        if self.attn_type == 'add':
            query = query.expand_as(torch.zeros((bs, klen, query.size(2))))
            e = self.v(torch.tanh(self.key + self.w_query(query))).squeeze(2)

        elif self.attn_type == 'location':
            query = query.expand_as(torch.zeros((bs, klen, query.size(2))))
            conv_feat = self.conv(aw.unsqueeze(3).transpose(3, 1)).squeeze(2)  # `[B, conv_out_channels, klen]`
            conv_feat = conv_feat.transpose(2, 1).contiguous()  # `[B, klen, conv_out_channels]`
            e = self.v(torch.tanh(self.key + self.w_query(query) + self.w_conv(conv_feat))).squeeze(2)

        elif self.attn_type == 'dot':
            e = torch.bmm(self.key, self.w_query(query).transpose(-2, -1)).squeeze(2)

        elif self.attn_type == 'luong_dot':
            e = torch.bmm(self.key, query.transpose(-2, -1)).squeeze(2)

        elif self.attn_type == 'luong_general':
            e = torch.bmm(self.key, query.transpose(-2, -1)).squeeze(2)

        elif self.attn_type == 'luong_concat':
            query = query.expand_as(torch.zeros((bs, klen, query.size(2))))
            e = self.v(torch.tanh(self.w(torch.cat([self.key, query], dim=-1)))).squeeze(2)

        if self.attn_type == 'no':
            last_state = [key[b, klens[b] - 1] for b in range(bs)]
            cv = torch.stack(last_state, dim=0).unsqueeze(1)
        else:
            # Compute attention weights, context vector
            e = e.masked_fill_(self.mask == 0, -1024)  # `[B, klen]`
            if self.sigmoid_smoothing:
                aw = F.sigmoid(e) / F.sigmoid(e).sum(-1).unsqueeze(-1)
            else:
                aw = F.softmax(e * self.sharpening_factor, dim=-1)
            aw = self.attn_dropout(aw)
            cv = torch.bmm(aw.unsqueeze(1), value)

        return cv, aw.unsqueeze(2)
