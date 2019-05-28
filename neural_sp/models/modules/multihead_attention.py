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
from neural_sp.models.torch_utils import make_pad_mask


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
                 n_heads=8):

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
            self.w_key = LinearND(key_dim, attn_dim, bias=False)
            self.w_value = LinearND(key_dim, attn_dim, bias=False)
            self.w_query = LinearND(query_dim, attn_dim, bias=False)
        elif attn_type == 'add':
            self.w_key = LinearND(key_dim, attn_dim, bias=True)
            self.w_value = LinearND(key_dim, attn_dim, bias=False)
            self.w_query = LinearND(query_dim, attn_dim, bias=False)
            self.v = LinearND(attn_dim, n_heads, bias=False)
        else:
            raise NotImplementedError(attn_type)

        self.w_out = LinearND(attn_dim, key_dim)

    def reset(self):
        self.key = None
        self.value = None
        self.mask = None

    def forward(self, key, klens, value, query, qlens=None, aw=None, diagonal=False):
        """Forward computation.

        Args:
            key (FloatTensor): `[B, klen, key_dim]`
            klens (list): A list of length `[B]`
            value (FloatTensor): `[B, klen, value_dim]`
            query (FloatTensor): `[B, qlen, query_dim]`
            qlens (list): A list of length `[B]`
            aw (FloatTensor): dummy (not used)
            diagonal (bool): for Transformer decoder to hide future information
        Returns:
            cv (FloatTensor): `[B, qlen, value_dim]`
            aw (FloatTensor): `[B, n_heads, qlen, klen]`

        """
        bs, klen = key.size()[: 2]
        qlen = query.size(1)

        # Mask attention distribution
        # if self.mask is None:
        device_id = torch.cuda.device_of(key.data).idx
        mask = make_pad_mask(klens, device_id).unsqueeze(1).expand(bs, qlen, klen)  # `[B, qlen, klen]`
        if qlens is not None:
            query_mask = make_pad_mask(qlens, device_id).unsqueeze(2).expand(bs, qlen, klen)  # `[B, qlen, klen]`
        elif klen == qlen:
            assert klen == qlen
            query_mask = make_pad_mask(klens, device_id).unsqueeze(2).expand(bs, qlen, klen)  # `[B, qlen, klen]`
        self.mask = (mask * query_mask).unsqueeze(1)  # `[B, 1, qlen, klen]`

        # Hide future information for self-attention in the Transformer decoder
        if diagonal:
            assert qlen == klen
            subsequent_mask = torch.tril(key.new_ones((qlen, klen)).byte(), diagonal=0)
            subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1).expand(
                bs, self.n_heads, qlen, klen)  # `[B, n_heads, qlen, klen]`
            self.mask = self.mask & subsequent_mask

        # if self.key is None:
        key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)
        value = self.w_value(value).view(bs, -1, self.n_heads, self.d_k)
        self.key = key.transpose(2, 1).contiguous()      # `[B, n_heads, klen, d_k]`
        self.value = value.transpose(2, 1).contiguous()  # `[B, n_heads, klen, d_k]`
        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)
        query = query.transpose(2, 1).contiguous()  # `[B, n_heads, qlen, d_k]`

        if self.attn_type == 'scaled_dot':
            e = torch.matmul(query, self.key.transpose(3, 2)) * (self.d_k ** -0.5)
        elif self.attn_type == 'add':
            e = torch.tanh(self.key.unsqueeze(2) + query.unsqueeze(3))
            e = e.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)
            e = self.v(e).permute(0, 3, 1, 2)

        # Compute attention weights
        e = e.masked_fill_(self.mask == 0, -1e9)  # `[B, n_heads, qlen, klen]`
        aw = F.softmax(e, dim=-1)
        aw = self.attn_dropout(aw)
        cv = torch.matmul(aw, self.value)  # `[B, n_heads, qlen, d_k]`
        cv = cv.transpose(2, 1).contiguous().view(bs, -1,  self.n_heads * self.d_k)
        cv = self.w_out(cv)

        return cv, aw
