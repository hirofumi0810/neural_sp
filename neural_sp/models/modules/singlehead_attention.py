# ! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Single-head attention layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

NEG_INF = float(np.finfo(np.float32).min)


class AttentionMechanism(nn.Module):
    """Single-head attention layer.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        atype (str): type of attention mechanisms
        adim: (int) dimension of the attention layer
        sharpening_factor (float): sharpening factor in the softmax layer
            for attention weights
        sigmoid_smoothing (bool): replace the softmax layer for attention weights
            with the sigmoid function
        conv_out_channels (int): number of channles of conv outputs.
            This is used for location-based attention.
        conv_kernel_size (int): size of kernel.
            This must be the odd number.
        dropout (float): attention dropout probability

    """

    def __init__(self,
                 kdim,
                 qdim,
                 adim,
                 atype,
                 sharpening_factor=1,
                 sigmoid_smoothing=False,
                 conv_out_channels=10,
                 conv_kernel_size=201,
                 dropout=0.):

        super(AttentionMechanism, self).__init__()

        assert conv_kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        self.atype = atype
        self.adim = adim
        self.sharpening_factor = sharpening_factor
        self.sigmoid_smoothing = sigmoid_smoothing
        self.n_heads = 1
        self.key = None
        self.mask = None

        # attention dropout applied after the softmax layer
        self.attn_dropout = nn.Dropout(p=dropout)

        if atype == 'no':
            raise NotImplementedError
            # NOTE: sequence-to-sequence without attetnion (use the last state as a context vector)

        elif atype == 'add':
            self.w_key = nn.Linear(kdim, adim, bias=True)
            self.w_query = nn.Linear(qdim, adim, bias=False)
            self.v = nn.Linear(adim, 1, bias=False)

        elif atype == 'location':
            self.w_key = nn.Linear(kdim, adim, bias=True)
            self.w_query = nn.Linear(qdim, adim, bias=False)
            self.w_conv = nn.Linear(conv_out_channels, adim, bias=False)
            self.conv = nn.Conv2d(in_channels=1,
                                  out_channels=conv_out_channels,
                                  kernel_size=(1, conv_kernel_size),
                                  stride=1,
                                  padding=(0, (conv_kernel_size - 1) // 2),
                                  bias=False)
            self.v = nn.Linear(adim, 1, bias=False)

        elif atype == 'dot':
            self.w_key = nn.Linear(kdim, adim, bias=False)
            self.w_query = nn.Linear(qdim, adim, bias=False)

        elif atype == 'luong_dot':
            pass
            # NOTE: no additional parameters

        elif atype == 'luong_general':
            self.w_key = nn.Linear(kdim, qdim, bias=False)

        elif atype == 'luong_concat':
            self.w = nn.Linear(kdim + qdim, adim, bias=False)
            self.v = nn.Linear(adim, 1, bias=False)

        else:
            raise ValueError(atype)

    def reset(self):
        self.key = None
        self.mask = None

    def forward(self, key, value, query, mask=None, aw_prev=None, mode=''):
        """Forward computation.

        Args:
            key (FloatTensor): `[B, kmax, kdim]`
            klens (IntTensor): `[B]`
            value (FloatTensor): `[B, kmax, vdim]`
            query (FloatTensor): `[B, 1, qdim]`
            mask (ByteTensor): `[B, qmax, kmax]`
            aw_prev (FloatTensor): `[B, kmax, 1 (n_heads)]`
            mode: dummy
        Returns:
            cv (FloatTensor): `[B, 1, vdim]`
            aw (FloatTensor): `[B, kmax, 1 (n_heads)]`

        """
        bs, kmax = key.size()[:2]

        if aw_prev is None:
            aw_prev = key.new_zeros(bs, kmax, 1)

        # Pre-computation of encoder-side features for computing scores
        if self.key is None:
            if self.atype in ['add', 'location', 'dot', 'luong_general']:
                self.key = self.w_key(key)
            else:
                self.key = key
            self.mask = mask

        if self.atype == 'no':
            raise NotImplementedError
            # last_state = [key[b, klens[b] - 1] for b in range(bs)]
            # cv = torch.stack(last_state, dim=0).unsqueeze(1)
            # return cv, None

        elif self.atype == 'add':
            query = query.repeat([1, kmax, 1])
            e = self.v(torch.tanh(self.key + self.w_query(query)))

        elif self.atype == 'location':
            query = query.repeat([1, kmax, 1])
            conv_feat = self.conv(aw_prev.unsqueeze(3).transpose(3, 1)).squeeze(2)  # `[B, ch, kmax]`
            conv_feat = conv_feat.transpose(2, 1).contiguous()  # `[B, kmax, ch]`
            e = self.v(torch.tanh(self.key + self.w_query(query) + self.w_conv(conv_feat)))

        elif self.atype == 'dot':
            e = torch.bmm(self.key, self.w_query(query).transpose(-2, -1))

        elif self.atype == 'luong_dot':
            e = torch.bmm(self.key, query.transpose(-2, -1))

        elif self.atype == 'luong_general':
            e = torch.bmm(self.key, query.transpose(-2, -1))

        elif self.atype == 'luong_concat':
            query = query.repeat([1, kmax, 1])
            e = self.v(torch.tanh(self.w(torch.cat([self.key, query], dim=-1))))

        # Compute attention weights, context vector
        e = e.squeeze(2)  # `[B, kmax]`
        if self.mask is not None:
            e = e.masked_fill_(self.mask == 0, NEG_INF)
        if self.sigmoid_smoothing:
            aw = torch.sigmoid(e) / torch.sigmoid(e).sum(1).unsqueeze(1)
        else:
            aw = torch.softmax(e * self.sharpening_factor, dim=-1)
        aw = self.attn_dropout(aw)
        cv = torch.bmm(aw.unsqueeze(1), value)

        return cv, aw.unsqueeze(2)
