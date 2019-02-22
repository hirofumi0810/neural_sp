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


class AttentionMechanism(nn.Module):
    """Single-head attention layer.

    Args:
        enc_nunits (int): the number of units in each layer of the encoder
        dec_nunits (int): the number of units in each layer of the decoder
        attn_type (str): the type of attention mechanisms
        attn_dim: (int) the dimension of the attention layer
        sharpening_factor (float): a sharpening factor in the softmax layer
            for attention weights
        sigmoid_smoothing (bool): replace the softmax layer for attention weights
            with the sigmoid function
        conv_out_channels (int): the number of channles of conv outputs.
            This is used for location-based attention.
        conv_kernel_size (int): the size of kernel.
            This must be the odd number.
        dropout (float):

    """

    def __init__(self,
                 enc_nunits,
                 dec_nunits,
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
        self.nheads = 1
        self.enc_out_a = None
        self.mask = None

        # attention dropout applied AFTER the softmax layer
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        if attn_type == 'add':
            self.w_enc = nn.Linear(enc_nunits, attn_dim)
            self.w_dec = nn.Linear(dec_nunits, attn_dim, bias=False)
            self.v = nn.Linear(attn_dim, 1, bias=False)

        elif attn_type == 'location':
            self.w_enc = nn.Linear(enc_nunits, attn_dim)
            self.w_dec = nn.Linear(dec_nunits, attn_dim, bias=False)
            self.w_conv = nn.Linear(conv_out_channels, attn_dim, bias=False)
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
            self.v = nn.Linear(attn_dim, 1, bias=False)

        elif attn_type == 'dot':
            self.w_enc = nn.Linear(enc_nunits, attn_dim, bias=False)
            self.w_dec = nn.Linear(dec_nunits, attn_dim, bias=False)

        elif attn_type == 'luong_dot':
            pass
            # NOTE: no additional parameters

        elif attn_type == 'luong_general':
            self.w_enc = nn.Linear(enc_nunits, dec_nunits, bias=False)

        elif attn_type == 'luong_concat':
            self.w = nn.Linear(enc_nunits + dec_nunits, attn_dim, bias=False)
            self.v = nn.Linear(attn_dim, 1, bias=False)

        else:
            raise ValueError(attn_type)

    def reset(self):
        self.enc_out_a = None
        self.mask = None

    def forward(self, enc_out, x_lens, dec_out, aw_step):
        """Forward computation.

        Args:
            enc_out (FloatTensor): `[B, T, enc_units]`
            x_lens (list): A list of length `[B]`
            dec_out (FloatTensor): `[B, 1, dec_units]`
            aw_step (FloatTensor): `[B, T]`
        Returns:
            context (FloatTensor): `[B, 1, enc_units]`
            aw_step (FloatTensor): `[B, T]`

        """
        bs, enc_time = enc_out.size()[:2]

        if aw_step is None:
            aw_step = enc_out.new_zeros(bs, enc_time)

        # Pre-computation of encoder-side features for computing scores
        if self.enc_out_a is None:
            if self.attn_type in ['add', 'location', 'dot', 'luong_general']:
                self.enc_out_a = self.w_enc(enc_out)

        # Mask attention distribution
        if self.mask is None:
            self.mask = enc_out.new_ones(bs, enc_time)
            for b in range(bs):
                if x_lens[b] < enc_time:
                    self.mask[b, x_lens[b]:] = 0

        if self.attn_type == 'add':
            dec_out = dec_out.expand_as(torch.zeros((bs, enc_time, dec_out.size(2))))
            energy = self.v(F.tanh(self.enc_out_a + self.w_dec(dec_out))).squeeze(2)

        elif self.attn_type == 'location':
            dec_out = dec_out.expand_as(torch.zeros((bs, enc_time, dec_out.size(2))))
            # For 1D conv
            # conv_feat = self.conv(aw_step[:, :].contiguous().unsqueeze(1))
            # For 2D conv
            conv_feat = self.conv(aw_step.view(bs, 1, 1, enc_time)).squeeze(2)  # `[B, conv_out_channels, T]`
            conv_feat = conv_feat.transpose(1, 2).contiguous()  # `[B, T, conv_out_channels]`
            energy = self.v(F.tanh(self.enc_out_a + self.w_dec(dec_out) + self.w_conv(conv_feat))).squeeze(2)

        elif self.attn_type == 'dot':
            energy = torch.matmul(self.enc_out_a, self.w_dec(dec_out).transpose(-2, -1)).squeeze(2)

        elif self.attn_type == 'luong_dot':
            energy = torch.matmul(enc_out, dec_out.transpose(-2, -1)).squeeze(2)

        elif self.attn_type == 'luong_general':
            energy = torch.matmul(self.enc_out_a, dec_out.transpose(-2, -1)).squeeze(2)

        elif self.attn_type == 'luong_concat':
            dec_out = dec_out.expand_as(torch.zeros((bs, enc_time, dec_out.size(2))))
            energy = self.v(F.tanh(self.w(torch.cat([enc_out, dec_out], dim=-1)))).squeeze(2)

        # Compute attention weights
        energy = energy.masked_fill_(self.mask == 0, -float('inf'))  # `[B, T]`
        if self.sigmoid_smoothing:
            aw_step = F.sigmoid(energy) / F.sigmoid(energy).sum(-1).unsqueeze(-1)
        else:
            aw_step = F.softmax(energy * self.sharpening_factor, dim=-1)  # `[B, T]`
        # attention dropout
        if self.dropout is not None:
            aw_step = self.dropout(aw_step)

        # Compute context vector (weighted sum of encoder outputs)
        context = torch.matmul(aw_step.unsqueeze(1), enc_out)

        return context, aw_step
