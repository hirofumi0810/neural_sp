# ! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-head attention layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.linear import LinearND


class MultiheadAttentionMechanism(nn.Module):
    """Multi-headed attention layer.

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
        nheads (int): the number of heads in the multi-head attention

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
                 dropout=0,
                 nheads=4):

        super(MultiheadAttentionMechanism, self).__init__()

        self.attn_type = attn_type
        self.attn_dim = attn_dim
        self.sharpening_factor = sharpening_factor
        self.sigmoid_smoothing = sigmoid_smoothing
        self.nheads = nheads
        self.enc_out_a = None
        self.mask = None

        # attention dropout applied AFTER the softmax layer
        if dropout > 0:
            self.dropout = nn.ModuleList([nn.Dropout(p=dropout)] * nheads)
        else:
            self.dropout = None

        if attn_type == 'add':
            self.w_enc = nn.ModuleList([LinearND(enc_nunits, attn_dim)] * nheads)
            self.w_dec = nn.ModuleList([LinearND(dec_nunits, attn_dim, bias=False)] * nheads)
            self.v = nn.ModuleList([LinearND(attn_dim, 1, bias=False)] * nheads)

        elif attn_type == 'location':
            self.w_enc = nn.ModuleList([LinearND(enc_nunits, attn_dim)] * nheads)
            self.w_dec = nn.ModuleList([LinearND(dec_nunits, attn_dim, bias=False)] * nheads)
            self.w_conv = nn.ModuleList([LinearND(conv_out_channels, attn_dim, bias=False)] * nheads)
            # self.conv = nn.ModuleList([nn.Conv1d(in_channels=1,
            #                                       out_channels=conv_out_channels,
            #                                       kernel_size=conv_kernel_size * 2 + 1,
            #                                       stride=1,
            #                                       padding=conv_kernel_size,
            #                                       bias=False) for _ in range(nheads)] * nheads)
            self.conv = nn.ModuleList([nn.Conv2d(in_channels=1,
                                                 out_channels=conv_out_channels,
                                                 kernel_size=(1, conv_kernel_size * 2 + 1),
                                                 stride=1,
                                                 padding=(0, conv_kernel_size),
                                                 bias=False) for _ in range(nheads)] * nheads)
            self.v = nn.ModuleList([LinearND(attn_dim, 1, bias=False)] * nheads)

        elif attn_type == 'dot':
            self.w_enc = nn.ModuleList([LinearND(enc_nunits, attn_dim, bias=False)] * nheads)
            self.w_dec = nn.ModuleList([LinearND(dec_nunits, attn_dim, bias=False)] * nheads)

        elif attn_type == 'luong_dot':
            pass
            # NOTE: no additional parameters

        elif attn_type == 'luong_general':
            self.w_enc = nn.ModuleList([LinearND(enc_nunits, dec_nunits, bias=False)] * nheads)

        elif attn_type == 'luong_concat':
            self.w = nn.ModuleList([LinearND(enc_nunits + dec_nunits, attn_dim, bias=False)] * nheads)
            self.v = nn.ModuleList([LinearND(attn_dim, 1, bias=False)] * nheads)

        else:
            raise ValueError(attn_type)

        self.w_out = LinearND(enc_nunits * nheads, enc_nunits)

    def reset(self):
        self.enc_out_a = None
        self.mask = None

    def forward(self, enc_out, x_lens, dec_out, aw_step):
        """Forward computation.

        Args:
            enc_out (FloatTensor): `[B, T, enc_units]`
            x_lens (list): A list of length `[B]`
            dec_out (FloatTensor): `[B, 1, dec_units]`
            aw_step (FloatTensor): `[B, T, nheads]`
        Returns:
            context (FloatTensor): `[B, 1, enc_units]`
            aw_step (FloatTensor): `[nheads, B, T]`

        """
        bs, enc_time = enc_out.size()[:2]

        if aw_step is None:
            aw_step = enc_out.new_ones(self.nheads, bs, enc_time)

        # Pre-computation of encoder-side features for computing scores
        if self.enc_out_a is None:
            if self.attn_type in ['add', 'location', 'dot', 'luong_general']:
                self.enc_out_a = [self.w_enc[h](enc_out) for h in range(self.nheads)]

        # Mask attention distribution
        if self.mask is None:
            self.mask = enc_out.new_ones(bs, enc_time)
            for b in range(bs):
                if x_lens[b] < enc_time:
                    self.mask[b, x_lens[b]:] = 0
            # TODO(hirofumi): prepare mask per attention

        # Compute per head
        contexts = []
        aw_steps = []
        for h in range(self.nheads):
            if self.attn_type == 'add':
                dec_out_h = dec_out.expand_as(torch.zeros((bs, enc_time, dec_out.size(2))))
                energy_h = self.v[h](F.tanh(self.enc_out_a[h] + self.w_dec[h](dec_out_h))).squeeze(2)

            elif self.attn_type == 'location':
                dec_out_h = dec_out.expand_as(torch.zeros((bs, enc_time, dec_out.size(2))))
                # For 1D conv
                # conv_feat = self.conv[h](aw_step[h][:, :].contiguous().unsqueeze(1))
                # For 2D conv
                conv_feat_h = self.conv[h](aw_step[h].view(bs, 1, 1, enc_time)
                                           ).squeeze(2)  # `[B, conv_out_channels, T]`
                conv_feat_h = conv_feat_h.transpose(1, 2).contiguous()  # `[B, T, conv_out_channels]`
                energy_h = self.v[h](F.tanh(self.enc_out_a[h] + self.w_dec[h]
                                            (dec_out_h) + self.w_conv[h](conv_feat_h))).squeeze(2)

            elif self.attn_type == 'dot':
                energy_h = torch.matmul(self.enc_out_a[h], self.w_dec[h](dec_out).transpose(-2, -1)).squeeze(2)

            elif self.attn_type == 'luong_dot':
                energy_h = torch.matmul(enc_out, dec_out.transpose(-2, -1)).squeeze(2)

            elif self.attn_type == 'luong_general':
                energy_h = torch.matmul(self.enc_out_a[h], dec_out.transpose(-2, -1)).squeeze(2)

            elif self.attn_type == 'luong_concat':
                dec_out_h = dec_out.expand_as(torch.zeros((bs, enc_time, dec_out.size(2))))
                energy_h = self.v[h](F.tanh(self.w[h](torch.cat([enc_out, dec_out_h], dim=-1)))).squeeze(2)

            # Compute attention weights
            energy_h = energy_h.masked_fill_(self.mask == 0, -float('inf'))  # `[B, T]`
            if self.sigmoid_smoothing:
                aw_step_h = F.sigmoid(energy_h) / F.sigmoid(energy_h).sum(-1).unsqueeze(-1)
            else:
                aw_step_h = F.softmax(energy_h * self.sharpening_factor, dim=-1)  # `[B, T]`
            # attention dropout
            if self.dropout is not None:
                aw_step_h = self.dropout[h](aw_step_h)
            aw_steps.append(aw_step_h)

            # Compute context vector (weighted sum of encoder outputs)
            context = torch.matmul(aw_step[h].unsqueeze(1), enc_out)
            contexts.append(context)

        return self.w_out(torch.cat(contexts, dim=-1)), torch.stack(aw_steps, dim=0)


class TransformerMultiheadAttentionMechanism(nn.Module):
    def __init__(self, nheads, d_model, dropout):
        """Multi-headed attention layer impelemented in Transformer.

        Args:
            nheads (int):
            d_model (int):
            dropout (float):

        """
        super(TransformerMultiheadAttentionMechanism, self).__init__()

        assert d_model % nheads == 0
        # We assume d_v always equals d_k
        self.d_model = d_model
        self.d_k = d_model // nheads
        self.nheads = nheads

        self.w_key = nn.Linear(d_model, d_model)
        self.w_value = nn.Linear(d_model, d_model)
        self.w_query = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
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
                0: place to pad with -inf
                1: otherwise
        Returns:
            context (FloatTensor): `[B, query_time, key_time]`
            aw (FloatTensor): `[B, nheads, query_time, key_time]`

        """
        bs = query.size(0)

        # 1) Do all the linear projections in batch from d_model => head x d_k
        key = self.w_key(key).view(bs, -1, self.nheads, self.d_k).transpose(1, 2)
        value = self.w_value(value).view(bs, -1, self.nheads, self.d_k).transpose(1, 2)
        query = self.w_query(query).view(bs, -1, self.nheads, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        energy = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)  # Same mask applied to all heads.
            # scores_test = energy.masked_fill(mask == 0, -float('inf'))  # this is buggy
            # scores_test = energy.masked_fill(mask == 0, -10e9)  # this is ok
            energy = energy.masked_fill(mask == 0, -1024)
        aw = self.dropout(F.softmax(energy, dim=-1))
        context = torch.matmul(aw, value)

        # 3) "Concat" using a view and apply a final linear.
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.nheads * self.d_k)
        context = self.w_out(context)

        return context, aw
