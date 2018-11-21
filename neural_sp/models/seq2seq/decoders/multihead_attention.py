# ! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-head attention layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.linear import LinearND

ATTENTION_TYPE = ['content', 'location', 'dot_product',
                  'luong_dot', 'luong_general', 'luong_concat']


class MultiheadAttentionMechanism(nn.Module):
    """Multi-head attention layer.

    Args:
        enc_num_units (int): the number of units in each layer of the encoder
        dec_num_units (int): the number of units in each layer of the decoder
        att_type (str): the type of attention mechanisms
        att_dim: (int) the dimension of the attention layer
        sharpening_factor (float): a sharpening factor in the softmax layer
            for attention weights
        sigmoid_smoothing (bool): replace the softmax layer for attention weights
            with the sigmoid function
        conv_out_channels (int): the number of channles of conv outputs.
            This is used for location-based attention.
        conv_kernel_size (int): the size of kernel.
            This must be the odd number.
        dropout (float):
        num_heads (int): the number of heads in the multi-head attention

    """

    def __init__(self,
                 enc_num_units,
                 dec_num_units,
                 att_type,
                 att_dim,
                 sharpening_factor=1,
                 sigmoid_smoothing=False,
                 conv_out_channels=10,
                 conv_kernel_size=201,
                 dropout=0,
                 num_heads=4):

        super(MultiheadAttentionMechanism, self).__init__()

        self.att_type = att_type
        self.att_dim = att_dim
        self.sharpening_factor = sharpening_factor
        self.sigmoid_smoothing = sigmoid_smoothing
        self.num_heads = num_heads
        self.enc_out_a = None
        self.mask = None

        # attention dropout applied AFTER the softmax layer
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        if self.att_type == 'content':
            self.w_enc = LinearND(enc_num_units, att_dim * num_heads)
            self.w_dec = LinearND(dec_num_units, att_dim * num_heads, bias=False)
            self.v = LinearND(att_dim * num_heads, 1, bias=False)

        elif self.att_type == 'location':
            assert conv_kernel_size % 2 == 1
            self.w_enc = LinearND(enc_num_units, att_dim * num_heads)
            self.w_dec = LinearND(dec_num_units, att_dim * num_heads, bias=False)
            self.w_conv = LinearND(conv_out_channels, att_dim * num_heads, bias=False)
            # self.convs = nn.ModuleList([nn.Conv1d(in_channels=1,
            #                                       out_channels=conv_out_channels,
            #                                       kernel_size=conv_kernel_size,
            #                                       stride=1,
            #                                       padding=conv_kernel_size // 2,
            #                                       bias=False) for _ in range(num_heads)])
            self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                                  out_channels=conv_out_channels,
                                                  kernel_size=(1, conv_kernel_size),
                                                  stride=1,
                                                  padding=(0, conv_kernel_size // 2),
                                                  bias=False) for _ in range(num_heads)])
            self.v = LinearND(att_dim * num_heads, 1, bias=False)

        elif self.att_type == 'dot_product':
            self.w_enc = LinearND(enc_num_units, att_dim * num_heads, bias=False)
            self.w_dec = LinearND(dec_num_units, att_dim * num_heads, bias=False)

        elif self.att_type == 'luong_dot':
            raise NotImplementedError()

        elif self.att_type == 'luong_general':
            raise NotImplementedError()

        elif self.att_type == 'luong_concat':
            raise NotImplementedError()

        else:
            raise TypeError("att_type should be one of [%s], you provided %s." %
                            (", ".join(ATTENTION_TYPE), att_type))

        self.w_out = LinearND(enc_num_units * num_heads, enc_num_units)

    def reset(self):
        self.enc_out_a = None
        self.mask = None

    def forward(self, enc_out, x_lens, dec_out, aw_step):
        """Forward computation.

        Args:
            enc_out (torch.autograd.Variable, float): `[B, T, enc_units]`
            x_lens (list): A list of length `[B]`
            dec_out (torch.autograd.Variable, float): `[B, 1, dec_units]`
            aw_step (torch.autograd.Variable, float): `[B, T, num_heads]`
        Returns:
            context_vec (torch.autograd.Variable, float): `[B, 1, enc_units]`
            aw_step (torch.autograd.Variable, float): `[B, T, num_heads]`

        """
        batch_size, enc_time = enc_out.size()[:2]

        if aw_step is None:
            aw_step = Variable(enc_out.data.new(batch_size, enc_time, self.num_heads).fill_(0.))

        # Pre-computation of encoder-side features for computing scores
        if self.enc_out_a is None:
            self.enc_out_a = self.w_enc(enc_out).view(batch_size, enc_time, self.num_heads, -1)
            self.enc_out_a = self.enc_out_a.transpose(1, 2)  # `[B, head, T, att_dim]`

        # Mask attention distribution
        if self.mask is None:
            self.mask = Variable(enc_out.new(batch_size, enc_time).fill_(1.))
            for b in range(batch_size):
                if x_lens[b] < enc_time:
                    self.mask[b, x_lens[b]:] = 0

        if self.att_type in ['content', 'location']:
            dec_out = dec_out.expand_as(torch.zeros((batch_size, 1, enc_time, dec_out.size(2))))

        conv_feats = []
        if self.att_type == 'location':
            for h in range(self.num_heads):
                # For 1D conv
                conv_feat = self.convs[h](aw_step[:, :, h].contiguous().unsqueeze(1))
                # For 2D conv
                conv_feat = self.convs[h](aw_step[:, :, h].contiguous().view(batch_size, 1, 1, enc_time)).squeeze(2)
                # `[B, conv_out_channels, T]`

                conv_feat = conv_feat.transpose(1, 2).contiguous()  # `[B, T, conv_out_channels]`
                conv_feats.append(conv_feat)

        if self.att_type == 'content':
            energy = self.v(F.tanh(self.enc_out_a + self.w_dec(dec_out))).squeeze(3)
            # `[B, head, T]`

        elif self.att_type == 'location':
            # For 1D conv
            # conv_feat = getattr(self, 'conv_head' + str(h))(
            #     aw_step[:, :, h].contiguous().unsqueeze(1))
            # For 2D conv
            conv_feat = getattr(self, 'conv_head' + str(h))(
                aw_step[:, :, h].contiguous().view(batch_size, 1, 1, enc_time)).squeeze(2)
            # `[B, conv_out_channels, T]`

            conv_feat = conv_feat.transpose(1, 2).contiguous()
            # `[B, T, conv_out_channels]`

            energy = getattr(self, 'v_head' + str(h))(F.tanh(
                self.enc_out_a[:, :, :, h] +
                getattr(self, 'w_dec_head' + str(h))(dec_out) +
                getattr(self, 'w_conv_head' + str(h))(conv_feat))).squeeze(2)

        elif self.att_type == 'dot_product':
            energy = torch.matmul(self.enc_out_a, self.w_dec(dec_out).transpose(-2, -1)).squeeze(3)

        elif self.att_type == 'luong_dot':
            raise NotImplementedError()

        elif self.att_type == 'luong_general':
            raise NotImplementedError()

        elif self.att_type == 'luong_concat':
            raise NotImplementedError()

        else:
            raise NotImplementedError()

        # Compute attention weights
        energy = energy.masked_fill_(self.mask == 0, -float('inf'))  # `[B, head, T]`
        if self.sigmoid_smoothing:
            aw_step = F.sigmoid(energy * self.sharpening_factor)
            # for b in range(batch_size):
            #     aw_step[b] /= aw_step[b].sum()
        else:
            aw_step = F.softmax(energy * self.sharpening_factor, dim=-1)
        # attention dropout
        if self.dropout is not None:
            aw_step = self.dropout(aw_step)
            # NOTE: apply the same dropout mask over multiple heads

        # Compute context vector (weighted sum of encoder outputs)
        # context_vec = torch.sum(enc_out * aw_step_h.unsqueeze(2), dim=1, keepdim=True)
        context_vec = torch.matmul(aw_step.unsqueeze(2), enc_out)
        # `[B, head, 1, T]` * `[B, head, ? , ? ]`

        return self.w_out(context_vec), aw_step
