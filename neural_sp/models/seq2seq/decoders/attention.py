# ! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Single-head attention layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.linear import LinearND

ATTENTION_TYPE = ['content', 'location', 'dot_product',
                  'luong_dot', 'luong_general', 'luong_concat']


class AttentionMechanism(nn.Module):
    """Single-head attention layer.

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

    """

    def __init__(self,
                 enc_num_units,
                 dec_num_units,
                 att_type,
                 att_dim,
                 sharpening_factor=1,
                 sigmoid_smoothing=False,
                 conv_out_channels=10,
                 conv_kernel_size=201):

        super(AttentionMechanism, self).__init__()

        self.att_type = att_type
        self.att_dim = att_dim
        self.sharpening_factor = sharpening_factor
        self.sigmoid_smoothing = sigmoid_smoothing
        self.num_heads = 1
        self.enc_out_a = None

        if self.att_type == 'content':
            self.w_enc = LinearND(enc_num_units, att_dim)
            self.w_dec = LinearND(dec_num_units, att_dim, bias=False)
            self.v = LinearND(att_dim, 1, bias=False)

        elif self.att_type == 'location':
            assert conv_kernel_size % 2 == 1
            self.w_enc = LinearND(enc_num_units, att_dim)
            self.w_dec = LinearND(dec_num_units, att_dim, bias=False)
            self.w_conv = LinearND(conv_out_channels, att_dim, bias=False)
            # self.conv = nn.Conv1d(in_channels=1,
            #                       out_channels=conv_out_channels,
            #                       kernel_size=conv_kernel_size,
            #                       stride=1,
            #                       padding=conv_kernel_size // 2,
            #                       bias=False)
            self.conv = nn.Conv2d(in_channels=1,
                                  out_channels=conv_out_channels,
                                  kernel_size=(1, conv_kernel_size),
                                  stride=1,
                                  padding=(0, conv_kernel_size // 2),
                                  bias=False)
            self.v = LinearND(att_dim, 1, bias=False)

        elif self.att_type == 'dot_product':
            self.w_enc = LinearND(enc_num_units, att_dim, bias=False)
            self.w_dec = LinearND(dec_num_units, att_dim, bias=False)

        elif self.att_type == 'luong_dot':
            raise NotImplementedError()

        elif self.att_type == 'luong_general':
            raise NotImplementedError()

        elif self.att_type == 'luong_concat':
            raise NotImplementedError()

        else:
            raise TypeError("att_type should be one of [%s], you provided %s." %
                            (", ".join(ATTENTION_TYPE), att_type))

    def reset(self):
        self.enc_out_a = None

    def forward(self, enc_out, x_lens, dec_out, aw_step):
        """Forward computation.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, enc_num_units]`
            x_lens (list): A list of length `[B]`
            dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, dec_num_units]`
            aw_step (torch.autograd.Variable, float): A tensor of size
                `[B, T]`
        Returns:
            context_vec (torch.autograd.Variable, float): A tensor of size
                `[B, 1, enc_num_units]`
            aw_step (torch.autograd.Variable, float): A tensor of size
                `[B, T]`

        """
        batch_size, enc_time = enc_out.size()[:2]

        if aw_step is None:
            volatile = enc_out.volatile
            aw_step = Variable(enc_out.new(batch_size, enc_time).fill_(0.), volatile=volatile)

        # Pre-computation of encoder-side features for computing scores
        if self.enc_out_a is None:
            self.enc_out_a = self.w_enc(enc_out)

        if self.att_type in ['content', 'location']:
            dec_out = dec_out.expand_as(torch.zeros((batch_size, enc_time, dec_out.size(2))))

        if self.att_type == 'content':
            energy = self.v(F.tanh(self.enc_out_a + self.w_dec(dec_out))).squeeze(2)

        elif self.att_type == 'location':
            # For 1D conv
            # conv_feat = self.conv(aw_step[:, :].contiguous().unsqueeze(dim=1))
            # For 2D conv
            conv_feat = self.conv(aw_step.view(batch_size, 1, 1, enc_time)).squeeze(2)  # -> `[B, conv_out_channels, T]`
            conv_feat = conv_feat.transpose(1, 2).contiguous()  # -> `[B, T, conv_out_channels]`
            energy = self.v(F.tanh(self.enc_out_a + self.w_dec(dec_out) + self.w_conv(conv_feat))).squeeze(2)

        elif self.att_type == 'dot_product':
            energy = torch.bmm(self.enc_out_a, self.w_dec(dec_out).transpose(1, 2)).squeeze(2)

        elif self.att_type == 'luong_dot':
            raise NotImplementedError()

        elif self.att_type == 'luong_general':
            raise NotImplementedError()

        elif self.att_type == 'luong_concat':
            raise NotImplementedError()

        # Mask attention distribution
        energy_mask = Variable(enc_out.new(batch_size, enc_time).fill_(1.))
        for b in six.moves.range(batch_size):
            if x_lens[b] < enc_time:
                energy_mask[b, x_lens[b]:] = -1024.0
        energy *= energy_mask
        # NOTE: energy: `[B, T]`

        # Compute attention weights
        if self.sigmoid_smoothing:
            aw_step = F.sigmoid(energy * self.sharpening_factor)
            # for b in six.moves.range(batch_size):
            #     aw_step[b] /= aw_step[b].sum()
        else:
            aw_step = F.softmax(energy * self.sharpening_factor, dim=-1)

        # Compute context vector (weighted sum of encoder outputs)
        context_vec = torch.sum(enc_out * aw_step.unsqueeze(2), dim=1, keepdim=True)

        return context_vec, aw_step
