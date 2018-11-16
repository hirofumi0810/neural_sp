# ! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-head attention layer."""

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
                 num_heads=1):

        super(MultiheadAttentionMechanism, self).__init__()

        self.att_type = att_type
        self.att_dim = att_dim
        self.sharpening_factor = sharpening_factor
        self.sigmoid_smoothing = sigmoid_smoothing
        self.num_heads = num_heads
        self.enc_out_a = None

        self.w_mha = LinearND(enc_num_units * num_heads, enc_num_units)

        for h in six.moves.range(num_heads):
            if self.att_type == 'content':
                setattr(self, 'w_enc_head' + str(h), LinearND(enc_num_units, att_dim))
                setattr(self, 'w_dec_head' + str(h), LinearND(dec_num_units, att_dim, bias=False))
                setattr(self, 'v_head' + str(h), LinearND(att_dim, 1, bias=False))

            elif self.att_type == 'location':
                assert conv_kernel_size % 2 == 1
                setattr(self, 'w_enc_head' + str(h), LinearND(enc_num_units, att_dim))
                setattr(self, 'w_dec_head' + str(h), LinearND(dec_num_units, att_dim, bias=False))
                setattr(self, 'w_conv_head' + str(h), LinearND(conv_out_channels, att_dim, bias=False))
                # setattr(self, 'conv_head' + str(h),
                #         nn.Conv1d(in_channels=1,
                #                   out_channels=conv_out_channels,
                #                   kernel_size=conv_kernel_size,
                #                   stride=1,
                #                   padding=conv_kernel_size // 2,
                #                   bias=False))
                setattr(self, 'conv_head' + str(h),
                        nn.Conv2d(in_channels=1,
                                  out_channels=conv_out_channels,
                                  kernel_size=(1, conv_kernel_size),
                                  stride=1,
                                  padding=(0, conv_kernel_size // 2),
                                  bias=False))
                setattr(self, 'v_head' + str(h), LinearND(att_dim, 1, bias=False))

            elif self.att_type == 'dot_product':
                setattr(self, 'w_enc_head' + str(h), LinearND(enc_num_units, dec_num_units, bias=False))

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
            volatile = enc_out.volatile
            aw_step = Variable(enc_out.data.new(batch_size, enc_time, self.num_heads).fill_(0.), volatile=volatile)

        # Pre-computation of encoder-side features for computing scores
        if self.enc_out_a is None:
            self.enc_out_a = []
            for h in six.moves.range(self.num_heads):
                self.enc_out_a += [getattr(self, 'w_enc_head' + str(h))(enc_out)]
            self.enc_out_a = torch.stack(self.enc_out_a, dim=-1)

        if self.att_type in ['content', 'location']:
            dec_out = dec_out.expand_as(torch.zeros((batch_size, enc_time, dec_out.size(2))))

        energy = []
        for h in six.moves.range(self.num_heads):
            if self.att_type == 'content':
                ##############################################################
                # energy = <v, tanh(W([h_dec; h_enc] + b))>
                ##############################################################
                energy_head = getattr(self, 'v_head' + str(h))(F.tanh(
                    self.enc_out_a[:, :, :, h] +
                    getattr(self, 'w_dec_head' + str(h))(dec_out))).squeeze(2)

            elif self.att_type == 'location':
                ##############################################################
                # f = F * α_{i-1}
                # energy = <v, tanh(W([h_dec; h_enc] + W_conv(f) + b))>
                ##############################################################
                # For 1D conv
                # conv_feat = getattr(self, 'conv_head' + str(h))(
                #     aw_step[:, :, h].contiguous().unsqueeze(dim=1))
                # For 2D conv
                conv_feat = getattr(self, 'conv_head' + str(h))(
                    aw_step[:, :, h].contiguous().view(batch_size, 1, 1, enc_time)).squeeze(2)
                # -> `[B, conv_out_channels, T]`

                conv_feat = conv_feat.transpose(1, 2).contiguous()
                # -> `[B, T, conv_out_channels]`

                energy_head = getattr(self, 'v_head' + str(h))(F.tanh(
                    self.enc_out_a[:, :, :, h] +
                    getattr(self, 'w_dec_head' + str(h))(dec_out) +
                    getattr(self, 'w_conv_head' + str(h))(conv_feat))).squeeze(2)

            elif self.att_type == 'dot_product':
                ##############################################################
                # energy = <W_enc(h_enc), h_dec>
                ##############################################################
                energy_head = torch.bmm(self.enc_out_a[:, :, :, h], dec_out.transpose(1, 2)).squeeze(2)

            elif self.att_type == 'luong_dot':
                raise NotImplementedError()

            elif self.att_type == 'luong_general':
                raise NotImplementedError()

            elif self.att_type == 'luong_concat':
                raise NotImplementedError()

            else:
                raise NotImplementedError()

            energy.append(energy_head)

        # Mask attention distribution
        energy_mask = Variable(enc_out.data.new(batch_size, enc_time).fill_(1))
        for b in six.moves.range(batch_size):
            if x_lens[b] < enc_time:
                energy_mask.data[b, x_lens[b]:] = -1024.0

        context_vec, aw_step = [], []
        for h in six.moves.range(self.num_heads):
            energy[h] *= energy_mask
            # NOTE: energy[h]: `[B, T]`

            # Sharpening
            energy[h] *= self.sharpening_factor

            # Compute attention weights
            if self.sigmoid_smoothing:
                aw_step_head = F.sigmoid(energy[h])
                # for b in six.moves.range(batch_size):
                #     aw_step_head.data[b] /= aw_step_head.data[b].sum()
            else:
                aw_step_head = F.softmax(energy[h], dim=-1)
            aw_step.append(aw_step_head)

            # Compute context vector (weighted sum of encoder outputs)
            context_vec_head = torch.sum(enc_out * aw_step_head.unsqueeze(2), dim=1, keepdim=True)
            context_vec.append(context_vec_head)

        # Concatenate all convtext vectors and attention distributions
        context_vec = torch.cat(context_vec, dim=-1)
        aw_step = torch.stack(aw_step, dim=-1)

        context_vec = self.w_mha(context_vec)

        return context_vec, aw_step
