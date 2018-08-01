# ! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention layer (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from src.models.pytorch_v3.linear import LinearND

ATTENTION_TYPE = ['content', 'location', 'dot_product', 'rnn_attention', 'coverage']


class AttentionMechanism(nn.Module):
    """Single-head attention layer.

    Args:
        enc_n_units (int): the number of units in each layer of the encoder
        dec_n_units (int): the number of units in each layer of the decoder
        att_type (string): the type of attention
        att_dim: (int) the dimension of the attention layer
        sharpening_factor (float): a sharpening factor in the softmax
            layer for computing attention weights
        sigmoid_smoothing (bool): if True, replace softmax function
            in computing attention weights with sigmoid function for smoothing
        out_channels (int): the number of channles of conv outputs.
            This is used for location-based attention.
        kernel_size (int): the size of kernel.
            This must be the odd number.

    """

    def __init__(self,
                 enc_n_units,
                 dec_n_units,
                 att_type,
                 att_dim,
                 sharpening_factor=1,
                 sigmoid_smoothing=False,
                 out_channels=10,
                 kernel_size=201):

        super(AttentionMechanism, self).__init__()

        self.att_type = att_type
        self.att_dim = att_dim
        self.sharpening_factor = sharpening_factor
        self.sigmoid_smoothing = sigmoid_smoothing
        self.n_heads = 1
        self.enc_out_a = None

        if self.att_type == 'content':
            self.w_enc = LinearND(enc_n_units, att_dim, bias=True)
            self.w_dec = LinearND(dec_n_units, att_dim, bias=False)
            self.v = LinearND(att_dim, 1, bias=False)

        elif self.att_type == 'location':
            assert kernel_size % 2 == 1
            self.w_enc = LinearND(enc_n_units, att_dim, bias=True)
            self.w_dec = LinearND(dec_n_units, att_dim, bias=False)
            self.w_conv = LinearND(out_channels, att_dim, bias=False)
            # self.conv = nn.Conv1d(in_channels=1,
            #                             out_channels=out_channels,
            #                             kernel_size=kernel_size,
            #                             stride=1,
            #                             padding=kernel_size // 2,
            #                             bias=False)
            self.conv = nn.Conv2d(in_channels=1,
                                  out_channels=out_channels,
                                  kernel_size=(1, kernel_size),
                                  stride=1,
                                  padding=(0, kernel_size // 2),
                                  bias=False)
            self.v = LinearND(att_dim, 1, bias=False)

        elif self.att_type == 'dot_product':
            self.w_enc = LinearND(enc_n_units, dec_n_units, bias=False)

        elif self.att_type == 'rnn_attention':
            raise NotImplementedError

        elif self.att_type == 'coverage':
            self.w_enc = LinearND(enc_n_units, att_dim, bias=True)
            self.w_dec = LinearND(dec_n_units, att_dim, bias=False)
            self.w_cov = LinearND(enc_n_units, att_dim, bias=False)
            self.v = LinearND(att_dim, 1, bias=False)
            self.aw_cumsum = None

        else:
            raise TypeError("att_type should be one of [%s], you provided %s." %
                            (", ".join(ATTENTION_TYPE), att_type))

    def reset(self):
        self.enc_out_a = None

    def forward(self, enc_out, x_lens, dec_out, aw_step):
        """Forward computation.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, enc_n_units]`
            x_lens (list): A list of length `[B]`
            dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, dec_n_units]`
            aw_step (torch.autograd.Variable, float): A tensor of size
                `[B, T]`
        Returns:
            context_vec (torch.autograd.Variable, float): A tensor of size
                `[B, 1, enc_n_units]`
            aw_step (torch.autograd.Variable, float): A tensor of size
                `[B, T]`

        """
        batch, enc_time = enc_out.size()[:2]

        if aw_step is None:
            volatile = enc_out.volatile
            aw_step = Variable(enc_out.data.new(
                batch, enc_time).fill_(0.), volatile=volatile)

        # Pre-computation of encoder-side features for computing scores
        if self.enc_out_a is None:
            self.enc_out_a = self.w_enc(enc_out)

        if self.att_type in ['content', 'location', 'coverage']:
            dec_out = dec_out.expand_as(torch.zeros(
                (batch, enc_time, dec_out.size(2))))

        if self.att_type == 'content':
            ##############################################################
            # energy = <v, tanh(W([h_dec; h_enc] + b))>
            ##############################################################
            energy = self.v(F.tanh(
                self.enc_out_a + self.w_dec(dec_out))).squeeze(2)

        elif self.att_type == 'location':
            ##############################################################
            # f = F * α_{i-1}
            # energy = <v, tanh(W([h_dec; h_enc] + W_conv(f) + b))>
            ##############################################################
            # For 1D conv
            # conv_feat = self.conv(aw_step[:, :].contiguous().unsqueeze(dim=1))
            # For 2D conv
            conv_feat = self.conv(aw_step.view(batch, 1, 1, enc_time)).squeeze(2)
            # -> `[B, out_channels, T]`

            conv_feat = conv_feat.transpose(1, 2).contiguous()
            # -> `[B, T, out_channels]`

            energy = self.v(F.tanh(
                self.enc_out_a + self.w_dec(dec_out) + self.w_conv(conv_feat))).squeeze(2)

        elif self.att_type == 'dot_product':
            ##############################################################
            # energy = <W_enc(h_enc), h_dec>
            ##############################################################
            energy = torch.bmm(self.enc_out_a, dec_out.transpose(1, 2)).squeeze(2)

        elif self.att_type == 'rnn_attention':
            raise NotImplementedError

        elif self.att_type == 'coverage':
            raise NotImplementedError
            ##############################################################
            # energy = <v, tanh(W([h_dec; h_enc, coverage] + b))>
            ##############################################################
            # Sum all previous attention weights
            if self.aw_cumsum is None:
                self.aw_cumsum = aw_step
            else:
                self.aw_cumsum += aw_step

            energy = self.v(F.tanh(
                self.enc_out_a[:, :, :] + self.w_dec(dec_out) + self.w_cov(self.aw_cumsum))).squeeze(2)

        else:
            raise NotImplementedError

        # Mask attention distribution
        energy_mask = Variable(enc_out.data.new(batch, enc_time).fill_(1))
        for b in six.moves.range(batch):
            if x_lens[b] < enc_time:
                energy_mask.data[b, x_lens[b]:] = 0

        energy *= energy_mask
        # NOTE: energy: `[B, T]`

        # Sharpening
        energy *= self.sharpening_factor

        # Compute attention weights
        if self.sigmoid_smoothing:
            aw_step = F.sigmoid(energy)
            # for b in six.moves.range(batch):
            #     aw_step.data[b] /= aw_step.data[b].sum()
        else:
            aw_step = F.softmax(energy, dim=-1)

        # Compute context vector (weighted sum of encoder outputs)
        context_vec = torch.sum(enc_out * aw_step.unsqueeze(2), dim=1, keepdim=True)

        return context_vec, aw_step


class MultiheadAttentionMechanism(nn.Module):
    """Multi-head attention layer.

    Args:
        enc_n_units (int): the number of units in each layer of the encoder
        dec_n_units (int): the number of units in each layer of the decoder
        att_type (string): the type of attention
        att_dim: (int) the dimension of the attention layer
        sharpening_factor (float): a sharpening factor in the softmax
            layer for computing attention weights
        sigmoid_smoothing (bool): if True, replace softmax function
            in computing attention weights with sigmoid function for smoothing
        out_channels (int): the number of channles of conv outputs.
            This is used for location-based attention.
        kernel_size (int): the size of kernel.
            This must be the odd number.
        n_heads (int): the number of heads in the multi-head attention

    """

    def __init__(self,
                 enc_n_units,
                 dec_n_units,
                 att_type,
                 att_dim,
                 sharpening_factor=1,
                 sigmoid_smoothing=False,
                 out_channels=10,
                 kernel_size=201,
                 n_heads=1):

        super(MultiheadAttentionMechanism, self).__init__()

        self.att_type = att_type
        self.att_dim = att_dim
        self.sharpening_factor = sharpening_factor
        self.sigmoid_smoothing = sigmoid_smoothing
        self.n_heads = n_heads
        self.enc_out_a = None

        self.w_mha = LinearND(enc_n_units * n_heads, enc_n_units)

        for h in six.moves.range(n_heads):
            if self.att_type == 'content':
                setattr(self, 'w_enc_head' + str(h), LinearND(enc_n_units, att_dim, bias=True))
                setattr(self, 'w_dec_head' + str(h), LinearND(dec_n_units, att_dim, bias=False))
                setattr(self, 'v_head' + str(h), LinearND(att_dim, 1, bias=False))

            elif self.att_type == 'location':
                assert kernel_size % 2 == 1
                setattr(self, 'w_enc_head' + str(h), LinearND(enc_n_units, att_dim, bias=True))
                setattr(self, 'w_dec_head' + str(h), LinearND(dec_n_units, att_dim, bias=False))
                setattr(self, 'w_conv_head' + str(h), LinearND(out_channels, att_dim, bias=False))
                # setattr(self, 'conv_head' + str(h),
                #         nn.Conv1d(in_channels=1,
                #                   out_channels=out_channels,
                #                   kernel_size=kernel_size,
                #                   stride=1,
                #                   padding=kernel_size // 2,
                #                   bias=False))
                setattr(self, 'conv_head' + str(h),
                        nn.Conv2d(in_channels=1,
                                  out_channels=out_channels,
                                  kernel_size=(1, kernel_size),
                                  stride=1,
                                  padding=(0, kernel_size // 2),
                                  bias=False))
                setattr(self, 'v_head' + str(h), LinearND(att_dim, 1, bias=False))

            elif self.att_type == 'dot_product':
                setattr(self, 'w_enc_head' + str(h), LinearND(enc_n_units, dec_n_units, bias=False))

            elif self.att_type == 'rnn_attention':
                raise NotImplementedError

            elif self.att_type == 'coverage':
                setattr(self, 'w_enc_head' + str(h), LinearND(enc_n_units, att_dim, bias=True))
                setattr(self, 'w_dec_head' + str(h), LinearND(dec_n_units, att_dim, bias=False))
                setattr(self, 'w_cov_head' + str(h), LinearND(enc_n_units, att_dim, bias=False))
                setattr(self, 'v_head' + str(h), LinearND(att_dim, 1, bias=False))
                self.aw_cumsum = None

            else:
                raise TypeError(
                    "att_type should be one of [%s], you provided %s." %
                    (", ".join(ATTENTION_TYPE), att_type))

    def reset(self):
        self.enc_out_a = None

    def forward(self, enc_out, x_lens, dec_out, aw_step):
        """Forward computation.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, enc_n_units]`
            x_lens (list): A list of length `[B]`
            dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, dec_n_units]`
            aw_step (torch.autograd.Variable, float): A tensor of size
                `[B, T, n_heads]`
        Returns:
            context_vec (torch.autograd.Variable, float): A tensor of size
                `[B, 1, enc_n_units]`
            aw_step (torch.autograd.Variable, float): A tensor of size
                `[B, T, n_heads]`

        """
        batch, enc_time = enc_out.size()[:2]

        if aw_step is None:
            volatile = enc_out.volatile
            aw_step = Variable(enc_out.data.new(
                batch, enc_time, self.n_heads).fill_(0.), volatile=volatile)

        # Pre-computation of encoder-side features for computing scores
        if self.enc_out_a is None:
            self.enc_out_a = []
            for h in six.moves.range(self.n_heads):
                self.enc_out_a += [getattr(self, 'w_enc_head' + str(h))(enc_out)]
            self.enc_out_a = torch.stack(self.enc_out_a, dim=-1)

        if self.att_type in ['content', 'location', 'coverage']:
            dec_out = dec_out.expand_as(torch.zeros((batch, enc_time, dec_out.size(2))))

        energy = []
        for h in six.moves.range(self.n_heads):
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
                    aw_step[:, :, h].contiguous().view(batch, 1, 1, enc_time)).squeeze(2)
                # -> `[B, out_channels, T]`

                conv_feat = conv_feat.transpose(1, 2).contiguous()
                # -> `[B, T, out_channels]`

                energy_head = getattr(self, 'v_head' + str(h))(F.tanh(
                    self.enc_out_a[:, :, :, h] +
                    getattr(self, 'w_dec_head' + str(h))(dec_out) +
                    getattr(self, 'w_conv_head' + str(h))(conv_feat))).squeeze(2)

            elif self.att_type == 'dot_product':
                ##############################################################
                # energy = <W_enc(h_enc), h_dec>
                ##############################################################
                energy_head = torch.bmm(
                    self.enc_out_a[:, :, :, h], dec_out.transpose(1, 2)).squeeze(2)

            elif self.att_type == 'rnn_attention':
                raise NotImplementedError

            elif self.att_type == 'coverage':
                raise NotImplementedError
                ##############################################################
                # energy = <v, tanh(W([h_dec; h_enc, coverage] + b))>
                ##############################################################
                # Sum all previous attention weights
                if self.aw_cumsum is None:
                    self.aw_cumsum = aw_step
                else:
                    self.aw_cumsum += aw_step

                energy_head = getattr(self, 'v_head' + str(h))(F.tanh(
                    self.enc_out_a[:, :, :, h] +
                    getattr(self, 'w_dec_head' + str(h))(dec_out) +
                    getattr(self, 'w_cov_head' + str(h))(self.aw_cumsum))).squeeze(2)

            else:
                raise NotImplementedError

            energy.append(energy_head)

        # Mask attention distribution
        energy_mask = Variable(enc_out.data.new(batch, enc_time).fill_(1))
        for b in six.moves.range(batch):
            if x_lens[b] < enc_time:
                energy_mask.data[b, x_lens[b]:] = 0

        context_vec, aw_step = [], []
        for h in six.moves.range(self.n_heads):
            energy[h] *= energy_mask
            # NOTE: energy[h]: `[B, T]`

            # Sharpening
            energy[h] *= self.sharpening_factor

            # Compute attention weights
            if self.sigmoid_smoothing:
                aw_step_head = F.sigmoid(energy[h])
                # for b in six.moves.range(batch):
                #     aw_step_head.data[b] /= aw_step_head.data[b].sum()
            else:
                aw_step_head = F.softmax(energy[h], dim=-1)
            aw_step.append(aw_step_head)

            # Compute context vector (weighted sum of encoder outputs)
            context_vec_head = torch.sum(
                enc_out * aw_step_head.unsqueeze(2), dim=1, keepdim=True)
            context_vec.append(context_vec_head)

        # Concatenate all convtext vectors and attention distributions
        context_vec = torch.cat(context_vec, dim=-1)
        aw_step = torch.stack(aw_step, dim=-1)

        context_vec = self.w_mha(context_vec)

        return context_vec, aw_step
