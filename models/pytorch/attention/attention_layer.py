# ! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention layer (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.pytorch.linear import LinearND

ATTENTION_TYPE = ['content', 'location', 'dot_product', 'rnn_attention']


class AttentionMechanism(nn.Module):
    """Attention layer.
    Args:
        encoder_num_units (int): the number of units in each layer of the
            encoder
        decoder_num_units (int): the number of units in each layer of the
            decoder
        attention_type (string): the type of attention
        attention_dim: (int) the dimension of the attention layer
        sharpening_factor (float, optional): a sharpening factor in the softmax
            layer for computing attention weights
        sigmoid_smoothing (bool, optional): if True, replace softmax function
            in computing attention weights with sigmoid function for smoothing
        out_channels (int, optional): the number of channles of conv outputs.
            This is used for location-based attention.
        kernel_size (int, optional): the size of kernel.
            This must be the odd number.
    """

    def __init__(self,
                 encoder_num_units,
                 decoder_num_units,
                 attention_type,
                 attention_dim,
                 sharpening_factor=1,
                 sigmoid_smoothing=False,
                 out_channels=10,
                 kernel_size=201):

        super(AttentionMechanism, self).__init__()

        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.sharpening_factor = sharpening_factor
        self.sigmoid_smoothing = sigmoid_smoothing

        if self.attention_type == 'content':
            self.W_enc = LinearND(encoder_num_units,
                                  attention_dim, bias=True)
            self.W_dec = LinearND(decoder_num_units, attention_dim, bias=False)
            self.V = LinearND(attention_dim, 1, bias=False)

        elif self.attention_type == 'location':
            assert kernel_size % 2 == 1

            self.W_enc = LinearND(encoder_num_units,
                                  attention_dim, bias=True)
            self.W_dec = LinearND(decoder_num_units,
                                  attention_dim, bias=False)
            self.W_conv = LinearND(out_channels, attention_dim, bias=False)
            # self.conv = nn.Conv1d(in_channels=1,
            #                       out_channels=out_channels,
            #                       kernel_size=kernel_size,
            #                       stride=1,
            #                       padding=kernel_size // 2,
            #                       bias=False)
            self.conv = nn.Conv2d(in_channels=1,
                                  out_channels=out_channels,
                                  kernel_size=(1, kernel_size),
                                  stride=1,
                                  padding=(0, kernel_size // 2),
                                  bias=False)
            self.V = LinearND(attention_dim, 1, bias=False)

        elif self.attention_type == 'dot_product':
            self.W_keys = LinearND(encoder_num_units, attention_dim,
                                   bias=False)
            self.W_query = LinearND(decoder_num_units, attention_dim,
                                    bias=False)

        elif self.attention_type == 'rnn_attention':
            raise NotImplementedError

        else:
            raise TypeError(
                "attention_type should be one of [%s], you provided %s." %
                (", ".join(ATTENTION_TYPE), attention_type))

    def forward(self, enc_out, x_lens, dec_out, att_weights_step):
        """Forward computation.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            x_lens (torch.autograd.Variable, int): A tensor of size `[B]`
            dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, decoder_num_units]`
            att_weights_step (torch.autograd.Variable, float): A tensor of size
                `[B, T_in]`
        Returns:
            context_vec (torch.autograd.Variable, float): A tensor of size
                `[B, 1, encoder_num_units]`
            att_weights_step (torch.autograd.Variable, float): A tensor of size
                `[B, T_in]`
        """
        batch_size, max_time = enc_out.size()[:2]

        if self.attention_type == 'content':
            ###################################################################
            # energy = <v, tanh(W([h_de; h_en] + b))>
            ###################################################################
            dec_out = dec_out.expand_as(torch.zeros(
                (batch_size, max_time, dec_out.size(2))))
            energy = self.V(F.tanh(self.W_enc(enc_out) +
                                   self.W_dec(dec_out))).squeeze(2)

        elif self.attention_type == 'location':
            ###################################################################
            # f = F * α_{i-1}
            # energy = <v, tanh(W([h_de; h_en] + W_conv(f) + b))>
            ###################################################################
            # For 1D conv
            # conv_feat = self.conv(att_weights_step.unsqueeze(dim=1))
            # -> `[B, out_channels, T_in]`

            # For 2D conv
            conv_feat = self.conv(
                att_weights_step.view(batch_size, 1, 1, max_time)).squeeze(2)
            # -> `[B, out_channels, T_in]`
            conv_feat = conv_feat.transpose(1, 2).contiguous()
            # -> `[B, T_in, out_channels]`

            dec_out = dec_out.expand_as(torch.zeros(
                (batch_size, max_time, dec_out.size(2))))
            energy = self.V(F.tanh(self.W_enc(enc_out) +
                                   self.W_dec(dec_out) +
                                   self.W_conv(conv_feat))).squeeze(2)

        elif self.attention_type == 'dot_product':
            ###################################################################
            # energy = <W_keys(h_en), W_query(h_de)>
            ###################################################################
            keys = self.W_keys(enc_out)
            query = self.W_query(dec_out).transpose(1, 2)
            energy = torch.bmm(keys, query).squeeze(2)

        elif self.attention_type == 'rnn_attention':
            raise NotImplementedError

        else:
            raise NotImplementedError

        # Mask attention distribution
        energy_mask = Variable(torch.ones(batch_size, max_time))
        if enc_out.is_cuda:
            energy_mask = energy_mask.cuda()
        for b in range(batch_size):
            if x_lens[b].data[0] < max_time:
                energy_mask.data[b, x_lens[b].data[0]:] = 0
        energy *= energy_mask
        # NOTE: energy: `[B, T_in]`

        # Sharpening
        energy *= self.sharpening_factor

        # Compute attention weights
        if self.sigmoid_smoothing:
            att_weights_step = F.sigmoid(energy)
            # for b in range(batch_size):
            #     att_weights_step.data[b] /= att_weights_step.data[b].sum()
        else:
            att_weights_step = F.softmax(energy, dim=-1)

        # Compute context vector (weighted sum of encoder outputs)
        context_vec = torch.sum(
            enc_out * att_weights_step.unsqueeze(2), dim=1, keepdim=True)

        return context_vec, att_weights_step
