# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTENTION_TYPE = [
    'content', 'normed_content', 'location', 'dot_product',
    'luong_dot', 'scaled_luong_dot', 'luong_general', 'luong_concat',
    'rnn_attention']


class AttentionMechanism(nn.Module):
    """Attention layer.
    Args:
        encoder_num_units (int): the number of units in each layer of the
            encoder
        decoder_num_units (int): the number of units in each layer of the
            decoder
        attention_type (string): the type of attention
        attention_dim: (int) the dimension of the attention layer
        sharpening_factor (float): a sharpening factor in the softmax layer for
            computing attention weights
        sigmoid_smoothing (bool): if True, replace softmax function in
            computing attention weights with sigmoid function for smoothing
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
                 sharpening_factor,
                 sigmoid_smoothing,
                 out_channels=10,
                 kernel_size=101):

        super(AttentionMechanism, self).__init__()

        if attention_type not in ATTENTION_TYPE:
            raise TypeError(
                "attention_type should be one of [%s], you provided %s." %
                (", ".join(ATTENTION_TYPE), attention_type))

        self.encoder_num_units = encoder_num_units
        self.decoder_num_units = decoder_num_units
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.sharpening_factor = sharpening_factor
        self.sigmoid_smoothing = sigmoid_smoothing

        if self.attention_type == 'content':
            self.W_enc = nn.Linear(encoder_num_units, attention_dim)
            self.W_dec = nn.Linear(decoder_num_units, attention_dim)
            self.v_a = nn.Linear(attention_dim, 1)

        elif self.attention_type == 'normed_content':
            raise NotImplementedError

        elif self.attention_type == 'location':
            assert kernel_size % 2 == 1
            self.W_enc = nn.Linear(encoder_num_units, attention_dim)
            self.W_dec = nn.Linear(decoder_num_units, attention_dim)
            self.conv = nn.Conv1d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=True)
            self.W_conv = nn.Linear(out_channels, attention_dim)
            self.v_a = nn.Linear(attention_dim, 1)

        elif self.attention_type == 'dot_product':
            self.W_enc = nn.Linear(encoder_num_units, attention_dim)
            self.W_dec = nn.Linear(decoder_num_units, attention_dim)

        elif self.attention_type == 'luong_dot':
            # NOTE: no parameter
            pass

        elif self.attention_type == 'scaled_luong_dot':
            raise NotImplementedError

        elif self.attention_type == 'luong_general':
            self.W_a = nn.Linear(decoder_num_units, decoder_num_units)

        elif self.attention_type == 'luong_concat':
            self.W_a = nn.Linear(decoder_num_units * 2, attention_dim)
            self.v_a = nn.Linear(attention_dim, 1)

        elif self.attention_type == 'rnn_attention':
            raise NotImplementedError

    def forward(self, encoder_outputs, decoder_outputs, attention_weights_step):
        """
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            decoder_outputs (FloatTensor): A tensor of size
                `[B, 1, decoder_num_units]`
            attention_weights_step (FloatTensor): A tensor of size `[B, T_in]`
        Returns:
            context_vector (FloatTensor): A tensor of size
                `[B, 1, encoder_num_units]`
            attention_weights_step (FloatTensor): A tensor of size `[B, T_in]`
        """
        if self.attention_type == 'content':
            ###################################################################
            # energy = <v_a, tanh(W_enc(h_en) + W_dec(h_de))>
            ###################################################################
            keys = self.W_enc(encoder_outputs)
            query = self.W_dec(decoder_outputs).expand_as(keys)
            energy = self.v_a(F.tanh(keys + query)).squeeze(dim=2)

        elif self.attention_type == 'normed_content':
            raise NotImplementedError

        elif self.attention_type == 'location':
            ###################################################################
            # f = F * α_{i-1}
            # energy = <v_a,
            # tanh(W_enc(h_en) + W_dec(h_de) + W_conv(f))>
            ###################################################################
            keys = self.W_enc(encoder_outputs)
            query = self.W_dec(decoder_outputs).expand_as(keys)
            conv_feat = self.conv(attention_weights_step.unsqueeze(dim=1))
            conv_feat = self.W_conv(conv_feat.transpose(1, 2))
            energy = self.v_a(F.tanh(keys + query + conv_feat)).squeeze(dim=2)

        elif self.attention_type == 'dot_product':
            ###################################################################
            # energy = <W_enc(h_en), W_dec(h_de)>
            ###################################################################
            keys = self.W_enc(encoder_outputs)
            query = self.W_dec(decoder_outputs).transpose(1, 2)
            energy = torch.bmm(keys, query).squeeze(dim=2)

        elif self.attention_type == 'luong_dot':
            ###################################################################
            # energy = <h_en, h_de>
            # NOTE: both the encoder and decoder must be the same size
            ###################################################################
            keys = encoder_outputs
            query = decoder_outputs.transpose(1, 2)
            energy = torch.bmm(keys, query).squeeze(dim=2)

        elif self.attention_type == 'scaled_luong_dot':
            raise NotImplementedError

        elif self.attention_type == 'luong_general':
            ###################################################################
            # energy = <W(h_en), h_de>
            ###################################################################
            keys = self.W_a(encoder_outputs)
            query = decoder_outputs.transpose(1, 2)
            energy = torch.bmm(keys, query).squeeze(dim=2)

        elif self.attention_type == 'luong_concat':
            ###################################################################
            # energy = <v_a, tanh(W_a([h_de; h_en]))>
            # NOTE: both the encoder and decoder must be the same size
            ###################################################################
            keys = encoder_outputs
            query = decoder_outputs.expand_as(keys)
            concat = torch.cat((keys, query), dim=2)
            energy = self.v_a(F.tanh(self.W_a(concat))).squeeze(dim=2)

        else:
            raise NotImplementedError

        # Sharpening
        energy *= self.sharpening_factor

        # Compute attention weights
        if self.sigmoid_smoothing:
            attention_weights_step = F.sigmoid(energy)
        else:
            attention_weights_step = F.softmax(energy, dim=energy.dim() - 1)

        # Compute context vector (weighted sum of encoder outputs)
        context_vector = torch.sum(
            encoder_outputs * attention_weights_step.unsqueeze(dim=2),
            dim=1, keepdim=True)

        return context_vector, attention_weights_step
