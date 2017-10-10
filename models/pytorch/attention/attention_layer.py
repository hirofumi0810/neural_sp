# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMechanism(nn.Module):

    """Attention-besed RNN decoder.
    Args:
        encoder_num_units (int)
        attention_type (string): content or location or hybrid
        rnn_type (string): lstm or gru or rnn
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        embedding_dim (int): the dimension of the output embedding layer
        num_classes (int): the number of classes of target labels.
            If 0, return hidden states before passing through the softmax layer
        parameter_init (float): Range of uniform distribution to initialize
            weight parameters
        att_softmax_temperature (float, optional):
    """

    def __init__(self,
                 encoder_num_units,
                 decoder_num_units,
                 attention_type,
                 attention_dim,
                 att_softmax_temperature=1):
                #  use_cuda=False):

        super(AttentionMechanism, self).__init__()

        self.encoder_num_units = encoder_num_units
        self.decoder_num_units = decoder_num_units
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.att_softmax_temperature = att_softmax_temperature

        if encoder_num_units != decoder_num_units:
            raise NotImplementedError(
                'Add the bridge layer between the encoder and decoder.')

        if self.attention_type == 'content':
            self.W_encoder = nn.Linear(encoder_num_units, attention_dim)
            self.W_decoder = nn.Linear(decoder_num_units, attention_dim)
            self.v_a = nn.Parameter(torch.FloatTensor(1, attention_dim))
            # self.v_a = nn.Parameter(torch.FloatTensor(attention_dim))

        elif self.attention_type == 'dot_product':
            raise NotImplementedError

        elif self.attention_type == 'location':
            raise NotImplementedError

        elif self.attention_type == 'hybrid':
            # self.filter = torch.nn.Conv1d(
            #     in_channels,
            #     out_channels,
            #     kernel_size,
            #     stride=1, padding=0, dilation=1, groups=1, bias=True)
            raise NotImplementedError

        elif self.attention_type == 'luong_general':
            self.W_a = nn.Linear(decoder_num_units, decoder_num_units)

        elif self.attention_type == 'luong_concat':
            self.W_a = nn.Linear(decoder_num_units * 2, attention_dim)
            self.v_a = nn.Parameter(torch.FloatTensor(1, attention_dim))

    def forward(self, encoder_outputs, dec_output, att_weight_vec):
        """
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            dec_output (FloatTensor): A tensor of size
                `[B, 1, decoder_num_units]`
            att_weight_vec (FloatTensor): A tensor of size `[B, T_in]`
        Returns:
            content_vector (FloatTensor): A tensor of size
                `[B, 1, encoder_num_units]`
            att_weight_vec (FloatTensor): A tensor of size `[B, T_in]`
        """
        # TODO: Add the bridge layer

        ###################################
        # ↓ bahdanau's implementation
        ###################################
        if self.attention_type == 'content':
            energy = self.W_encoder(encoder_outputs)
            energy += self.W_decoder(dec_output).expand_as(energy)
            energy = torch.mean(self.v_a * energy, dim=2)

        elif self.attention_type == 'location':
            raise NotImplementedError

        elif self.attention_type == 'hybrid':
            raise NotImplementedError

        ###################################
        # ↓ Luong's impementation
        ###################################
        elif self.attention_type == 'luong_dot':
            energy = torch.bmm(dec_output,
                               encoder_outputs.transpose(1, 2)).squeeze(dim=1)

        elif self.attention_type == 'luong_general':
            energy = self.W_a(encoder_outputs).transpose(1, 2)
            energy = torch.bmm(dec_output, energy).squeeze(dim=1)

        elif self.attention_type == 'luong_concat':
            raise NotImplementedError

        else:
            raise TypeError
        # NOTE: energy: `[B, T_in]`

        # if att_weight_vec is not None:
        #     att_weight_vec = att_weight_vec.unsqueeze(dim=1)
        #     att_weight_vec = self.conv(
        #         att_weight_vec).squeeze(dim=1)
        #     pax = pax + att_weight_vec

        # Compute attention weights (including smoothing)
        att_weight_vec = F.softmax(
            energy / self.att_softmax_temperature)

        # Compute context vector (weighted sum of encoder outputs)
        context_vector = torch.sum(
            encoder_outputs * att_weight_vec.unsqueeze(dim=2),
            dim=1, keepdim=True)

        return context_vector, att_weight_vec
