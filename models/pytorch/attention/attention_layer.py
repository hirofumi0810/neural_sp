# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTENTION_TYPE = [
    'bahdanau_content', 'normed_bahdanau_content',
    'location', 'hybrid', 'dot_product',
    'luong_dot', 'scaled_luong_dot', 'luong_general', 'luong_concat',
    'baidu_attetion']


class AttentionMechanism(nn.Module):
    """Attention-besed RNN decoder.
    Args:
        encoder_num_units (int): the number of units in each layer of the
            encoder
        decoder_num_units (int): the number of units in each layer of the
            decoder
        attention_type (string): the type of attention
        attention_dim: (int) the dimension of the attention layer
        sharpening_factor (float, optional): a sharpening factor in the
            softmax layer for computing attention weights
        sigmoid_smoothing (bool, optional): if True, replace softmax function
            in computing attention weights with sigmoid function for smoothing
    """

    def __init__(self,
                 encoder_num_units,
                 decoder_num_units,
                 attention_type,
                 attention_dim,
                 sharpening_factor=1,
                 sigmoid_smoothing=False):

        super(AttentionMechanism, self).__init__()

        self.encoder_num_units = encoder_num_units
        self.decoder_num_units = decoder_num_units
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.sharpening_factor = sharpening_factor
        self.sigmoid_smoothing = sigmoid_smoothing

        if encoder_num_units != decoder_num_units:
            raise NotImplementedError(
                'Add the bridge layer between the encoder and decoder.')

        if attention_type not in ATTENTION_TYPE:
            raise TypeError(
                "attention_type should be one of [%s], you provided %s." %
                (", ".join(ATTENTION_TYPE), attention_type))

        if self.attention_type == 'bahdanau_content':
            self.W_enc = nn.Linear(encoder_num_units, attention_dim)
            self.W_dec = nn.Linear(decoder_num_units, attention_dim)
            self.v_a = nn.Linear(attention_dim, 1)

        elif self.attention_type == 'normed_bahdanau_content':
            raise NotImplementedError

        elif self.attention_type == 'location':
            raise NotImplementedError

        elif self.attention_type == 'hybrid':
            self.W_enc = nn.Linear(encoder_num_units, attention_dim)
            self.W_dec = nn.Linear(decoder_num_units, attention_dim)
            self.W_fil = nn.Linear(100, attention_dim)
            self.v_a = nn.Linear(attention_dim, 1)
            self.fil = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=100,
                stride=1,
                padding=0,
                dilation=1, groups=1, bias=True)
            # TODO: make filter size parameter

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

        elif self.attention_type == 'baidu_attetion':
            raise NotImplementedError

    def forward(self, encoder_states, decoder_state, att_weight_vec):
        """
        Args:
            encoder_states (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            decoder_state (FloatTensor): A tensor of size
                `[1, B, decoder_num_units]`
            att_weight_vec (FloatTensor): A tensor of size `[B, T_in]`
        Returns:
            content_vector (FloatTensor): A tensor of size
                `[B, 1, encoder_num_units]`
            att_weight_vec (FloatTensor): A tensor of size `[B, T_in]`
        """
        # TODO: Add the bridge layer between encoder and decoder_dropout

        if self.attention_type == 'bahdanau_content':
            ###################################################################
            # energy = <v_a, tanh(W_enc(hidden_enc) + W_dec(hidden_dec))>
            ###################################################################
            # `[B, T_in, att_dim]`
            keys = self.W_enc(encoder_states)
            # `[1, B, decoder_num_units]` -> `[B, T_in, att_dim]`
            query = self.W_dec(decoder_state).transpose(0, 1).expand_as(keys)
            energy = self.v_a(F.tanh(keys + query)).squeeze(dim=2)

        elif self.attention_type == 'normed_bahdanau_content':
            raise NotImplementedError

        elif self.attention_type == 'location':
            raise NotImplementedError

        elif self.attention_type == 'hybrid':
            # f = F * α_{i-1}
            # energy = dot(v_a, tanh(W_enc(hidden_enc) + W_dec(hidden_dec) + W_fil(f)))
            keys = self.W_enc(encoder_states)  # `[B, T_in, att_dim]`
            query = self.W_dec(decoder_state).transpose(
                0, 1)  # `[B, 1, att_dim]`
            query = query.expand_as(keys)  # `[B, T_in, att_dim]`
            _fil = self.fil(att_weight_vec.unsqueeze(dim=1))  # `[B, 1, 11]`
            _fil = self.W_fil(_fil)  # `[B, 1, att_dim]`
            _fil = _fil.expand_as(keys)  # `[B, T_in, att_dim]`
            energy = F.tanh(keys + query + _fil)  # `[B, T_in, att_dim]`
            energy = self.v_a(energy)  # `[B, T_in, 1]`
            energy = energy.squeeze(dim=2)  # `[B, T_in]`

        elif self.attention_type == 'dot_product':
            ###################################################################
            # energy = <W_enc(hidden_enc), W_dec(hidden_dec)>
            ###################################################################
            # `[B, T_in, att_dim]`
            keys = self.W_enc(encoder_states)
            # `[1, B, decoder_num_units]` -> `[B, 1, att_dim]`
            query = self.W_dec(decoder_state).transpose(0, 1)
            # `[B, 1, att_dim]` -> `[B, att_dim, 1]`
            query = query.transpose(1, 2)
            energy = torch.bmm(keys, query).squeeze(dim=2)

        elif self.attention_type == 'luong_dot':
            ###################################################################
            # energy = <hidden_enc, hidden_dec>
            # NOTE: both the encoder and decoder must be the same size
            ###################################################################
            # `[B, T_in, encoder_num_units]`
            keys = encoder_states
            # `[1, B, decoder_num_units]` -> `[B, 1, decoder_num_units]`
            query = decoder_state.transpose(0, 1)
            # `[B, 1, decoder_num_units]` -> `[B, decoder_num_units, 1]`
            query = query.transpose(1, 2)
            energy = torch.bmm(keys, query).squeeze(dim=2)

        elif self.attention_type == 'scaled_luong_dot':
            raise NotImplementedError

        elif self.attention_type == 'luong_general':
            ###################################################################
            # energy = <W(hidden_enc), hidden_dec>
            ###################################################################
            # `[B, T_in, encoder_num_units]` -> `[B, T_in, decoder_num_units]`
            keys = self.W_a(encoder_states)
            # `[1, B, decoder_num_units]` -> `[B, 1, decoder_num_units]`
            query = decoder_state.transpose(0, 1)
            # `[B, 1, decoder_num_units]` -> `[B, decoder_num_units, 1]`
            query = query.transpose(1, 2)
            energy = torch.bmm(keys, query).squeeze(dim=2)

        elif self.attention_type == 'luong_concat':
            ###################################################################
            # energy = <v_a, tanh(W_a([hidden_dec;hidden_enc]))>
            # NOTE: both the encoder and decoder must be the same size
            ###################################################################
            # `[B, T_in, encoder_num_units]`
            keys = encoder_states
            # `[1, B, decoder_num_units]` -> `[B, T_in, decoder_num_units]`
            query = decoder_state.transpose(0, 1).expand_as(keys)
            # `[B, T_in, decoder_num_units * 2]`
            concat = torch.cat((keys, query), dim=2)
            energy = self.v_a(F.tanh(self.W_a(concat))).squeeze(dim=2)

        else:
            raise NotImplementedError

        # if att_weight_vec is not None:
        #     att_weight_vec = att_weight_vec.unsqueeze(dim=1)
        #     att_weight_vec = self.conv(
        #         att_weight_vec).squeeze(dim=1)
        #     pax = pax + att_weight_vec

        # Compute attention weights
        if self.sigmoid_smoothing:
            raise NotImplementedError
        else:
            att_weight_vec = F.softmax(energy * self.sharpening_factor)

        # Compute context vector (weighted sum of encoder outputs)
        context_vector = torch.sum(
            encoder_states * att_weight_vec.unsqueeze(dim=2),
            dim=1, keepdim=True)

        return context_vector, att_weight_vec
