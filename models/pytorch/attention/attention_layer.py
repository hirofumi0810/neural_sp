# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTENTION_TYPE = ['content', 'location', 'hybrid', 'MLP_dot',
                  'luong_dot', 'luong_general', 'luong_concat',
                  'baidu_attetion']


class AttentionMechanism(nn.Module):

    """Attention-besed RNN decoder.
    Args:
        encoder_num_units (int):
        decoder_num_units (int):
        attention_type (string):
        attention_dim (int):
        sharpening_factor (float, optional):
    """

    def __init__(self,
                 encoder_num_units,
                 decoder_num_units,
                 attention_type,
                 attention_dim,
                 sharpening_factor=1):

        super(AttentionMechanism, self).__init__()

        self.encoder_num_units = encoder_num_units
        self.decoder_num_units = decoder_num_units
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.sharpening_factor = sharpening_factor

        if encoder_num_units != decoder_num_units:
            raise NotImplementedError(
                'Add the bridge layer between the encoder and decoder.')

        if attention_type not in ATTENTION_TYPE:
            raise TypeError(
                "attention_type should be one of [%s], you provided %s." %
                (", ".join(ATTENTION_TYPE), attention_type))

        if self.attention_type == 'content':
            self.W_enc = nn.Linear(encoder_num_units, attention_dim)
            self.W_dec = nn.Linear(decoder_num_units, attention_dim)
            self.v_a = nn.Linear(attention_dim, 1)

        elif self.attention_type == 'MLP_dot':
            self.W_enc = nn.Linear(encoder_num_units, attention_dim)
            self.W_dec = nn.Linear(decoder_num_units, attention_dim)

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

        elif self.attention_type == 'luong_general':
            self.W_a = nn.Linear(decoder_num_units, decoder_num_units)

        elif self.attention_type == 'luong_concat':
            self.W_a = nn.Linear(decoder_num_units * 2, attention_dim)
            self.v_a = nn.Linear(attention_dim, 1)

        elif self.attention_type == 'baidu_attetion':
            raise NotImplementedError

    def forward(self, encoder_states, dec_state, att_weight_vec):
        """
        Args:
            encoder_states (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            dec_state (FloatTensor): A tensor of size
                `[B, 1, decoder_num_units]`
            att_weight_vec (FloatTensor): A tensor of size `[B, T_in]`
        Returns:
            content_vector (FloatTensor): A tensor of size
                `[B, 1, encoder_num_units]`
            att_weight_vec (FloatTensor): A tensor of size `[B, T_in]`
        """
        # TODO: Add the bridge layer

        if self.attention_type == 'content':
            # energy = dot(v_a, tanh(W_enc(hidden_enc) + W_dec(hidden_dec)))
            _enc = self.W_enc(encoder_states)  # `[B, T_in, att_dim]`
            _dec = self.W_dec(dec_state)  # `[B, 1, att_dim]`
            _dec = _dec.expand_as(_enc)  # `[B, T_in, att_dim]`
            energy = F.tanh(_enc + _dec)  # `[B, T_in, att_dim]`
            energy = self.v_a(energy)  # `[B, T_in, 1]`
            energy = energy.squeeze(dim=2)  # `[B, T_in]`

        elif self.attention_type == 'location':
            raise NotImplementedError

        elif self.attention_type == 'hybrid':
            # f = F * α_{i-1}
            # energy = dot(v_a, tanh(W_enc(hidden_enc) + W_dec(hidden_dec) + W_fil(f)))
            _enc = self.W_enc(encoder_states)  # `[B, T_in, att_dim]`
            _dec = self.W_dec(dec_state)  # `[B, 1, att_dim]`
            _dec = _dec.expand_as(_enc)  # `[B, T_in, att_dim]`
            _fil = self.fil(att_weight_vec.unsqueeze(dim=1))  # `[B, 1, 11]`
            print(_fil.size())
            _fil = self.W_fil(_fil)  # `[B, 1, att_dim]`
            _fil = _fil.expand_as(_enc)  # `[B, T_in, att_dim]`
            energy = F.tanh(_enc + _dec + _fil)  # `[B, T_in, att_dim]`
            energy = self.v_a(energy)  # `[B, T_in, 1]`
            energy = energy.squeeze(dim=2)  # `[B, T_in]`

        elif self.attention_type == 'MLP_dot':
            # energy = dot(W_enc(hidden_enc), W_dec(hidden_dec))
            _enc = self.W_enc(encoder_states)  # `[B, T_in, att_dim]`
            _dec = self.W_dec(dec_state)  # `[B, 1, att_dim]`
            _dec = _dec.transpose(1, 2)  # `[B, att_dim, 1]`
            energy = torch.bmm(_enc, _dec)  # `[B, T_in, 1]`
            energy = energy.squeeze(dim=2)  # `[B, T_in]`

        elif self.attention_type == 'luong_dot':
            # dot(hidden_enc, hidden_dec)
            # NOTE: both the encoder and decoder must be the same size
            _enc = encoder_states  # `[B, T_in, hidden_size]`
            _dec = dec_state.transpose(1, 2)  # `[B, 1, hidden_size]`
            energy = torch.bmm(_enc, _dec)  # `[B, T_in, 1]`
            energy = energy.squeeze(dim=2)  # `[B, T_in]`

        elif self.attention_type == 'luong_general':
            # energy = dot(W(hidden_enc), hidden_dec)
            _enc = self.W_a(encoder_states)  # `[B, T_in, hidden_size]`
            _dec = dec_state.transpose(1, 2)  # `[B, hidden_size, 1]`
            energy = torch.bmm(_enc, _dec)  # `[B, T_in, 1]`
            energy = energy.squeeze(dim=2)  # `[B, T_in]`

        elif self.attention_type == 'luong_concat':
            # energy = dot(v_a, tanh(W_a([hidden_dec;hidden_enc])))
            # NOTE: both the encoder and decoder must be the same size
            _enc = encoder_states  # `[B, T_in, hidden_size]`
            _dec = dec_state.expand_as(_enc)  # `[B, T_in, hidden_size]`
            # `[B, T_in, hidden_size * 2]`
            concat = torch.cat((_enc, _dec), dim=2)
            concat = F.tanh(self.W_a(concat))  # `[B, T_in, att_dim]`
            energy = self.v_a(concat)  # `[B, T_in, 1]`
            energy = energy.squeeze(dim=2)  # `[B, T_in]`

        else:
            raise TypeError

        # if att_weight_vec is not None:
        #     att_weight_vec = att_weight_vec.unsqueeze(dim=1)
        #     att_weight_vec = self.conv(
        #         att_weight_vec).squeeze(dim=1)
        #     pax = pax + att_weight_vec

        # Compute attention weights
        att_weight_vec = F.softmax(energy * self.sharpening_factor)

        # Compute context vector (weighted sum of encoder outputs)
        context_vector = torch.sum(
            encoder_states * att_weight_vec.unsqueeze(dim=2),
            dim=1, keepdim=True)

        return context_vector, att_weight_vec
