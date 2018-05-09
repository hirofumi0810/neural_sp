#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""RNN decoders (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class RNNDecoder(nn.Module):
    """RNN decoder.
    Args:
        input_size (int): the dimension of decoder inputs
        rnn_type (string): lstm or gru
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        dropout (float): the probability to drop nodes
        residual (bool, optional):
        dense_residual (bool, optional):
    """

    def __init__(self,
                 input_size,
                 rnn_type,
                 num_units,
                 num_layers,
                 dropout,
                 residual=False,
                 dense_residual=False):

        super(RNNDecoder, self).__init__()

        self.input_size = input_size
        self.rnn_type = rnn_type
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.dense_residual = dense_residual

        for l in range(num_layers):
            decoder_input_size = input_size if l == 0 else num_units

            if rnn_type == 'lstm':
                rnn_i = nn.LSTMCell(input_size=decoder_input_size,
                                    hidden_size=num_units,
                                    bias=True)
            elif rnn_type == 'gru':
                rnn_i = nn.GRUCell(input_size=decoder_input_size,
                                   hidden_size=num_units,
                                   bias=True)
            else:
                raise ValueError('rnn_type must be "lstm" or "gru".')

            setattr(self, rnn_type + '_l' + str(l), rnn_i)

            # Dropout for hidden-hidden or hidden-output connection
            setattr(self, 'dropout_l' + str(l), nn.Dropout(p=dropout))

    def forward(self, dec_in, dec_state):
        """Forward computation.
        Args:
            dec_in (torch.FloatTensor): A tensor of size
                `[B, 1, embedding_dim + encoder_num_units]`
            dec_state (torch.FloatTensor or tuple):
        Returns:
            dec_out (torch.FloatTensor): A tensor of size `[B, 1, num_units]`
            dec_state (torch.FloatTensor or tuple):
        """
        if self.rnn_type == 'lstm':
            hx_list, cx_list = dec_state
        elif self.rnn_type == 'gru':
            hx_list = dec_state

        dec_in = dec_in.squeeze(1)
        # NOTE: exclude residual connection from decoder's inputs
        for l in range(self.num_layers):
            if self.rnn_type == 'lstm':
                if l == 0:
                    hx_list[0], cx_list[0] = getattr(self, 'lstm_l0')(
                        dec_in, (hx_list[0], cx_list[0]))
                else:
                    hx_list[l], cx_list[l] = getattr(self, 'lstm_l' + str(l))(
                        hx_list[l - 1], (hx_list[l], cx_list[l]))
            elif self.rnn_type == 'gru':
                if l == 0:
                    hx_list[0] = getattr(self, 'gru_l0')(dec_in, hx_list[0])
                else:
                    hx_list[l] = getattr(self, 'gru_l' + str(l))(
                        hx_list[l - 1], hx_list[l])

            # Dropout for hidden-hidden or hidden-output connection
            hx_list[l] = getattr(self, 'dropout_l' + str(l))(hx_list[l])

            # Residual connection
            if l > 0 and self.residual or self.dense_residual:
                if self.residual:
                    hx_list[l] += sum(hx_list[l - 1])
                elif self.dense_residual:
                    hx_list[l] += sum(hx_list[:l])

        dec_out = hx_list[-1].unsqueeze(1)

        if self.rnn_type == 'lstm':
            dec_state = (hx_list, cx_list)
        elif self.rnn_type == 'gru':
            dec_state = hx_list

        return dec_out, dec_state
