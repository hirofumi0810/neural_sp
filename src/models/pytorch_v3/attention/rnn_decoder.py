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
        residual (bool):
        dense_residual (bool):
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

        for i_l in range(num_layers):
            decoder_input_size = input_size if i_l == 0 else num_units

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

            setattr(self, rnn_type + '_l' + str(i_l), rnn_i)

            # Dropout for hidden-hidden or hidden-output connection
            setattr(self, 'dropout_l' + str(i_l), nn.Dropout(p=dropout))

    def forward(self, dec_in, hx_list, cx_list):
        """Forward computation.
        Args:
            dec_in (torch.autograd.Variable, float): A tensor of size
                `[B, 1, embedding_dim + encoder_num_units]`
            hx_list (list of torch.autograd.Variable(float)):
            cx_list (list of torch.autograd.Variable(float)):
        Returns:
            dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, num_units]`
            hx_list (list of torch.autograd.Variable(float)):
            cx_list (list of torch.autograd.Variable(float)):
        """
        dec_in = dec_in.squeeze(1)
        # NOTE: exclude residual connection from decoder's inputs
        for i_l in range(self.num_layers):
            if self.rnn_type == 'lstm':
                if i_l == 0:
                    hx_list[0], cx_list[0] = getattr(self, 'lstm_l0')(
                        dec_in, (hx_list[0], cx_list[0]))
                else:
                    hx_list[i_l], cx_list[i_l] = getattr(self, 'lstm_l' + str(i_l))(
                        hx_list[i_l - 1], (hx_list[i_l], cx_list[i_l]))
            elif self.rnn_type == 'gru':
                if i_l == 0:
                    hx_list[0] = getattr(self, 'gru_l0')(dec_in, hx_list[0])
                else:
                    hx_list[i_l] = getattr(self, 'gru_l' + str(i_l))(
                        hx_list[i_l - 1], hx_list[i_l])

            # Dropout for hidden-hidden or hidden-output connection
            hx_list[i_l] = getattr(self, 'dropout_l' + str(i_l))(hx_list[i_l])

            # Residual connection
            if i_l > 0 and self.residual or self.dense_residual:
                if self.residual:
                    hx_list[i_l] += sum(hx_list[i_l - 1])
                elif self.dense_residual:
                    hx_list[i_l] += sum(hx_list[:i_l])

        dec_out = hx_list[-1].unsqueeze(1)

        return dec_out, hx_list, cx_list
