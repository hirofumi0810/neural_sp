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
                rnn_i = nn.LSTM(decoder_input_size,
                                hidden_size=num_units,
                                num_layers=1,
                                bias=True,
                                batch_first=True,
                                dropout=0,
                                bidirectional=False)
            elif rnn_type == 'gru':
                rnn_i = nn.GRU(decoder_input_size,
                               hidden_size=num_units,
                               num_layers=1,
                               bias=True,
                               batch_first=True,
                               dropout=0,
                               bidirectional=False)
            else:
                raise ValueError('rnn_type must be "lstm" or "gru".')

            setattr(self, rnn_type + '_l' + str(l), rnn_i)

            # Dropout for hidden-hidden or hidden-output connection
            setattr(self, 'dropout_l' + str(l), nn.Dropout(p=dropout))

    def forward(self, y, dec_state):
        """Forward computation.
        Args:
            y (torch.autograd.Variable, float): A tensor of size
                `[B, 1, input_size]`
            dec_state (torch.autograd.Variable(float) or tuple): A tensor of size
                `[1, B, num_units]`
        Returns:
            dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, num_units]`
            dec_state (torch.autograd.Variable(float) or tuple):
        """
        dec_out = y
        res_outputs_list = []
        # NOTE: exclude residual connection from decoder's inputs
        for l in range(self.num_layers):
            dec_out, dec_state = getattr(self, self.rnn_type + '_l' + str(l))(
                dec_out, hx=dec_state)

            # Dropout for hidden-hidden or hidden-output connection
            dec_out = getattr(self, 'dropout_l' + str(l))(dec_out)

            # Residual connection
            if self.residual or self.dense_residual:
                if self.residual or self.dense_residual:
                    for outputs_lower in res_outputs_list:
                        dec_out = dec_out + outputs_lower
                    if self.residual:
                        res_outputs_list = [dec_out]
                    elif self.dense_residual:
                        res_outputs_list.append(dec_out)

        return dec_out, dec_state
