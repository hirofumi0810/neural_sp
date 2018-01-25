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
        rnn_type (string): lstm or gru or rnn
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        dropout (float): the probability to drop nodes
        batch_first (bool): if True, batch-major computation will be performed
        residual (bool, optional):
        dense_residual (bool, optional):
    """

    def __init__(self,
                 input_size,
                 rnn_type,
                 num_units,
                 num_layers,
                 dropout,
                 batch_first,
                 residual=False,
                 dense_residual=False):

        super(RNNDecoder, self).__init__()

        self.input_size = input_size
        self.rnn_type = rnn_type
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.residual = residual
        self.dense_residual = dense_residual

        for i_layer in range(num_layers):
            if i_layer == 0:
                decoder_input_size = input_size
            else:
                decoder_input_size = num_units

            if rnn_type == 'lstm':
                rnn_i = nn.LSTM(decoder_input_size,
                                hidden_size=num_units,
                                num_layers=1,
                                bias=True,
                                batch_first=batch_first,
                                dropout=0,
                                bidirectional=False)
            elif rnn_type == 'gru':
                rnn_i = nn.GRU(decoder_input_size,
                               hidden_size=num_units,
                               num_layers=1,
                               bias=True,
                               batch_first=batch_first,
                               dropout=0,
                               bidirectional=False)
            elif rnn_type == 'rnn':
                rnn_i = nn.RNN(decoder_input_size,
                               hidden_size=num_units,
                               num_layers=1,
                               bias=True,
                               batch_first=batch_first,
                               dropout=0,
                               bidirectional=False)
            else:
                raise ValueError('rnn_type must be "lstm" or "gru" or "rnn".')

            setattr(self, rnn_type + '_l' + str(i_layer), rnn_i)

            # Dropout for hidden-hidden or hidden-output connection
            setattr(self, 'dropout_l' + str(i_layer), nn.Dropout(p=dropout))

    def forward(self, y, dec_state, volatile=False):
        """Forward computation.
        Args:
            y (torch.autograd.Variable, float): A tensor of size
                `[B, 1, input_size]`
            dec_state (torch.autograd.Variable(float) or tuple): A tensor of size
                `[1, B, num_units]`
            volatile (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            dec_out (torch.autograd.Variable, float):
                if batch_first is True, a tensor of size `[B, 1, num_units]`
                else `[1, B, num_units]`
            dec_state (torch.autograd.Variable(float) or tuple):
        """
        if not self.batch_first:
            # Reshape y to the time-major
            y = y.transpose(0, 1)

        dec_out = y
        res_outputs_list = []
        # NOTE: exclude residual connection from decoder's inputs
        for i_layer in range(self.num_layers):
            dec_out, dec_state = getattr(self, self.rnn_type + '_l' + str(i_layer))(
                dec_out, hx=dec_state)

            # Dropout for hidden-hidden or hidden-output connection
            dec_out = getattr(self, 'dropout_l' + str(i_layer))(dec_out)

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
