#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""RNN decoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class RNNDecoder(nn.Module):
    """RNN decoder.
    Args:
        embedding_dim (int): the dimension of input features
        rnn_type (string): lstm or gru or rnn
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        dropout (float): the probability to drop nodes
        parameter_init (float): the range of uniform distribution to
            initialize weight parameters (>= 0)
        use_cuda (bool, optional): if True, use GPUs
        batch_first (bool, optional): if True, batch-major computation will be
            performed
        residual (bool, optional):
        dense_residual (bool, optional):
    """

    def __init__(self,
                 embedding_dim,
                 rnn_type,
                 num_units,
                 num_layers,
                 dropout,
                 parameter_init,
                 use_cuda=False,
                 batch_first=False,
                 residual=False,
                 dense_residual=False):

        super(RNNDecoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.rnn_type = rnn_type
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.parameter_init = parameter_init
        self.use_cuda = use_cuda
        self.batch_first = batch_first
        self.residual = residual
        self.dense_residual = dense_residual

        self.rnns = []
        for i_layer in range(num_layers):
            if i_layer == 0:
                decoder_input_size = embedding_dim
            else:
                decoder_input_size = num_units

            if rnn_type == 'lstm':
                rnn_i = nn.LSTM(decoder_input_size,
                                hidden_size=num_units,
                                num_layers=1,
                                bias=True,
                                batch_first=batch_first,
                                dropout=dropout,
                                bidirectional=False)
            elif rnn_type == 'gru':
                rnn_i = nn.GRU(decoder_input_size,
                               hidden_size=num_units,
                               num_layers=1,
                               bias=True,
                               batch_first=batch_first,
                               dropout=dropout,
                               bidirectional=False)
            elif rnn_type == 'rnn':
                rnn_i = nn.RNN(decoder_input_size,
                               hidden_size=num_units,
                               num_layers=1,
                               bias=True,
                               batch_first=batch_first,
                               dropout=dropout,
                               bidirectional=False)
            else:
                raise ValueError('rnn_type must be "lstm" or "gru" or "rnn".')

            setattr(self, rnn_type + '_l' + str(i_layer), rnn_i)
            if use_cuda:
                rnn_i = rnn_i.cuda()
            self.rnns.append(rnn_i)

    def forward(self, y, decoder_state_init, volatile=False):
        """Forward computation.
        Args:
            y (FloatTensor): A tensor of size `[B, 1, embedding_dim]`
            decoder_state_init (FloatTensor): A tensor of size
                `[num_layers, B, num_units]`
            volatile (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            outputs: if batch_first is True, a tensor of size `[B, 1, num_units]`
                else `[1, B, num_units]`
            decoder_state (FloatTensor or tuple):
        """
        if not self.batch_first:
            # Reshape y to the time-major
            y = y.transpose(0, 1)

        outputs = y
        res_outputs_list = []
        # NOTE: exclude residual connection from decoder's inputs
        for i_layer in range(self.num_layers):
            if self.rnn_type == 'lstm':
                outputs, decoder_state = self.rnns[i_layer](
                    outputs, hx=decoder_state_init)
            else:
                outputs, decoder_state = self.rnns[i_layer](
                    outputs, hx=decoder_state_init)

            # Residual connection
            if self.residual or self.dense_residual:
                if self.residual or self.dense_residual:
                    for outputs_lower in res_outputs_list:
                        outputs = outputs + outputs_lower
                    if self.residual:
                        res_outputs_list = [outputs]
                    elif self.dense_residual:
                        res_outputs_list.append(outputs)

        return outputs, decoder_state
