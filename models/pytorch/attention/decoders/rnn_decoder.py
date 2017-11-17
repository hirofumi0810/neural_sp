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
        num_proj (int): the number of nodes in the projection layer
        num_layers (int): the number of layers
        dropout (float): the probability to drop nodes
        parameter_init (float): the range of uniform distribution to
            initialize weight parameters (>= 0)
        use_cuda (bool, optional): if True, use GPUs
        batch_first (bool, optional): if True, batch-major computation will be
            performed
    """

    def __init__(self,
                 embedding_dim,
                 rnn_type,
                 num_units,
                 num_proj,
                 num_layers,
                 dropout,
                 parameter_init,
                 use_cuda=False,
                 batch_first=False):

        super(RNNDecoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.rnn_type = rnn_type
        self.num_units = num_units
        self.num_proj = num_proj
        self.num_layers = num_layers
        self.dropout = dropout
        # NOTE: dropout is applied except the last layer
        self.parameter_init = parameter_init
        self.use_cuda = use_cuda
        self.batch_first = batch_first

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_size=num_units,
                num_layers=num_layers,
                bias=True,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=False)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_size=num_units,
                num_layers=num_layers,
                bias=True,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=False)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(
                embedding_dim,
                hidden_size=num_units,
                num_layers=num_layers,
                bias=True,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=False)
        else:
            raise TypeError('rnn_type must be "lstm" or "gru" or "rnn".')

    def forward(self, y, decoder_state, volatile=False):
        """Forward computation.
        Args:
            y (FloatTensor): A tensor of size `[B, 1, embedding_dim]`
            decoder_state (FloatTensor): A tensor of size
                `[num_layers, B, num_units]`
            volatile (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            outputs:
                if batch_first is True, a tensor of size
                    `[B, 1, num_units]`
                else
                    `[1, B, num_units]`
            decoder_state:
        """
        if not self.batch_first:
            # Reshape y to the time-major
            y = y.transpose(0, 1)

        outputs, decoder_state = self.rnn(y, hx=decoder_state)

        # TODO: add the projection layer

        return outputs, decoder_state
