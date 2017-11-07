#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""RNN encoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable


class RNNEncoder(nn.Module):
    """RNN encoder.
    Args:
        input_size (int): the dimension of input features
        rnn_type (string): lstm or gru or rnn
        bidirectional (bool): if True, use the bidirectional encoder
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
                 input_size,
                 rnn_type,
                 bidirectional,
                 num_units,
                 num_proj,
                 num_layers,
                 dropout,
                 parameter_init,
                 use_cuda=False,
                 batch_first=False):

        super(RNNEncoder, self).__init__()

        self.input_size = input_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
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
                input_size,
                hidden_size=num_units,
                num_layers=num_layers,
                bias=True,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size,
                hidden_size=num_units,
                num_layers=num_layers,
                bias=True,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(
                input_size,
                hidden_size=num_units,
                num_layers=num_layers,
                bias=True,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional)
        else:
            raise TypeError('rnn_type must be "lstm" or "gru" or "rnn".')

    def _init_hidden(self, batch_size, volatile):
        """Initialize hidden states.
        Args:
            batch_size (int): the size of mini-batch
            volatile (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            if rnn_type is 'lstm', return a tuple of tensors (h_0, c_0).
                h_0: A tensor of size
                    `[num_layers * num_directions, batch_size, num_units]`
                c_0: A tensor of size
                    `[num_layers * num_directions, batch_size, num_units]`
            otherwise return h_0.
        """
        h_0 = Variable(torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.num_units))

        if volatile:
            h_0.volatile = True

        if self.use_cuda:
            h_0 = h_0.cuda()

        if self.rnn_type == 'lstm':
            c_0 = Variable(torch.zeros(
                self.num_layers * self.num_directions, batch_size, self.num_units))

            if volatile:
                c_0.volatile = True

            if self.use_cuda:
                c_0 = c_0.cuda()

            return (h_0, c_0)
        else:
            # gru or rnn
            return h_0

    def forward(self, inputs, volatile=False):
        """Forward computation.
        Args:
            inputs: A tensor of size `[B, T, input_size]`
            volatile (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            outputs:
                if batch_first is True, a tensor of size
                    `[B, T, num_units * num_directions]`
                else
                    `[T, B, num_units * num_directions]`
            h_n: A tensor of size
                `[num_layers * num_directions, B, num_units]`
        """
        batch_size, max_time = inputs.size()[:2]

        # Initialize hidden states (and memory cells) per mini-batch
        h_0 = self._init_hidden(batch_size=batch_size, volatile=volatile)

        if not self.batch_first:
            # Reshape inputs to the time-major
            inputs = inputs.transpose(0, 1)

        if self.rnn_type == 'lstm':
            outputs, (h_n, c_n) = self.rnn(inputs, hx=h_0)
        else:
            # gru or rnn
            outputs, h_n = self.rnn(inputs, hx=h_0)

        # TODO: add the projection layer

        return outputs, h_n
