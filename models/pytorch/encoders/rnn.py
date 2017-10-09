#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn


class RNNEncoder(nn.Module):
    """RNN encoder.
    Args:
        input_size (int): the dimension of input features
        rnn_type (string): lstm or gru or rnn
        bidirectional (bool): if True, use the bidirectional encoder
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        dropout (float): the probability to drop nodes
        parameter_init (float): Range of uniform distribution to initialize
            weight parameters
        num_classes (int): the number of classes of target labels.
            If 0, return hidden states before passing through the softmax layer
        use_cuda (bool, optional):
        batch_first (bool, optional):

        num_proj (int): the number of nodes in recurrent projection layer
        bottleneck_dim (int): the dimensions of the bottleneck layer
    """

    def __init__(self,
                 input_size,
                 rnn_type,
                 bidirectional,
                 num_units,
                 num_layers,
                 dropout,
                 parameter_init,
                 num_classes=0,
                 use_cuda=False,
                 batch_first=False):
                #  num_proj,
                #  bottleneck_dim):

        super(RNNEncoder, self).__init__()

        self.input_size = input_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_units = num_units
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        # NOTE: dropout is applied except the last layer
        self.parameter_init = parameter_init
        self.use_cuda = use_cuda
        self.batch_first = batch_first

        self.return_hidden_states = True if num_classes == 0 else False

        # self.num_proj = num_proj
        # self.bottleneck_dim = bottleneck_dim

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size,
                               hidden_size=num_units,
                               num_layers=num_layers,
                               bias=True,
                               batch_first=batch_first,
                               dropout=dropout,
                               bidirectional=bidirectional)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size,
                              hidden_size=num_units,
                              num_layers=num_layers,
                              bias=True,
                              batch_first=batch_first,
                              dropout=dropout,
                              bidirectional=bidirectional)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size,
                              hidden_size=num_units,
                              num_layers=num_layers,
                              bias=True,
                              batch_first=batch_first,
                              dropout=dropout,
                              bidirectional=bidirectional)
        else:
            raise ValueError('rnn_type must be "lstm" or "gru" or "rnn".')

        # TODO: Add bottleneck layer

        if not self.return_hidden_states:
            self.fc = nn.Linear(num_units * self.num_directions, num_classes)

    def _init_hidden(self, batch_size):
        """Initialize hidden states.
        Args:
            batch_size (int): the size of mini-batch
        Returns:
            if rnn_type is 'lstm', return a tuple of tensors (h_0, c_0),
            otherwise return a tensor h_0.
        """
        h_0 = Variable(torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.num_units))

        if self.use_cuda:
            h_0 = h_0.cuda()

        if self.rnn_type == 'lstm':
            c_0 = Variable(torch.zeros(
                self.num_layers * self.num_directions, batch_size, self.num_units))

            if self.use_cuda:
                c_0 = c_0.cuda()

            return (h_0, c_0)
        else:
            # gru or rnn
            return h_0

    def forward(self, inputs):
        """Forward computation.
        Args:
            inputs: A tensor of size `[B, T, input_size]`
        Returns:
            if batch_first is True,
                logits: A tensor of size `[T, B, num_classes]`
                final_state: A tensor of size
                    `[num_layers * num_directions, B, num_units]`
            else
                logits: A tensor of size `[B, T, num_classes]`
                final_state: A tensor of size
                    `[B, num_layers * num_directions, num_units]`
        """
        batch_size, max_time = inputs.size()[:2]

        # Initialize hidden states (and memory cells) per mini-batch
        h_0 = self._init_hidden(batch_size=batch_size)

        if not self.batch_first:
            # Reshape to the time-major
            inputs = inputs.transpose(0, 1)

        if self.rnn_type == 'lstm':
            outputs, (h_n, c_n) = self.rnn(inputs, hx=h_0)
        else:  # gru or rnn
            outputs, h_n = self.rnn(inputs, hx=h_0)
        # NOTE: outputs:
        # `[B, T, num_units * num_directions]` (batch_first is True)
        # `[T, B, num_units * num_directions]` (batch_first is False)

        if self.return_hidden_states:
            return outputs, h_n
        # NOTE: this is for seq2seq models

        # Reshape to 2D tensor
        outputs = outputs.view(max_time * batch_size, -1)

        # Pass through the output layer
        logits = self.fc(outputs)

        # Reshape back to 3D tensor
        if self.batch_first:
            logits = logits.view(batch_size, max_time, -1)
        else:
            logits = logits.view(max_time, batch_size, -1)

        return logits, h_n
