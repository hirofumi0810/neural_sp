#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Multi-task RNN encodrs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable


class MultitaskRNNEncoder(nn.Module):
    """Multi-task RNN encoder.
    Args:
        input_size (int): the dimension of input features
        rnn_type (string): lstm or gru or rnn
        bidirectional (bool): if True, use the bidirectional encoder
        num_units (int): the number of units in each layer
        # num_proj (int): the number of nodes in recurrent projection layer
        num_layers_main (int): the number of layers in the main task
        num_layers_sub (int): the number of layers in the sub task
        dropout (float): the probability to drop nodes
        parameter_init (float): Range of uniform distribution to initialize
            weight parameters
        use_cuda (bool, optional):
        batch_first (bool, optional):
    """

    def __init__(self,
                 input_size,
                 rnn_type,
                 bidirectional,
                 num_units,
                 #  num_proj,
                 num_layers_main,
                 num_layers_sub,
                 dropout,
                 parameter_init,
                 use_cuda=False,
                 batch_first=False):

        super(MultitaskRNNEncoder, self).__init__()

        self.input_size = input_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_units = num_units
        # self.num_proj = num_proj
        self.num_layers_main = num_layers_main
        self.num_layers_sub = num_layers_sub
        self.dropout = dropout
        # NOTE: dropout is applied except the last layer

        self.parameter_init = parameter_init
        self.use_cuda = use_cuda
        self.batch_first = batch_first

        if num_layers_sub < 1 or num_layers_main < self.num_layers_sub:
            raise ValueError(
                'Set num_layers_sub between 1 to num_layers_main.')

        self.rnns = []
        for i in range(num_layers_main):
            if rnn_type == 'lstm':
                rnn = nn.LSTM(
                    input_size if i == 0 else num_units * self.num_directions,
                    hidden_size=num_units,
                    num_layers=1,
                    bias=True,
                    batch_first=batch_first,
                    dropout=dropout,
                    bidirectional=bidirectional)
            elif rnn_type == 'gru':
                rnn = nn.GRU(
                    input_size if i == 0 else num_units * self.num_directions,
                    hidden_size=num_units,
                    num_layers=1,
                    bias=True,
                    batch_first=batch_first,
                    dropout=dropout,
                    bidirectional=bidirectional)
            elif rnn_type == 'rnn':
                rnn = nn.RNN(
                    input_size if i == 0 else num_units * self.num_directions,
                    hidden_size=num_units,
                    num_layers=1,
                    bias=True,
                    batch_first=batch_first,
                    dropout=dropout,
                    bidirectional=bidirectional)
            else:
                raise ValueError('rnn_type must be "lstm" or "gru" or "rnn".')

            if use_cuda:
                rnn = rnn.cuda()

            self.rnns.append(rnn)

    def _init_hidden(self, batch_size):
        """Initialize hidden states.
        Args:
            batch_size (int): the size of mini-batch
        Returns:
            if rnn_type is 'lstm', return a tuple of tensors (h_0, c_0),
            otherwise return a tensor h_0.
        """
        h_0 = Variable(torch.zeros(
            1 * self.num_directions, batch_size, self.num_units))

        if self.use_cuda:
            h_0 = h_0.cuda()

        if self.rnn_type == 'lstm':
            c_0 = Variable(torch.zeros(
                1 * self.num_directions, batch_size, self.num_units))

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
                outputs: A tensor of size `[T, B, num_units * num_directions]`
                h_n: A tensor of size
                    `[num_layers * num_directions, B, num_units]`
                outputs_sub (): A tensor of size `[]`
                h_n_sub (): A tensor of size `[]`
            else
                outputs: A tensor of size `[B, T, num_units * num_directions]`
                h_n: A tensor of size
                    `[B, num_layers * num_directions, num_units]`
                outputs_sub (): A tensor of size `[]`
                h_n_sub (): A tensor of size `[]`
        """
        batch_size, max_time = inputs.size()[:2]

        # Initialize hidden states (and memory cells) per mini-batch
        h_0 = self._init_hidden(batch_size=batch_size)

        if not self.batch_first:
            # Reshape to the time-major
            inputs = inputs.transpose(0, 1)

        outputs = inputs
        final_state_list = []
        for i in range(self.num_layers_main):
            if self.rnn_type == 'lstm':
                outputs, (h_n, c_n) = self.rnns[i](outputs, hx=h_0)
            else:
                # gru or rnn
                outputs, h_n = self.rnns[i](outputs, hx=h_0)

            final_state_list.append(h_n)
            # NOTE: outputs:
            # `[B, T, num_units * num_directions]` (batch_first: True)
            # `[T, B, num_units * num_directions]` (batch_first: False)

            if i == self.num_layers_sub - 1:
                outputs_sub = outputs
                h_n_sub = torch.cat(final_state_list, dim=0)
                # `[B, num_layers_sub * num_directions, num_units]`

        h_n = torch.cat(final_state_list, dim=0)

        return outputs, h_n, outputs_sub, h_n_sub
