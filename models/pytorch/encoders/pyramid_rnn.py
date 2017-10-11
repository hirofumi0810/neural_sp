#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Pyramid RNN encoders.
    This implementation is bases on
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable


class PyramidRNNEnocer(nn.Module):
    """Pyramid RNN encoder.
    Args:

    """

    def __init__(self,
                 input_size,
                 rnn_type,
                 bidirectional,
                 num_units,
                 #  num_proj,
                 num_layers,
                 downsample_list,
                 dropout,
                 parameter_init,
                 use_cuda=False,
                 batch_first=False):

        super(PyramidRNNEnocer, self).__init__()

        if len(downsample_list) != num_layers:
            raise ValueError

        self.input_size = input_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_units = num_units
        # self.num_proj = num_proj
        self.num_layers = num_layers
        self.downsample_list = downsample_list
        self.dropout = dropout
        # NOTE: dropout is applied except the last layer

        self.parameter_init = parameter_init
        self.use_cuda = use_cuda
        self.batch_first = batch_first

        self.rnns = []
        for i in range(num_layers):
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
                raise ValueError(
                    'rnn_type must be "lstm" or "gru" or "rnn".')

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

        """
        batch_size, max_time = inputs.size()[:2]

        # Initialize hidden states (and memory cells) per mini-batch
        h_0 = self._init_hidden(batch_size=batch_size)

        if not self.batch_first:
            # Reshape to the time-major
            inputs = inputs.transpose(0, 1)

        # print(inputs.size())

        outputs = inputs
        final_state_list = []
        for i in range(self.num_layers):
            # self.rnns[i].flatten_parameters()
            if self.rnn_type == 'lstm':
                outputs, (h_n, c_n) = self.rnns[i](outputs, hx=h_0)
            else:
                # gru or rnn
                outputs, h_n = self.rnns[i](outputs, hx=h_0)

            final_state_list.append(h_n)
            # NOTE: outputs:
            # `[B, T, num_units * num_directions]` (batch_first: True)
            # `[T, B, num_units * num_directions]` (batch_first: False)

            outputs_list = []
            if self.downsample_list[i]:
                for t in range(max_time):
                    # Pick up features at even time step
                    if (t + 1) % 2 == 0:
                        if self.batch_first:
                            outputs_t = outputs[:, t, :].unsqueeze(1)
                        else:
                            outputs_t = outputs[t, :, :].unsqueeze(0)
                        outputs_list.append(outputs_t)
                if self.batch_first:
                    outputs = torch.cat(outputs_list, dim=1)
                    max_time = outputs.size(1)
                else:
                    outputs = torch.cat(outputs_list, dim=0)
                    max_time = outputs.size(0)

                # print(outputs.size())

        h_n = torch.cat(final_state_list, dim=0)

        return outputs, h_n
