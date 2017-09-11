#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class RNN_Encoder(nn.Module):

    def __init__(self,
                 input_size,
                 num_units,
                 num_layers,
                 num_classes,
                 rnn_type,
                 bidirectional,
                 #  use_peephole,
                 parameter_init):
                #  clip_activation,
                #  num_proj,
                #  bottleneck_dim):

        super(RNN_Encoder, self).__init__()
        if rnn_type not in ['lstm', 'gru', 'rnn']:
            raise ValueError

        self.input_size = input_size
        self.num_units = num_units
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.parameter_init = parameter_init

        # self.use_peephole = use_peephole
        # self.clip_activation = clip_activation

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.input_size, self.num_units,
                               num_layers=self.num_layers,
                               bias=True,
                               batch_first=False,
                               dropout=0,
                               bidirectional=self.bidirectional)

        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.input_size, self.num_units,
                              num_layers=self.num_layers,
                              bias=True,
                              batch_first=False,
                              dropout=0,
                              bidirectional=self.bidirectional)

        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(self.input_size, self.num_units,
                              num_layers=self.num_layers,
                              bias=True,
                              batch_first=False,
                              dropout=0,
                              bidirectional=self.bidirectional)

        else:
            raise ValueError

        self.output = nn.Linear(
            self.num_units * self.num_directions, self.num_classes)

        # Initialize hidden states (and memory cells)
        self.hidden = self.init_hidden(batch_size=2)
        # TODO: fix it

    def init_hidden(self, batch_size):
        """Initialize hidden states.
        Returns:
            if rnn_type is 'lstm', return a tuple of tensors (h_0, c_0),
            otherwise return a tensor h_0.
        """

        # `(num_layers * num_directions, batch_size, num_units)`
        h_0 = autograd.Variable(torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.num_units))

        if self.rnn_type == 'lstm':
            # hidden states & memory cells
            c_0 = autograd.Variable(torch.zeros(
                self.num_layers * self.num_directions, batch_size, self.num_units))
            return (h_0, c_0)
        else:  # gru or rnn
            # hidden states
            return h_0

    def forward(self, inputs):
        """Forward computation.
        Args:
            inputs: A tensor of size `(time, batch_size, input_size)`
        Returns:
            logits: A tensor of size `(time, batch_size, num_classes)`
            final_state: if rnn_type is 'lstm', return a tuple of tensors
                (h_n, c_n), otherwise return a tensor h_n.
        """
        assert isinstance(inputs, autograd.Variable), "inputs must be autograd.Variable."
        assert len(inputs.size()) == 3, "inputs must be a tensor of size `(time, batch_size, num_classes)`."

        # Reshape to the batct-major
        inputs = inputs.transpose(0, 1)

        outputs, self.hidden = self.rnn(inputs, self.hidden)
        # NOTE: outputs: `(time, batch_size, num_units * num_directions)`

        outputs = self.output(outputs)
        logits = F.softmax(outputs)

        return logits, self.hidden
