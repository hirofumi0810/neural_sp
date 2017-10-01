#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class RNN_Encoder(nn.Module):
    """Bidirectional LSTM encoder.
    Args:
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        num_classes (int): the number of classes of target labels.
            If 0, return hidden states before passing through the softmax layer
        rnn_type (string): lstm or gru or rnn
        bidirectional (bool): if True, use the bidirectional model
        use_peephole (bool): if True, use peephole connections
        parameter_init (float): Range of uniform distribution to initialize
            weight parameters
        clip_activation (float): Range of activation clipping (> 0)
        num_proj (int): the number of nodes in recurrent projection layer
        bottleneck_dim (int): the dimensions of the bottleneck layer
    """

    def __init__(self,
                 input_size,
                 num_units,
                 num_layers,
                 num_classes,
                 rnn_type,
                 bidirectional,
                 dropout,
                 #  use_peephole,
                 parameter_init):
                #  clip_activation,
                #  num_proj,
                #  bottleneck_dim):

        super(RNN_Encoder, self).__init__()

        self.input_size = input_size
        self.num_units = num_units
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = dropout
        # NOTE: dropout is applied except the last layer
        self.parameter_init = parameter_init

        # self.use_peephole = use_peephole
        # self.clip_activation = clip_activation
        # self.num_proj = num_proj
        # self.bottleneck_dim = bottleneck_dim

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, num_units,
                               num_layers=num_layers,
                               bias=True,
                               batch_first=False,
                               dropout=dropout,
                               bidirectional=bidirectional)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, num_units,
                              num_layers=num_layers,
                              bias=True,
                              batch_first=False,
                              dropout=dropout,
                              bidirectional=bidirectional)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, num_units,
                              num_layers=num_layers,
                              bias=True,
                              batch_first=False,
                              dropout=dropout,
                              bidirectional=bidirectional)
        else:
            raise ValueError('rnn_type must be "lstm" or "gru" or "rnn".')

        # TODO: Add bottleneck layer

        if num_classes != 0:
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

        if self.rnn_type == 'lstm':
            # return hidden states & memory cells
            c_0 = Variable(torch.zeros(
                self.num_layers * self.num_directions, batch_size, self.num_units))
            return (h_0, c_0)
        else:  # gru or rnn
            # return hidden states
            return h_0

    def forward(self, inputs):
        """Forward computation.
        Args:
            inputs: A tensor of size `[B, T, input_size]`
        Returns:
            logits: A tensor of size `[T, B, num_classes]`
            final_state: A tensor of size
                `[num_layers * num_directions, B, num_units]`
        """
        # TODO: Reshape inputs to time-major in advance

        # Initialize hidden states (and memory cells) per mini-batch
        h_0 = self._init_hidden(batch_size=inputs.size(0))

        # Reshape to the batct-major
        inputs = inputs.transpose(0, 1)

        if self.rnn_type == 'lstm':
            outputs, (h_n, c_n) = self.rnn(inputs, h_0)
        else:  # gru or rnn
            outputs, h_n = self.rnn(inputs, h_0)
        # NOTE: outputs: `[T, B, num_units * num_directions]`

        if self.num_classes == 0:
            return outputs, h_n
        # NOTE: this is for seq2seq

        logits = self.fc(outputs)

        return logits, h_n
