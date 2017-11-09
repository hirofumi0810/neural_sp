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
        num_proj (int): the number of nodes in the projection layer
        num_layers_main (int): the number of layers in the main task
        num_layers_sub (int): the number of layers in the sub task
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
        self.num_proj = num_proj
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
        for i_layer in range(num_layers_main):
            if rnn_type == 'lstm':
                rnn = nn.LSTM(
                    input_size if i_layer == 0 else num_units * self.num_directions,
                    hidden_size=num_units,
                    num_layers=1,
                    bias=True,
                    batch_first=batch_first,
                    dropout=dropout,
                    bidirectional=bidirectional)
                setattr(self, rnn_type + '_l' + str(i_layer), rnn)
            elif rnn_type == 'gru':
                rnn = nn.GRU(
                    input_size if i_layer == 0 else num_units * self.num_directions,
                    hidden_size=num_units,
                    num_layers=1,
                    bias=True,
                    batch_first=batch_first,
                    dropout=dropout,
                    bidirectional=bidirectional)
            elif rnn_type == 'rnn':
                rnn = nn.RNN(
                    input_size if i_layer == 0 else num_units * self.num_directions,
                    hidden_size=num_units,
                    num_layers=1,
                    bias=True,
                    batch_first=batch_first,
                    dropout=dropout,
                    bidirectional=bidirectional)
            else:
                raise ValueError('rnn_type must be "lstm" or "gru" or "rnn".')

            setattr(self, rnn_type + '_l' + str(i_layer), rnn)

            if use_cuda:
                rnn = rnn.cuda()

            self.rnns.append(rnn)

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
            1 * self.num_directions, batch_size, self.num_units))

        if volatile:
            h_0.volatile = True

        if self.use_cuda:
            h_0 = h_0.cuda()

        if self.rnn_type == 'lstm':
            c_0 = Variable(torch.zeros(
                1 * self.num_directions, batch_size, self.num_units))

            if volatile:
                c_0.volatile = True

            if self.use_cuda:
                c_0 = c_0.cuda()

            return (h_0, c_0)
        else:
            # gru or rnn
            return h_0

    def forward(self, inputs, volatile=True):
        """Forward computation.
        Args:
            inputs: A tensor of size `[B, T, input_size]`
            volatile (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            outputs_main:
                if batch_first is True, a tensor of size
                    `[B, T, num_units * num_directions]`
                else
                    `[T, B, num_units * num_directions]`
            final_state_fw_main: A tensor of size `[1, B, num_units]`
            outputs_sub:
                if batch_first is True, a tensor of size
                    `[B, T, num_units * num_directions]`
                else
                    `[T, B, num_units * num_directions]`
            final_state_fw_sub: A tensor of size `[1, B, num_units]`
        """
        batch_size, max_time = inputs.size()[:2]

        # Initialize hidden states (and memory cells) per mini-batch
        h_0 = self._init_hidden(batch_size=batch_size, volatile=volatile)

        if not self.batch_first:
            # Reshape to the time-major
            inputs = inputs.transpose(0, 1)

        outputs_main = inputs
        for i_layer in range(self.num_layers_main):
            if self.rnn_type == 'lstm':
                outputs_main, (h_n, c_n) = self.rnns[i_layer](
                    outputs_main, hx=h_0)
            else:
                # gru or rnn
                outputs_main, h_n = self.rnns[i_layer](
                    outputs_main, hx=h_0)

            if i_layer == self.num_layers_sub - 1:
                outputs_sub = outputs_main
                h_n_sub = h_n

        # Pick up the final state of the top layer (forward)
        if self.num_directions == 2:
            final_state_fw_main = h_n[-2:-1, :, :]
            final_state_fw_sub = h_n_sub[-2:-1, :, :]
        else:
            final_state_fw_main = h_n[-1, :, :].unsqueeze(dim=0)
            final_state_fw_sub = h_n_sub[-1, :, :].unsqueeze(dim=0)
        # NOTE: h_n: `[num_layers_main * num_directions, B, num_units]`
        #   h_n_sub: `[num_layers_sub * num_directions, B, num_units]`

        return outputs_main, final_state_fw_main, outputs_sub, final_state_fw_sub
