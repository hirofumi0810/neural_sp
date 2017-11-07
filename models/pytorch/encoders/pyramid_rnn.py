#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Pyramid RNN encoders.
   This implementation is bases on
        https://arxiv.org/abs/1508.01211
            Chan, William, et al. "Listen, attend and spell."
                arXiv preprint arXiv:1508.01211 (2015).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable


class PyramidRNNEncoder(nn.Module):
    """Pyramid RNN encoder.
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
        downsample_list (list): downsample in the corresponding layers (True)
            ex.) [False, True, True, False] means that downsample is conducted
                in the 2nd and 3rd layers.
        downsample_type (string): drop or concat
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
                 downsample_list,
                 downsample_type='drop',
                 use_cuda=False,
                 batch_first=False,):

        super(PyramidRNNEncoder, self).__init__()

        if len(downsample_list) != num_layers:
            raise ValueError(
                'downsample_list must be the same size as num_layers.')
        if downsample_type not in ['drop', 'concat']:
            raise TypeError('downsample_type must be "drop" or "concat".')

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

        self.downsample_list = downsample_list
        self.downsample_type = downsample_type

        self.rnns = []
        for i in range(num_layers):
            next_input_size = num_units * self.num_directions
            if downsample_type == 'concat' and i > 0 and downsample_list[i - 1]:
                next_input_size *= 2

            if rnn_type == 'lstm':
                rnn = nn.LSTM(
                    input_size if i == 0 else next_input_size,
                    hidden_size=num_units,
                    num_layers=1,
                    bias=True,
                    batch_first=batch_first,
                    dropout=dropout,
                    bidirectional=bidirectional)
            elif rnn_type == 'gru':
                rnn = nn.GRU(
                    input_size if i == 0 else next_input_size,
                    hidden_size=num_units,
                    num_layers=1,
                    bias=True,
                    batch_first=batch_first,
                    dropout=dropout,
                    bidirectional=bidirectional)
            elif rnn_type == 'rnn':
                rnn = nn.RNN(
                    input_size if i == 0 else next_input_size,
                    hidden_size=num_units,
                    num_layers=1,
                    bias=True,
                    batch_first=batch_first,
                    dropout=dropout,
                    bidirectional=bidirectional)
            else:
                raise TypeError('rnn_type must be "lstm" or "gru" or "rnn".')

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

    def forward(self, inputs, volatile=False):
        """Forward computation.
        Args:
            inputs: A tensor of size `[B, T, input_size]`
            volatile (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            outputs:
                if batch_first is True, a tensor of size
                    `[B, T // len(downsample_list), num_units * num_directions]`
                else
                    `[T // len(downsample_list), B, num_units * num_directions]`
            h_n: A tensor of size
                `[num_layers * num_directions, B, num_units]`
        """
        batch_size, max_time = inputs.size()[:2]

        # Initialize hidden states (and memory cells) per mini-batch
        h_0 = self._init_hidden(batch_size=batch_size, volatile=volatile)

        if not self.batch_first:
            # Reshape to the time-major
            inputs = inputs.transpose(0, 1)

        outputs = inputs
        final_state_list = []
        for i in range(self.num_layers):
            if self.rnn_type == 'lstm':
                outputs, (h_n, c_n) = self.rnns[i](outputs, hx=h_0)
            else:
                # gru or rnn
                outputs, h_n = self.rnns[i](outputs, hx=h_0)

            # Save the last hiddne state
            final_state_list.append(h_n)

            outputs_list = []
            if self.downsample_list[i]:
                for t in range(max_time):
                    # Pick up features at even time step
                    if (t + 1) % 2 == 0:
                        if self.batch_first:
                            outputs_t = outputs[:, t:t + 1, :]
                            # NOTE: `[B, 1, num_units * num_directions]`
                        else:
                            outputs_t = outputs[t:t + 1, :, :]
                            # NOTE: `[1, B, num_units * num_directions]`

                        # Concatenate the successive frames
                        if self.downsample_type == 'concat':
                            if self.batch_first:
                                outputs_t_prev = outputs[:, t - 1:t, :]
                            else:
                                outputs_t_prev = outputs[t - 1:t, :, :]
                            outputs_t = torch.cat(
                                [outputs_t_prev, outputs_t], dim=2)

                        outputs_list.append(outputs_t)

                if self.batch_first:
                    outputs = torch.cat(outputs_list, dim=1)
                    # `[B, T_prev // 2, num_units * num_directions (* 2)]`
                    max_time = outputs.size(1)
                else:
                    outputs = torch.cat(outputs_list, dim=0)
                    # `[T_prev // 2, B, num_units * num_directions (* 2)]`
                    max_time = outputs.size(0)

        h_n = torch.cat(final_state_list, dim=0)

        return outputs, h_n
