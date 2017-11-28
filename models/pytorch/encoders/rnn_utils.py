#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilitiels for RNN encoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable


def _init_hidden(batch_size, rnn_type, num_units, num_directions,
                 num_layers, use_cuda, volatile):
    """Initialize hidden states.
    Args:
        batch_size (int): the size of mini-batch
        rnn_type (string): lstm or gru or rnn
        num_units (int):
        num_directions (int):
        num_layers (int):
        use_cuda (bool, optional):
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
        num_layers * num_directions, batch_size, num_units))

    if volatile:
        h_0.volatile = True

    if use_cuda:
        h_0 = h_0.cuda()

    if rnn_type == 'lstm':
        c_0 = Variable(torch.zeros(
            num_layers * num_directions, batch_size, num_units))

        if volatile:
            c_0.volatile = True

        if use_cuda:
            c_0 = c_0.cuda()

        return (h_0, c_0)
    else:
        return h_0
