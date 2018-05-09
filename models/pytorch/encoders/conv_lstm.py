#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Convolutional LSTM encoders (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ConvLSTMEncoder(nn.Module):
    """Convolutional LSTM encoders."""

    def __init__(self, arg):

        super(ConvLSTMEncoder, self).__init__()
        self.arg = arg

    def forward(self, xs, x_lens):
        """Forward computation.
        Args:
            xs (torch.FloatTensor): A tensor of size `[B, T, input_size]`
            x_lens (torch.IntTensor):
        Returns:
            xs (torch.FloatTensor): A tensor of size `[B, T', feature_dim]`
            x_lens (torch.IntTensor):
        """
        raise NotImplementedError
