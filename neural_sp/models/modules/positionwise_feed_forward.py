#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise fully-connected feed-forward neural network (FFN)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import random
import torch
import torch.nn as nn

from neural_sp.models.modules.gelu import gelu, gelu_accurate
from neural_sp.models.modules.glu import LinearGLUBlock

random.seed(1)

logger = logging.getLogger(__name__)


class PositionwiseFeedForward(nn.Module):
    """Positionwise fully-connected feed-forward neural network (FFN) layer.

    Args:
        d_model (int): input and output dimension
        d_ff (int): hidden dimension
        dropout (float): dropout probability
        activation: non-linear function
        param_init (str): parameter initialization method

    """

    def __init__(self, d_model, d_ff, dropout, activation, param_init):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        if activation == 'relu':
            self.activation = torch.relu
        elif activation == 'gelu':
            self.activation = lambda x: gelu(x)
        elif activation == 'gelu_accurate':
            self.activation = lambda x: gelu_accurate(x)
        elif activation == 'glu':
            self.activation = LinearGLUBlock(d_ff)
        else:
            raise NotImplementedError(activation)
        logger.info('FFN activation: %s' % activation)

        if param_init == 'xavier_uniform':
            self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        nn.init.constant_(self.w_1.bias, 0.)
        nn.init.constant_(self.w_2.bias, 0.)

    def forward(self, xs):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
        Returns:
            xs (FloatTensor): `[B, T, d_model]`

        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))
