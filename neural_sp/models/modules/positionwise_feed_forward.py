#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise fully-connected feed-forward neural network (FFN)."""

import logging
import torch
import torch.nn as nn

from neural_sp.models.modules.gelu import gelu, gelu_accurate
from neural_sp.models.modules.glu import LinearGLUBlock
from neural_sp.models.modules.initialization import init_with_xavier_uniform
from neural_sp.models.modules.swish import Swish


logger = logging.getLogger(__name__)


class PositionwiseFeedForward(nn.Module):
    """Positionwise fully-connected feed-forward neural network (FFN) layer.

    Args:
        d_model (int): input and output dimension
        d_ff (int): hidden dimension
        dropout (float): dropout probability
        activation (str): non-linear activation function
        param_init (str): parameter initialization method
        bottleneck_dim (int): bottleneck dimension for low-rank FFN

    """

    def __init__(self, d_model, d_ff, dropout, activation, param_init,
                 bottleneck_dim=0):
        super(PositionwiseFeedForward, self).__init__()

        self.bottleneck_dim = bottleneck_dim
        if bottleneck_dim > 0:
            self.w_1_e = nn.Linear(d_model, bottleneck_dim)
            self.w_1_d = nn.Linear(bottleneck_dim, d_ff)
            self.w_2_e = nn.Linear(d_ff, bottleneck_dim)
            self.w_2_d = nn.Linear(bottleneck_dim, d_model)
        else:
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
        elif activation == 'swish':
            self.activation = Swish()
        else:
            raise NotImplementedError(activation)
        logger.info('FFN activation: %s' % activation)

        if param_init == 'xavier_uniform':
            self.reset_parameters()
        else:
            logger.info('Parameter initialization is skipped.')

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_xavier_uniform(n, p)

    def forward(self, xs):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
        Returns:
            xs (FloatTensor): `[B, T, d_model]`

        """
        if self.bottleneck_dim > 0:
            return self.w_2_d(self.w_2_e(self.dropout(self.activation(self.w_1_d(self.w_1_e(xs))))))
        else:
            return self.w_2(self.dropout(self.activation(self.w_1(xs))))
