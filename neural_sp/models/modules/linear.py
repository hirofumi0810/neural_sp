#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Linear layer for the N-dimentional tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.nn as nn


class LinearND(nn.Module):

    def __init__(self, in_size, out_size, bias=True, dropout=0):
        """Linear layer for the N-dimentional tensor

        A torch.nn.Linear layer modified to accept ND arrays.
            The function treats the last dimension of the input
            as the hidden dimension.
        Args:
            in_size (int):
            out_size (int):
            bias (bool): if False, remove a bias term
            dropout (float):

        """
        super(LinearND, self).__init__()

        self.fc = nn.Linear(in_size, out_size, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, xs):
        """Forward pass.

        Args:
            xs (FloatTensor):
        Returns:
            xs (FloatTensor):

        """
        size = list(xs.size())
        xs = xs.contiguous().view((int(np.prod(size[:-1])), int(size[-1])))
        xs = self.dropout(self.fc(xs))
        size[-1] = xs.size()[-1]
        return xs.view(size)
