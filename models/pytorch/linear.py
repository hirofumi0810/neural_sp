#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""MLP layer (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.nn as nn


class LinearND(nn.Module):

    def __init__(self, *size, bias=True):
        """
        A torch.nn.Linear layer modified to accept ND arrays.
            The function treats the last dimension of the input
            as the hidden dimension.
        Args:
            size ():
            bias (bool, optional): if False, remove a bias term
        """
        super(LinearND, self).__init__()
        self.fc = nn.Linear(*size, bias=bias)

    def forward(self, xs):
        size = list(xs.size())
        outputs = xs.contiguous().view(
            (int(np.prod(size[:-1])), int(size[-1])))
        outputs = self.fc(outputs)
        size[-1] = outputs.size()[-1]
        return outputs.view(size)
