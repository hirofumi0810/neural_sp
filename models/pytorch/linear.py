#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""MLP layer."""

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
        """
        super(LinearND, self).__init__()
        self.fc = nn.Linear(*size, bias=bias)

    def forward(self, x):
        size = x.size()
        n = np.prod(size[:-1])
        out = x.contiguous().view((int(n), int(size[-1])))
        out = self.fc(out)
        size = list(size)
        size[-1] = out.size()[-1]
        return out.view(size)
