#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""MLP & embedding layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.nn as nn


class LinearND(nn.Module):

    def __init__(self, in_size, out_size, bias=True, dropout=0):
        """Linear layer.

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
        """Forward computation.

        Args:
            xs (torch.autograd.Variable, float): A tensor of size
                `[B, T, input_dim]`
        Returns:
            xs (torch.autograd.Variable, float): A tensor of size
                `[B, T, size[-1]]`

        """
        size = list(xs.size())
        xs = xs.contiguous().view((int(np.prod(size[:-1])), int(size[-1])))
        xs = self.fc(xs)
        xs = self.dropout(xs)
        size[-1] = xs.size()[-1]
        return xs.view(size)


class Embedding(nn.Module):

    def __init__(self, num_classes, emb_dim, dropout=0, ignore_index=-1):
        """Embedding layer.

        Args:
            num_classes (int): the number of nodes in softmax layer
                (including <SOS> and <EOS> classes)
            emb_dim (int): the dimension of the embedding in target spaces
            dropout (float): the probability to dropout nodes of the embedding
            ignore_index (int):

        """
        super(Embedding, self).__init__()

        self.emb = nn.Embedding(num_classes, emb_dim, padding_idx=ignore_index)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, y):
        """Forward computation.

        Args:
            y (torch.autograd.Variable, long): A tensor of size `[B, L]`
        Returns:
            y (torch.autograd.Variable, float): A tensor of size
                `[B, L, emb_dim]`

        """
        return self.dropout(self.emb(y))
