#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""MLP & embedding layer (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.nn as nn


class LinearND(nn.Module):

    def __init__(self, *size, bias=True, dropout=0):
        """
        A torch.nn.Linear layer modified to accept ND arrays.
            The function treats the last dimension of the input
            as the hidden dimension.
        Args:
            size ():
            bias (bool, optional): if False, remove a bias term
            dropout (float, optional):
        """
        super(LinearND, self).__init__()

        self.fc = nn.Linear(*size, bias=bias)
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

    def __init__(self, num_classes, embedding_dim, dropout=0, ignore_index=-1):
        """Embedding layer.
        Args:
            num_classes (int): the number of nodes in softmax layer
                (including <SOS> and <EOS> classes)
            embedding_dim (int): the dimension of the embedding in target spaces
            dropout (float, optional): the probability to drop nodes of the embedding
            ignore_index (int, optional):
        """
        super(Embedding, self).__init__()

        self.embed = nn.Embedding(num_classes, embedding_dim,
                                  padding_idx=ignore_index)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, y):
        """Forward computation.
        Args:
            y (torch.autograd.Variable, long): A tensor of size `[B, L]`
        Returns:
            y (torch.autograd.Variable, float): A tensor of size
                `[B, L, embedding_dim]`
        """
        return self.dropout(self.embed(y))
