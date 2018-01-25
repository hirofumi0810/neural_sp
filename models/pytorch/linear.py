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

        self.dropout = dropout

        self.fc = nn.Linear(*size, bias=bias)
        if dropout > 0:
            self.drop = nn.Dropout(p=dropout)

    def forward(self, xs):
        """Forward computation.
        Args:
            xs (Variable, float): A tensor of size `[B, T, input_dim]`
        Returns:
            xs (Variable, float): A tensor of size `[B, T, size[-1]]`
        """
        size = list(xs.size())
        outputs = xs.contiguous().view(
            (int(np.prod(size[:-1])), int(size[-1])))
        outputs = self.fc(outputs)
        if self.dropout > 0:
            outputs = self.drop(outputs)
        size[-1] = outputs.size()[-1]
        return outputs.view(size)


class Embedding(nn.Module):

    def __init__(self, num_classes, embedding_dim, dropout=0):
        """
        Args:
            num_classes (int): the number of nodes in softmax layer
                (including <SOS> and <EOS> classes)
            embedding_dim (int): the dimension of the embedding in target spaces
            dropout (float, optional): the probability to drop nodes of the embedding
        """
        super(Embedding, self).__init__()

        self.dropout = dropout

        self.embed = nn.Embedding(num_classes, embedding_dim)
        if dropout > 0:
            self.drop = nn.Dropout(p=dropout)

    def forward(self, y):
        """Forward computation.
        Args:
            y (Variable, long): A tensor of size `[B, 1]`
        Returns:
            y (Variable, float): A tensor of size `[B, 1, embedding_dim]`
        """
        y = self.embed(y)
        if self.dropout > 0:
            y = self.drop(y)
        return y
