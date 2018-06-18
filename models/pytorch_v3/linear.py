#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""MLP & embedding layer (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.nn as nn
from models.pytorch_v3.utils import to_onehot


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
            y (torch.autograd.Variable, long): A tensor of size `[B, 1]`
        Returns:
            y (torch.autograd.Variable, float): A tensor of size
                `[B, 1, embedding_dim]`
        """
        return self.dropout(self.embed(y))


class Embedding_LS(nn.Module):

    def __init__(self, num_classes, embedding_dim, dropout=0,
                 label_smoothing_prob=0.):
        """Embedding layer with label smoothing.
        Args:
            num_classes (int): the number of nodes in softmax layer
                (including <SOS> and <EOS> classes)
            embedding_dim (int): the dimension of the embedding in target spaces
            dropout (float, optional): the probability to drop nodes of the embedding
            label_smoothing_prob (float, optional):
        """
        super(Embedding_LS, self).__init__()

        self.num_classes = num_classes
        self.ls_prob = label_smoothing_prob
        assert label_smoothing_prob > 0
        self.embed = LinearND(num_classes, embedding_dim,
                              bias=False,
                              dropout=dropout)

    def forward(self, ys):
        """Forward computation.
        Args:
            ys (torch.autograd.Variable, long): A tensor of size
                `[B, L]`
        Returns:
            ys (torch.autograd.Variable, float): A tensor of size
                `[B, L, embedding_dim]`
        """
        # Label smoothing
        ys = to_onehot(ys, self.num_classes) * (1 - self.ls_prob) + \
            (1 / self.num_classes) * self.ls_prob
        # ys: `[B, L, num_classes]`

        return self.embed(ys)
