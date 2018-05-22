#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""MLP & embedding layer (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
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
        xs = xs.contiguous().view(
            (int(np.prod(size[:-1])), int(size[-1])))
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
        y = self.embed(y)
        y = self.dropout(y)
        return y


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
        self.label_smoothing_prob = label_smoothing_prob

        self.embed = LinearND(num_classes, embedding_dim,
                              bias=False,
                              dropout=dropout)

    def forward(self, y):
        """Forward computation.
        Args:
            y (torch.autograd.Variable, long): A tensor of size
                `[B, 1]`
        Returns:
            y (torch.autograd.Variable, float): A tensor of size
                `[B, 1, embedding_dim]`
        """
        # Convert to one-hot labels
        y = to_onehot(y, self.num_classes, self.label_smoothing_prob)
        # y: `[B, 1, num_classes]`

        y = self.embed(y)

        return y


def to_onehot(y, num_classes, label_smoothing_prob=0):
    """Convert indices into one-hot encoding.
    Args:
        y (torch.autograd.Variable, long): Indices of labels.
            A tensor of size `[B, 1]`.
        num_classes (int): the number of classes
        label_smoothing_prob (float, optional):
    Returns:
        y (torch.autograd.Variable, float): A tensor of size
            `[B, 1, num_classes]`
    """
    batch_size = y.size(0)
    y_onehot = torch.FloatTensor(batch_size, num_classes).zero_()
    y_onehot.scatter_(1, y.data.cpu(), 1)
    y_onehot = torch.autograd.Variable(y_onehot).unsqueeze(1)
    if y.is_cuda:
        y_onehot = y_onehot.cuda()

    # Label smoothing
    if label_smoothing_prob > 0:
        y = y * (1 - label_smoothing_prob) + 1 / \
            num_classes * label_smoothing_prob

    # TODO: fix bugs
    # if y.volatile:
    #     y_onehot.volatile = True

    return y_onehot
