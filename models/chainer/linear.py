#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""MLP & embedding layer (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L


class LinearND(chainer.Chain):

    def __init__(self, *size, bias=True, dropout=0, use_cuda=False):
        """
        A chainer.links.Linear layer modified to accept ND arrays.
            The function treats the last dimension of the input
            as the hidden dimension.
        Args:
            size ():
            bias (bool, optional):
            dropout (float, optional):
            use_cuda (bool, optional): if True, use GPUs
        """
        super(LinearND, self).__init__()

        self.dropout = dropout

        with self.init_scope():
            self.fc = L.Linear(*size,
                               nobias=not bias,
                               initialW=None,
                               initial_bias=None)
            if use_cuda:
                self.fc.to_gpu()

    def __call__(self, xs):
        """Forward computation.
        Args:
            xs (chainer.Variable):
        Returns:

        """
        size = list(xs.shape)
        outputs = xs.reshape(np.prod(size[:-1]), size[-1])
        outputs = self.fc(outputs)
        if self.dropout > 0:
            outputs = F.dropout(outputs, ratio=self.dropout)
        size[-1] = outputs.shape[-1]
        return outputs.reshape(size)


class Embedding(chainer.Chain):

    def __init__(self, num_classes, embedding_dim, dropout=0, use_cuda=False):
        """
        Args:
            num_classes (int): the number of nodes in softmax layer
                (including <SOS> and <EOS> classes)
            embedding_dim (int): the dimension of the embedding in target spaces
            dropout (float, optional): the probability to drop nodes of the embedding
            use_cuda (bool, optional): if True, use GPUs
        """
        super(Embedding, self).__init__()

        self.dropout = dropout

        with self.init_scope():
            self.embed = L.EmbedID(num_classes, embedding_dim,
                                   initialW=None)
            if use_cuda:
                self.embed.to_gpu()

    def __call__(self, y):
        """Forward computation.
        Args:
            y (chainer.Variable): A tensor of size `[B, 1]`
        Returns:
            y (chainer.Variable): A tensor of size `[B, 1, embedding_dim]`
        """
        y = self.embed(y)
        if self.dropout > 0:
            y = F.dropout(y, ratio=self.dropout)
        return y
