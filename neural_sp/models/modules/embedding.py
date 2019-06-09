#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Embedding layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class Embedding(nn.Module):

    def __init__(self, vocab, emb_dim, dropout=0, ignore_index=-1):
        """Embedding layer.

        Args:
            vocab (int): number of nodes in softmax layer
            emb_dim (int): dimension of the embedding in target spaces
            dropout (float): probability to dropout nodes of the embedding
            ignore_index (int):

        """
        super(Embedding, self).__init__()

        self.emb_dim = emb_dim
        self.embed = nn.Embedding(vocab, emb_dim, padding_idx=ignore_index)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, ys):
        """Forward pass.

        Args:
            ys (LongTensor): `[B, L]`
        Returns:
            y_emb (FloatTensor): `[B, L, emb_dim]`

        """
        ys_emb = self.dropout(self.embed(ys.long()))
        return ys_emb
