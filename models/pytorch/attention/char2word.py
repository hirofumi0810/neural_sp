#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Chracter2Word (C2W) composition model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

from models.pytorch.encoders.rnn import _init_hidden


class LSTMChar2Word(nn.Module):
    """C2W model.
    Args:
        num_units (int):
        bidirectional (bool):
        char_embeddings (int):
        word_embedding_dim (int):
        dropout (float, optional):
    """

    def __init__(self,
                 num_units,
                 num_layers,
                 bidirectional,
                 char_embedding_dim,
                 word_embedding_dim,
                 dropout=0):
        super(LSTMChar2Word, self).__init__()

        self.num_units = num_units
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # Ling's (bidirectional) LSTM-based C2W composition model
        self.c2w_lstm = nn.LSTM(char_embedding_dim,
                                hidden_size=num_units,
                                num_layers=num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=bidirectional)

        self.W_fw = nn.Linear(num_units, word_embedding_dim)
        self.W_bw = nn.Linear(num_units, word_embedding_dim)

    def forward(self, char_embeddings, volatile=False):
        """Forward computation.
        Args:
            char_embeddings (Variable, float): A tensor of size
                `[1 (B), char_num, embedding_dim_sub]`
            volatile (bool, optional):
        Returns:
            word_repr (Variable, float): A tensor of size
                `[1 (B), 1, word_embedding_dim]`
        """
        # Initialize hidden states (and memory cells) per mini-batch
        h_0 = _init_hidden(batch_size=1,
                           rnn_type='lstm',
                           num_units=self.num_units,
                           num_directions=self.num_directions,
                           num_layers=self.num_layers,
                           use_cuda=char_embeddings.is_cuda,
                           volatile=volatile)

        outputs, _ = self.c2w_lstm(char_embeddings, hx=h_0)
        # NOTE: outputs: `[1 (B), T, num_units * num_directions]`

        final_state_fw = outputs[:, -1, :self.num_units]
        # NOTE: `[1 (B), num_units]`

        word_repr = self.W_fw(final_state_fw)

        if self.num_directions == 2:
            final_state_bw = outputs[:, -1, self.num_units:self.num_units * 2]
            word_repr += self.W_bw(final_state_bw)

        return word_repr.unsqueeze(1)


class CNNHighwayChar2Word(object):
    """docstring for CNNHighwayChar2Word."""

    def __init__(self, arg):
        super(CNNHighwayChar2Word, self).__init__()
        self.arg = arg

    def forward(self):
        raise NotImplementedError
