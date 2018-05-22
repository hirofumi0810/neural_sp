#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""RNN language model (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.pytorch.base import ModelBase
from models.pytorch.linear import LinearND, Embedding
from models.pytorch.encoders.rnn import _init_hidden
from utils.io.variable import np2var, var2np


class RNNLM(ModelBase):
    """RNN language model.
    Args:
        num_classes
        embedding_dim
        rnn_type
        bidirectional
        num_units
        num_layers
        dropout_embedding
        dropout_hidden
        dropout_output
        parameter_init_distribution
        parameter_init
        tie_weights
    """

    def __init__(self,
                 num_classes,
                 embedding_dim,
                 rnn_type,
                 bidirectional,
                 num_units,
                 num_layers,
                 dropout_embedding,
                 dropout_hidden,
                 dropout_output,
                 parameter_init_distribution='uniform',
                 parameter_init=0.1,
                 tie_weights=False,
                 init_forget_gate_bias_with_one=True):

        super(ModelBase, self).__init__()
        self.model_type = 'rnnlm'

        # TODO: clip_activation

        self.embedding_dim = embedding_dim
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_units = num_units
        self.num_layers = num_layers
        self.parameter_init = parameter_init
        self.tie_weights = tie_weights

        self.num_classes = num_classes + 1  # Add <EOS> class
        # self.padded_index = 0
        self.padded_index = -1

        self.embed = Embedding(num_classes=self.num_classes,
                               embedding_dim=embedding_dim,
                               dropout=dropout_embedding,
                               ignore_index=self.padded_index)
        # NOTE: share the embedding layer between inputs and outputs

        if rnn_type == 'lstm':
            rnn = nn.LSTM(embedding_dim,
                          hidden_size=num_units,
                          num_layers=num_layers,
                          bias=True,
                          batch_first=True,
                          dropout=dropout_hidden,
                          bidirectional=bidirectional)
        elif rnn_type == 'gru':
            rnn = nn.GRU(embedding_dim,
                         hidden_size=num_units,
                         num_layers=num_layers,
                         bias=True,
                         batch_first=True,
                         dropout=dropout_hidden,
                         bidirectional=bidirectional)
        elif rnn_type == 'rnn':
            rnn = nn.RNN(embedding_dim,
                         hidden_size=num_units,
                         num_layers=num_layers,
                         nonlinearity='tanh',
                         # nonlinearity='relu',
                         bias=True,
                         batch_first=True,
                         dropout=dropout_hidden,
                         bidirectional=bidirectional)
        setattr(self, rnn_type, rnn)

        self.output = LinearND(
            num_units * self.num_directions, self.num_classes,
            dropout=dropout_output)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if num_units != embedding_dim:
                raise ValueError(
                    'When using the tied flag, num_units must be equal to embedding_dim')
            self.output.fc.weight = self.embed.embed.weight

        ##################################################
        # Initialize parameters
        ##################################################
        self.init_weights(parameter_init,
                          distribution=parameter_init_distribution,
                          ignore_keys=['bias'])

        # Initialize all biases with 0
        self.init_weights(0, distribution='constant', keys=['bias'])

        # Initialize bias in forget gate with 1
        if init_forget_gate_bias_with_one:
            self.init_forget_gate_bias_with_one()

    def forward(self, ys, y_lens, is_eval=False):
        """Forward computation.
        Args:
            ys (np.ndarray): A tensor of size `[B, T]`
            y_lens (np.ndarray): A tensor of size `[B]`
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (torch.autograd.Variable(float) or float): A tensor of size `[1]`
        """
        # Wrap by Variable
        ys = np2var(
            ys, dtype='long', use_cuda=self.use_cuda, backend='pytorch')
        # NOTE: ys must be long
        y_lens = np2var(
            y_lens, dtype='int', use_cuda=self.use_cuda, backend='pytorch')

        # NOTE: exclude <EOS>
        y_lens = y_lens - 1

        if is_eval:
            self.eval()
        else:
            self.train()

        # Path through character embedding
        ys_embed = []
        for t in range(ys.size(1) - 1):
            ys_embed.append(self.embed(ys[:, t:t + 1]))
        ys_embed = torch.cat(ys_embed, dim=1)
        # ys_embed: `[B, T - 1, embedding_dim]`
        # NOTE: exclude the last token

        # Sort xs by lengths in descending order
        y_lens, perm_idx = y_lens.sort(dim=0, descending=True)
        ys = ys[perm_idx]
        ys_embed = ys_embed[perm_idx]
        # NOTE: batch-first yet here
        # NOTE: must be descending order for pack_padded_sequence
        y_lens = var2np(y_lens).tolist()

        # Initialize hidden states (and memory cells) per mini-batch
        h_0 = _init_hidden(batch_size=ys.size(0),
                           rnn_type=self.rnn_type,
                           num_units=self.num_units,
                           num_directions=self.num_directions,
                           num_layers=self.num_layers,
                           use_cuda=self.use_cuda,
                           volatile=is_eval)

        # Pack RNN inputs
        ys_embed = pack_padded_sequence(ys_embed, y_lens, batch_first=True)

        # Path through RNN
        ys_embed, _ = getattr(self, self.rnn_type)(ys_embed, hx=h_0)

        # Unpack RNN outputs
        ys_embed, unpacked_seq_len = pad_packed_sequence(
            ys_embed, batch_first=True, padding_value=0)
        # assert y_lens - 1 == unpacked_seq_len

        logits = self.output(ys_embed)

        # Compute XE sequence loss
        loss = F.cross_entropy(
            input=logits.view((-1, logits.size(2))),
            target=ys[:, 1:].contiguous().view(-1),
            ignore_index=self.padded_index, size_average=False) / len(ys)
        # NOTE: Exclude first <SOS>
        # NOTE: ys are padded by <SOS>

        if is_eval:
            loss = loss.data[0]

        return loss

    def _encode(self):
        pass

    def decode(self, start_token, beam_width, max_decode_len):
        """Decoding in the inference stage.
        Args:
            start_token (np.ndarray): A tensor of size `[B]`
            beam_width (int): the size of beam
            max_decode_len (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
            perm_idx (np.ndarray): A tensor of size `[B]`
        """
        raise NotImplementedError

        # Wrap by Variable
        start_token = np2var(
            start_token, use_cuda=self.use_cuda, volatile=True, backend='pytorch')

        # Change to evaluation mode
        self.eval()

        # Path through character embedding
        start_token_embed = self.embed(start_token.unsqueeze(1))
        # start_token_embed: `[B, 1, embedding_dim]`

        # Initialize hidden states (and memory cells) per mini-batch
        h_0 = _init_hidden(batch_size=ys.size(0),
                           rnn_type=self.rnn_type,
                           num_units=self.num_units,
                           num_directions=self.num_directions,
                           num_layers=self.num_layers,
                           use_cuda=self.use_cuda,
                           volatile=True)

        # Path through RNN
        ys_embed, _ = getattr(self, self.rnn_type)(ys_embed, hx=h_0)

        if beam_width == 1:
            best_hyps, _ = self._decode_infer_greedy(
                enc_out, y_lens, max_decode_len)
        else:
            best_hyps = self._decode_infer_beam(
                enc_out, y_lens, beam_width, max_decode_len,
                length_penalty, coverage_penalty)

        # NOTE: index 0 is reserved for <SOS>
        best_hyps -= 1

        # Permutate indices to the original order
        if perm_idx is None:
            perm_idx = np.arange(0, len(xs), 1)
        else:
            perm_idx = var2np(perm_idx, backend='pytorch')

        return best_hyps, perm_idx
