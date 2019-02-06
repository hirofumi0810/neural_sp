#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Sequence-level RNN language model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.base import ModelBase
from neural_sp.models.linear import Embedding
from neural_sp.models.linear import LinearND
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


class SeqRNNLM(ModelBase):
    """Sequence-level RNN language model. This is used for RNNLM training."""

    def __init__(self, args):

        super(ModelBase, self).__init__()

        self.emb_dim = args.emb_dim
        self.rnn_type = args.rnn_type
        assert args.rnn_type in ['lstm', 'gru']
        self.nunits = args.nunits
        self.nlayers = args.nlayers
        self.tie_embedding = args.tie_embedding
        self.residual = args.residual
        self.use_glu = args.use_glu
        self.backward = args.backward

        self.vocab = args.vocab
        self.sos = 2   # NOTE: the same as <eos>
        self.eos = 2
        self.pad = 3
        # NOTE: reserved in advance

        self.embed = Embedding(vocab=self.vocab,
                               emb_dim=args.emb_dim,
                               dropout=args.dropout_emb,
                               ignore_index=self.pad)

        self.fast_impl = False
        if args.nprojs == 0 and not args.residual:
            self.fast_impl = True
            if 'lstm' in args.rnn_type:
                rnn = nn.LSTM
            elif 'gru' in args.rnn_type:
                rnn = nn.GRU
            else:
                raise ValueError('rnn_type must be "(b)lstm" or "(b)gru".')

            self.rnn = rnn(args.emb_dim, args.nunits, args.nlayers,
                           bias=True,
                           batch_first=True,
                           dropout=args.dropout_hidden,
                           bidirectional=False)
            # NOTE: pytorch introduces a dropout layer on the outputs of each layer EXCEPT the last layer
            self.dropout_top = nn.Dropout(p=args.dropout_hidden)
        else:
            self.rnn = torch.nn.ModuleList()
            self.dropout = torch.nn.ModuleList()
            if args.nprojs > 0:
                self.proj = torch.nn.ModuleList()
            for l in range(args.nlayers):
                if l == 0:
                    rnn_idim = args.emb_dim
                elif args.nprojs > 0:
                    rnn_idim = args.nprojs
                else:
                    rnn_idim = args.nunits

                if args.rnn_type == 'lstm':
                    rnn_i = nn.LSTM
                elif args.rnn_type == 'gru':
                    rnn_i = nn.GRU

                self.rnn += [rnn_i(rnn_idim, args.nunits, 1,
                                   bias=True,
                                   batch_first=True,
                                   dropout=0,
                                   bidirectional=False)]
                self.dropout += [nn.Dropout(p=args.dropout_hidden)]

                if l != self.nlayers - 1 and args.nprojs > 0:
                    self.proj += [LinearND(args.nunits * self.ndirs, args.nprojs)]

        if self.use_glu:
            self.fc_glu = LinearND(args.nprojs if args.nprojs > 0 else args.nunits,
                                   args.nprojs * 2 if args.nprojs > 0 else args.nunits * 2,
                                   dropout=args.dropout_hidden)

        self.output = LinearND(args.nprojs if args.nprojs > 0 else args.nunits,
                               self.vocab,
                               dropout=args.dropout_out)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if args.tie_embedding:
            if args.nunits != args.emb_dim:
                raise ValueError('When using the tied flag, nunits must be equal to emb_dim.')
            self.output.fc.weight = self.embed.embed.weight

        # Initialize weight matrices
        self.init_weights(args.param_init, dist=args.param_init_dist)

        # Initialize bias vectors with zero
        self.init_weights(0, dist='constant', keys=['bias'])

        # Recurrent weights are orthogonalized
        if args.rec_weight_orthogonal:
            self.init_weights(args.param_init, dist='orthogonal',
                              keys=[args.rnn_type, 'weight'])

        # Initialize bias in forget gate with 1
        self.init_forget_gate_bias_with_one()

    def forward(self, ys, reporter=None, is_eval=False):
        """Forward computation.

        Args:
            ys (list): A list of length `[B]`, which contains arrays of size `[L]`
            reporter ():
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (FloatTensor): `[1]`
            reporter ():

        """
        if is_eval:
            self.eval()
            with torch.no_grad():
                loss, observation = self._forward(ys)
        else:
            self.train()
            loss, observation = self._forward(ys)

        # Report here
        if reporter is not None:
            reporter.add(observation, is_eval)

        return loss, reporter

    def _forward(self, ys):
        if self.backward:
            ys = [np2tensor(np.fromiter(y[::-1], dtype=np.int64), self.device_id).long() for y in ys]
        else:
            ys = [np2tensor(np.fromiter(y, dtype=np.int64), self.device_id).long() for y in ys]

        ys = pad_list(ys, self.pad)
        ys_in = ys[:, :-1]
        ys_out = ys[:, 1:]

        # Path through embedding
        ys_in = self.embed(ys_in)

        residual = None
        if self.fast_impl:
            ys_in, _ = self.rnn(ys_in, hx=None)
            ys_in = self.dropout_top(ys_in)
        else:
            for l in range(self.nlayers):
                # Path through RNN
                ys_in, _ = self.rnn[l](ys_in, hx=None)
                ys_in = self.dropout[l](ys_in)

                # Residual connection
                if self.residual and l > 0:
                    ys_in += residual
                residual = ys_in
                # NOTE: Exclude residual connection from the raw inputs

        if self.use_glu:
            if self.residual:
                residual = ys_in
            ys_in = F.glu(self.fc_glu(ys_in), dim=-1)
            if self.residual:
                ys_in += residual
        logits = self.output(ys_in)

        # Compute XE sequence loss
        loss = F.cross_entropy(logits.view((-1, logits.size(2))),
                               ys_out.contiguous().view(-1),
                               ignore_index=self.pad, size_average=True)

        # Compute token-level accuracy in teacher-forcing
        pad_pred = logits.view(ys_out.size(0), ys_out.size(1), logits.size(-1)).argmax(2)
        mask = ys_out != self.pad
        numerator = torch.sum(pad_pred.masked_select(mask) == ys_out.masked_select(mask))
        denominator = torch.sum(mask)
        acc = float(numerator) * 100 / float(denominator)

        observation = {'loss.rnnlm': loss.item(),
                       'acc.rnnlm': acc,
                       'ppl.rnnlm': math.exp(loss.item())}

        return loss, observation

    def predict(self, y, state):
        """Predict a token per step for ASR decoding.

        Args:
            y (): `[B, emb_dim]`
            state (list):
        Returns:
            logits_step (FloatTensor):
            y (FloatTensor):
            state ():

        """
        if state is None:
            state = [None] * self.nlayers

        # Path through RNN
        residual = None
        for l in range(self.nlayers):
            y, state[l] = self.rnn[l](y, hx=state[l])
            y = self.dropout[l](y)

            # Residual connection
            if self.residual and l > 0:
                y += residual
            residual = y

        if self.use_glu:
            if self.residual:
                residual = y
            y = F.glu(self.fc_glu(y), dim=-1)
            if self.residual:
                y += residual
        logits_step = self.output(y)

        return logits_step, y, state

    def initialize_hidden(self, bs):
        """Initialize hidden states.

        Args:
            bs (int): the size of mini-batch
        Returns:
            hx_list (list of FloatTensor):
            cx_list (list of FloatTensor):

        """
        if self.rnn_type == 'lstm':
            hx_list = [torch.zeros(bs, self.nunits, dtype=torch.float32).cuda(self.device_id)
                       for l in range(self.nlayers)]
            cx_list = [torch.zeros(bs, self.nunits, dtype=torch.float32).cuda(self.device_id)
                       for l in range(self.nlayers)]
        elif self.rnn_type == 'gru':
            hx_list = [torch.zeros(bs, self.nunits, dtype=torch.float32).cuda(self.device_id)
                       for l in range(self.nlayers)]
            cx_list = None
        return hx_list, cx_list
