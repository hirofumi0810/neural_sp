#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Sequence-level RNN language model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.base import ModelBase
from neural_sp.models.model_utils import Embedding
from neural_sp.models.model_utils import LinearND
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


class SeqRNNLM(ModelBase):
    """Sequence-level RNN language model implemented by torch.nn.LSTM (or GRU)."""

    def __init__(self, args):

        super(ModelBase, self).__init__()

        self.emb_dim = args.emb_dim
        self.rnn_type = args.rnn_type
        assert args.rnn_type in ['lstm', 'gru']
        self.n_units = args.n_units
        self.n_layers = args.n_layers
        self.tie_embedding = args.tie_embedding
        self.residual = args.residual
        self.use_glu = args.use_glu
        self.backward = args.backward

        self.vocab = args.vocab
        self.sos = 2   # NOTE: the same as <eos>
        self.eos = 2
        self.pad = 3
        # NOTE: reserved in advance

        # for cache
        self.cache_theta = 0.2  # smoothing parameter
        self.cache_lambda = 0.2  # cache weight
        self.cache_keys = []
        self.cache_values = []
        self.cache_attn = []

        self.embed = Embedding(vocab=self.vocab,
                               emb_dim=args.emb_dim,
                               dropout=args.dropout_emb,
                               ignore_index=self.pad)

        self.fast_impl = False
        if args.n_projs == 0 and not args.residual:
            self.fast_impl = True
            if 'lstm' in args.rnn_type:
                rnn = nn.LSTM
            elif 'gru' in args.rnn_type:
                rnn = nn.GRU
            else:
                raise ValueError('rnn_type must be "(b)lstm" or "(b)gru".')

            self.rnn = rnn(args.emb_dim, args.n_units, args.n_layers,
                           bias=True,
                           batch_first=True,
                           dropout=args.dropout_hidden,
                           bidirectional=False)
            # NOTE: pytorch introduces a dropout layer on the outputs of each layer EXCEPT the last layer
            rnn_idim = args.n_units
            self.dropout_top = nn.Dropout(p=args.dropout_hidden)
        else:
            self.rnn = torch.nn.ModuleList()
            self.dropout = torch.nn.ModuleList()
            if args.n_projs > 0:
                self.proj = torch.nn.ModuleList()
            rnn_idim = args.emb_dim
            for l in range(args.n_layers):
                self.rnn += [getattr(nn, args.rnn_type.upper())(
                    rnn_idim, args.n_units, 1,
                    bias=True,
                    batch_first=True,
                    dropout=0,
                    bidirectional=False)]
                self.dropout += [nn.Dropout(p=args.dropout_hidden)]
                rnn_idim = args.n_units

                if l != self.n_layers - 1 and args.n_projs > 0:
                    self.proj += [LinearND(args.n_units, args.n_projs)]
                    rnn_idim = args.n_projs

        if self.use_glu:
            self.fc_glu = LinearND(rnn_idim, rnn_idim * 2,
                                   dropout=args.dropout_hidden)

        self.output = LinearND(rnn_idim, self.vocab,
                               dropout=args.dropout_out)
        # NOTE: include bias even when tying weights

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if args.tie_embedding:
            if args.n_units != args.emb_dim:
                raise ValueError('When using the tied flag, n_units must be equal to emb_dim.')
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

    def forward(self, ys, hidden=None, reporter=None, is_eval=False, n_caches=0):
        """Forward computation.

        Args:
            ys (list): A list of length `[B]`, which contains arrays of size `[L]`
            hidden (tuple or list): (h_n, c_n) or (hxs, cxs)
            reporter ():
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
            n_caches (int):
        Returns:
            loss (FloatTensor): `[1]`
            hidden (tuple or list): (h_n, c_n) or (hxs, cxs)
            reporter ():

        """
        if is_eval:
            self.eval()
            with torch.no_grad():
                loss, hidden, observation = self._forward(ys, hidden, n_caches)
        else:
            self.train()
            loss, hidden, observation = self._forward(ys, hidden)

        # Report here
        if reporter is not None:
            reporter.add(observation, is_eval)

        return loss, hidden, reporter

    def _forward(self, ys, hidden, n_caches=0):
        if self.backward:
            ys = [np2tensor(np.fromiter(y[::-1], dtype=np.int64), self.device_id).long() for y in ys]
        else:
            ys = [np2tensor(np.fromiter(y, dtype=np.int64), self.device_id).long() for y in ys]

        ys = pad_list(ys, self.pad)
        ys_in = ys[:, :-1]
        ys_out = ys[:, 1:]

        # Path through embedding
        ys_in = self.embed(ys_in)

        if hidden is None:
            hidden = self.initialize_hidden(ys.size(0))

        residual = None
        if self.fast_impl:
            ys_in, hidden = self.rnn(ys_in, hx=hidden)
            ys_in = self.dropout_top(ys_in)
        else:
            for l in range(self.n_layers):
                # Path through RNN
                if self.rnn_type == 'lstm':
                    ys_in, (hidden[0][l], hidden[1][l]) = self.rnn[l](ys_in, hx=(hidden[0][l], hidden[1][l]))
                elif self.rnn_type == 'gru':
                    ys_in, hidden[0][l] = self.rnn[l](ys_in, hx=hidden[0][l])
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
        if n_caches > 0 and len(self.cache_keys) > 0:
            assert ys_out.size(1) == 1
            assert ys_out.size(0) == 1
            probs = F.softmax(logits, dim=-1)

            cache_probs = torch.zeros_like(probs)

            # Truncate cache
            self.cache_keys = self.cache_keys[-n_caches:]  # list of `[B, 1]`
            self.cache_values = self.cache_values[-n_caches:]  # list of `[B, 1, n_units]`

            # Compute inner-product over caches
            cache_attn = F.softmax(self.cache_theta * torch.matmul(
                torch.cat(self.cache_values, dim=1),
                ys_in.transpose(1, 2)).squeeze(2), dim=1)

            # For visualization
            if len(self.cache_keys) == n_caches:
                self.cache_attn += [cache_attn.cpu().numpy()]
                self.cache_attn = self.cache_attn[-n_caches:]

            # Sum all probabilities
            for offset, idx in enumerate(self.cache_keys):
                cache_probs[:, :, idx] += cache_attn[:, offset]
            probs = (1 - self.cache_lambda) * probs + self.cache_lambda * cache_probs
            loss = (-torch.log(probs[:, :, ys_out[:, -1]]))
        else:
            loss = F.cross_entropy(logits.view((-1, logits.size(2))),
                                   ys_out.contiguous().view(-1),
                                   ignore_index=self.pad, size_average=True)

        if n_caches > 0:
            # Register to cache
            # self.cache_keys += [ys_out[:, -1].cpu().numpy()]
            self.cache_keys += [ys_out[0, -1].item()]
            self.cache_values += [ys_in]

        # Compute token-level accuracy in teacher-forcing
        pad_pred = logits.view(ys_out.size(0), ys_out.size(1), logits.size(-1)).argmax(2)
        mask = ys_out != self.pad
        numerator = (pad_pred.masked_select(mask) == ys_out.masked_select(mask)).sum()
        denominator = mask.sum()
        acc = float(numerator) * 100 / float(denominator)

        observation = {'loss.rnnlm': loss.item(),
                       'acc.rnnlm': acc,
                       'ppl.rnnlm': np.exp(loss.item())}

        return loss, hidden, observation

    def predict(self, y, hidden):
        """Predict a token per step for ASR decoding.

        Args:
            y (FloatTensor): `[B, 1, emb_dim]`
            hidden (tuple or list): (h_n, c_n) or (hxs, cxs)
        Returns:
            logits_step (FloatTensor): `[B, 1, vocab]`
            y (FloatTensor): `[B, 1, n_units]`
            hidden (tuple or list): (h_n, c_n) or (hxs, cxs)

        """
        if hidden[0] is None:
            hidden = self.initialize_hidden(y.size(0))

        residual = None
        if self.fast_impl:
            y, hidden = self.rnn(y, hx=hidden)
            y = self.dropout_top(y)
        else:
            for l in range(self.n_layers):
                # Path through RNN
                if self.rnn_type == 'lstm':
                    y, (hidden[0][l], hidden[1][l]) = self.rnn[l](y, hx=(hidden[0][:][l], hidden[1][:][l]))
                elif self.rnn_type == 'gru':
                    y, hidden[0][l] = self.rnn[l](y, hx=hidden[0][:][l])
                y = self.dropout[l](y)

                # Residual connection
                if self.residual and l > 0:
                    y += residual
                residual = y
                # NOTE: Exclude residual connection from the raw inputs

        if self.use_glu:
            if self.residual:
                residual = y
            y = F.glu(self.fc_glu(y), dim=-1)
            if self.residual:
                y += residual
        logits_step = self.output(y)

        return logits_step, y, hidden

    def initialize_hidden(self, bs):
        """Initialize hidden states."""
        w = next(self.parameters())

        if self.fast_impl:
            # return None
            h_n = w.new_zeros(self.n_layers, bs, self.n_units)
            if self.rnn_type == 'lstm':
                c_n = w.new_zeros(self.n_layers, bs, self.n_units)
            elif self.rnn_type == 'gru':
                c_n = None
            return (h_n, c_n)
        else:
            hxs, cxs = [], []
            for l in range(self.n_layers):
                # h_l = None
                h_l = w.new_zeros(1, bs, self.n_units)
                if self.rnn_type == 'lstm':
                    c_l = w.new_zeros(1, bs, self.n_units)
                elif self.rnn_type == 'gru':
                    c_l = None
                hxs.append(h_l)
                cxs.append(c_l)
            return (hxs, cxs)

    def repackage_hidden(self, hidden):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if self.fast_impl:
            return hidden.detach()
        else:
            if self.rnn_type == 'lstm':
                return ([h.detach() for h in hidden[0]], [c.detach() for c in hidden[1]])
            else:
                return ([h.detach() for h in hidden[0]], hidden[1])
