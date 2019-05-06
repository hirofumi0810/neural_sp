#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Recurrent neural network language model (RNNLM)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.base import ModelBase
from neural_sp.models.modules.embedding import Embedding
from neural_sp.models.modules.linear import LinearND
from neural_sp.models.torch_utils import compute_accuracy
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


class RNNLM(ModelBase):
    """RNN language model."""

    def __init__(self, args):

        super(ModelBase, self).__init__()

        self.emb_dim = args.emb_dim
        self.rnn_type = args.lm_type
        assert args.lm_type in ['lstm', 'gru']
        self.n_units = args.n_units
        self.n_layers = args.n_layers
        self.residual = args.residual
        self.use_glu = args.use_glu
        self.backward = args.backward

        self.vocab = args.vocab
        self.eos = 2
        self.pad = 3
        # NOTE: reserved in advance

        # for cache
        self.cache_theta = 0.2  # smoothing parameter
        self.cache_lambda = 0.2  # cache weight
        self.cache_ids = []
        self.cache_keys = []
        self.cache_attn = []

        self.embed = Embedding(vocab=self.vocab,
                               emb_dim=args.emb_dim,
                               dropout=args.dropout_emb,
                               ignore_index=self.pad)

        self.fast_impl = False
        if args.n_projs == 0 and not args.residual:
            self.fast_impl = True
            if 'lstm' in args.lm_type:
                rnn = nn.LSTM
            elif 'gru' in args.lm_type:
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
                self.rnn += [getattr(nn, args.lm_type.upper())(
                    rnn_idim, args.n_units, 1,
                    bias=True,
                    batch_first=True,
                    dropout=0,
                    bidirectional=False)]
                self.dropout += [nn.Dropout(p=args.dropout_hidden)]
                rnn_idim = args.n_units

                if l != self.n_layers - 1 and args.n_projs > 0:
                    self.proj += [LinearND(rnn_idim, args.n_projs)]
                    rnn_idim = args.n_projs

        if self.use_glu:
            self.fc_glu = LinearND(rnn_idim, rnn_idim * 2,
                                   dropout=args.dropout_hidden)

        if args.adaptive_softmax:
            self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
                rnn_idim, self.vocab,
                # cutoffs=[self.vocab // 10, 3 * self.vocab // 10],
                cutoffs=[self.vocab // 25, self.vocab // 5],
                div_value=4.0)
            self.output = None
        else:
            self.adaptive_softmax = None
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

        # Initialize parameters
        self.reset_parameters(args.param_init, dist=args.param_init_dist)

        # Initialize bias vectors with zero
        self.reset_parameters(0, dist='constant', keys=['bias'])

        # Recurrent weights are orthogonalized
        if args.rec_weight_orthogonal:
            self.reset_parameters(args.param_init, dist='orthogonal',
                                  keys=['rnn', 'weight'])

        # Initialize bias in forget gate with 1
        # self.init_forget_gate_bias_with_one()

    def forward(self, ys, hidden=None, reporter=None, is_eval=False, n_caches=0,
                ylens=[]):
        """Forward computation.

        Args:
            ys (list): A list of length `[B]`, which contains arrays of size `[L]`
            hidden (tuple or list): (h_n, c_n) or (hxs, cxs)
            reporter ():
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
            n_caches (int):
            ylens (list): not used
        Returns:
            loss (FloatTensor): `[1]`
            hidden (tuple or list): (h_n, c_n) or (hxs, cxs)
            reporter ():

        """
        if is_eval:
            self.eval()
            with torch.no_grad():
                loss, hidden, reporter = self._forward(ys, hidden, reporter, n_caches)
        else:
            self.train()
            loss, hidden, reporter = self._forward(ys, hidden, reporter)

        return loss, hidden, reporter

    def _forward(self, ys, hidden, reporter, n_caches=0):
        ys = [np2tensor(y[::-1] if self.backward else y, self.device_id).long() for y in ys]
        ys = pad_list(ys, self.pad)
        ys_in = ys[:, :-1]
        ys_out = ys[:, 1:]

        lmout, hidden = self.decode(self.encode(ys_in), hidden)
        if self.adaptive_softmax is None:
            logits = self.generate(lmout)
        else:
            logits = lmout

        # Compute XE sequence loss
        if n_caches > 0 and len(self.cache_ids) > 0:
            assert ys_out.size(1) == 1
            assert ys_out.size(0) == 1
            if self.adaptive_softmax is None:
                probs = F.softmax(logits, dim=-1)
            else:
                probs = self.adaptive_softmax.log_prob(logits).exp()
            cache_probs = probs.new_zeros(probs.size())

            # Truncate cache
            self.cache_ids = self.cache_ids[-n_caches:]  # list of `[B, 1]`
            self.cache_keys = self.cache_keys[-n_caches:]  # list of `[B, 1, n_units]`

            # Compute inner-product over caches
            cache_attn = F.softmax(self.cache_theta * torch.matmul(
                torch.cat(self.cache_keys, dim=1),
                lmout.transpose(1, 2)).squeeze(2), dim=1)

            # For visualization
            if len(self.cache_ids) == n_caches:
                self.cache_attn += [cache_attn.cpu().numpy()]
                self.cache_attn = self.cache_attn[-n_caches:]

            # Sum all probabilities
            for offset, idx in enumerate(self.cache_ids):
                cache_probs[:, :, idx] += cache_attn[:, offset]
            probs = (1 - self.cache_lambda) * probs + self.cache_lambda * cache_probs
            loss = -torch.log(probs[:, :, ys_out[:, -1]])
        else:
            if self.adaptive_softmax is None:
                loss = F.cross_entropy(logits.view((-1, logits.size(2))),
                                       ys_out.contiguous().view(-1),
                                       ignore_index=self.pad, size_average=True)
            else:
                loss = self.adaptive_softmax(logits.view((-1, logits.size(2))),
                                             ys_out.contiguous().view(-1)).loss

        if n_caches > 0:
            # Register to cache
            self.cache_ids += [ys_out[0, -1].item()]
            self.cache_keys += [lmout]

        # Compute token-level accuracy in teacher-forcing
        if self.adaptive_softmax is None:
            acc = compute_accuracy(logits, ys_out, pad=self.pad)
        else:
            acc = compute_accuracy(self.adaptive_softmax.log_prob(
                logits.view((-1, logits.size(2)))), ys_out, pad=self.pad)

        observation = {'loss.lm': loss.item(),
                       'acc.lm': acc,
                       'ppl.lm': np.exp(loss.item())}

        # Report here
        if reporter is not None:
            is_eval = not self.training
            reporter.add(observation, is_eval)

        return loss, hidden, reporter

    def encode(self, ys):
        """Encode function.

        Args:
            ys (LongTensor): `[B, L]`
        Returns:
            ys (FloatTensor): `[B, L, emb_dim]`

        """
        return self.embed(ys)

    def decode(self, ys_emb, hidden):
        """Decode function.

        Args:
            ys_emb (FloatTensor): `[B, L, emb_dim]`
            hidden (tuple or list): (h_n, c_n) or (hxs, cxs)
        Returns:
            ys_emb (FloatTensor): `[B, L, n_units]`
            hidden (tuple or list): (h_n, c_n) or (hxs, cxs)

        """
        if hidden is None or hidden[0] is None:
            hidden = self.initialize_hidden(ys_emb.size(0))

        residual = None
        if self.fast_impl:
            # Path through RNN
            ys_emb, hidden = self.rnn(ys_emb, hx=hidden)
            ys_emb = self.dropout_top(ys_emb)
        else:
            new_hxs, new_cxs = [], []
            for l in range(self.n_layers):
                # Path through RNN
                if self.rnn_type == 'lstm':
                    ys_emb, (h_l, c_l) = self.rnn[l](ys_emb, hx=(hidden[0][l:l + 1], hidden[1][l:l + 1]))
                    new_cxs.append(c_l)
                elif self.rnn_type == 'gru':
                    ys_emb, h_l = self.rnn[l](ys_emb, hx=hidden[0][l:l + 1])
                new_hxs.append(h_l)
                ys_emb = self.dropout[l](ys_emb)

                # Residual connection
                if self.residual and l > 0:
                    ys_emb += residual
                residual = ys_emb
                # NOTE: Exclude residual connection from the raw inputs

            # Repackage
            new_hxs = torch.cat(new_hxs, dim=0)
            if self.rnn_type == 'lstm':
                new_cxs = torch.cat(new_cxs, dim=0)
            hidden = (new_hxs, new_cxs)

        if self.use_glu:
            if self.residual:
                residual = ys_emb
            ys_emb = F.glu(self.fc_glu(ys_emb), dim=-1)
            if self.residual:
                ys_emb += residual

        return ys_emb, hidden

    def generate(self, hidden):
        """Generate function.

        Args:
            hidden (FloatTensor): `[B, T, n_units]`
        Returns:
            logits (FloatTensor): `[B, T, vocab]`

        """
        return self.output(hidden)

    def initialize_hidden(self, batch_size):
        """Initialize hidden states.

        Args:
            batch_size (int):
        Returns:
            hidden (tuple):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`

        """
        w = next(self.parameters())
        if self.fast_impl:
            h_n = w.new_zeros(self.n_layers, batch_size, self.n_units)
            c_n = None
            if self.rnn_type == 'lstm':
                c_n = w.new_zeros(self.n_layers, batch_size, self.n_units)
            return (h_n, c_n)
        else:
            hxs, cxs = [], []
            for l in range(self.n_layers):
                if self.rnn_type == 'lstm':
                    cxs.append(w.new_zeros(1, batch_size, self.n_units))
                hxs.append(w.new_zeros(1, batch_size, self.n_units))
            hxs = torch.cat(hxs, dim=0)
            if self.rnn_type == 'lstm':
                cxs = torch.cat(cxs, dim=0)
            return (hxs, cxs)

    def repackage_hidden(self, hidden):
        """Wraps hidden states in new Tensors, to detach them from their history.

        Args:
            hidden (tuple):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`
        Returns:
            hidden (tuple):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`

        """
        hxs, cxs = hidden
        hxs = hxs.detach()
        if self.rnn_type == 'lstm':
            cxs = cxs.detach()
        return (hxs, cxs)
