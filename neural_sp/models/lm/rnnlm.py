#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Recurrent neural network language model (RNNLM)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn

from neural_sp.models.lm.lm_base import LMBase
from neural_sp.models.modules.glu import LinearGLUBlock
from neural_sp.models.torch_utils import repeat

logger = logging.getLogger(__name__)


class RNNLM(LMBase):
    """RNN language model."""

    def __init__(self, args, save_path=None):

        super(LMBase, self).__init__()
        logger.info(self.__class__.__name__)

        self.lm_type = args.lm_type
        self.save_path = save_path

        self.emb_dim = args.emb_dim
        self.rnn_type = args.lm_type
        assert args.lm_type in ['lstm', 'gru']
        self.n_units = args.n_units
        self.n_projs = args.n_projs
        self.n_layers = args.n_layers
        self.residual = args.residual
        self.n_units_cv = args.n_units_null_context
        self.lsm_prob = args.lsm_prob

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

        self.embed = nn.Embedding(self.vocab, args.emb_dim, padding_idx=self.pad)
        self.dropout_embed = nn.Dropout(p=args.dropout_in)

        rnn = nn.LSTM if args.lm_type == 'lstm' else nn.GRU
        self.rnn = nn.ModuleList()
        self.dropout = nn.Dropout(p=args.dropout_hidden)
        if args.n_projs > 0:
            self.proj = repeat(nn.Linear(args.n_units, args.n_projs), args.n_layers)
        rnn_idim = args.emb_dim + args.n_units_null_context
        for l in range(args.n_layers):
            self.rnn += [rnn(rnn_idim, args.n_units, 1, batch_first=True)]
            rnn_idim = args.n_units
            if args.n_projs > 0:
                rnn_idim = args.n_projs

        self.glu = None
        if args.use_glu:
            self.glu = LinearGLUBlock(rnn_idim)

        self._odim = rnn_idim

        self.adaptive_softmax = None
        self.output_proj = None
        self.output = None
        if args.adaptive_softmax:
            self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
                rnn_idim, self.vocab,
                # cutoffs=[self.vocab // 10, 3 * self.vocab // 10],
                cutoffs=[self.vocab // 25, self.vocab // 5],
                div_value=4.0)
        else:
            self.adaptive_softmax = None
            if args.tie_embedding:
                if rnn_idim != args.emb_dim:
                    self.output_proj = nn.Linear(rnn_idim, args.emb_dim)
                    rnn_idim = args.emb_dim
                    self._odim = rnn_idim
                self.output = nn.Linear(rnn_idim, self.vocab)
                self.output.weight = self.embed.weight
            else:
                self.output = nn.Linear(rnn_idim, self.vocab)

        self.reset_parameters(args.param_init)

        # Recurrent weights are orthogonalized
        if args.rec_weight_orthogonal:
            self.reset_parameters(args.param_init, dist='orthogonal',
                                  keys=['rnn', 'weight'])

        # Initialize bias in forget gate with 1
        # self.init_forget_gate_bias_with_one()

    @property
    def output_dim(self):
        return self._odim

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0.)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
            elif p.dim() == 2:
                nn.init.uniform_(p, a=-param_init, b=param_init)
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', param_init))
            else:
                raise ValueError(n)

    def decode(self, ys, state, mems=None, cache=None, incremental=False):
        """Decode function.

        Args:
            ys (FloatTensor): `[B, L]`
            state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`
            cache: dummy interfance for TransformerLM/TransformerXL
            incremental: dummy interfance for TransformerLM/TransformerXL
        Returns:
            logits (FloatTensor): `[B, L, vocab]`
            ys_emb (FloatTensor): `[B, L, n_units]` (for cache)
            new_state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`
            new_mems: dummy interfance for TransformerXL

        """
        bs, ymax = ys.size()
        ys_emb = self.dropout_embed(self.embed(ys.long()))

        if state is None:
            state = self.zero_state(bs)
        new_state = {'hxs': None, 'cxs': None}

        # for ASR decoder pre-training
        if self.n_units_cv > 0:
            cv = ys.new_zeros(bs, ymax, self.n_units_cv)
            ys_emb = torch.cat([ys_emb, cv], dim=-1)

        residual = None
        new_hxs, new_cxs = [], []
        for l in range(self.n_layers):
            self.rnn[l].flatten_parameters()  # for multi-GPUs

            # Path through RNN
            if self.rnn_type == 'lstm':
                ys_emb, (h, c) = self.rnn[l](ys_emb, hx=(state['hxs'][l:l + 1],
                                                         state['cxs'][l:l + 1]))
                new_cxs.append(c)
            elif self.rnn_type == 'gru':
                ys_emb, h = self.rnn[l](ys_emb, hx=state['hxs'][l:l + 1])
            new_hxs.append(h)
            ys_emb = self.dropout(ys_emb)
            if self.n_projs > 0:
                ys_emb = torch.tanh(self.proj[l](ys_emb))

            # Residual connection
            if self.residual and l > 0:
                ys_emb = ys_emb + residual
            residual = ys_emb
            # NOTE: Exclude residual connection from the raw inputs

        # Repackage
        new_state['hxs'] = torch.cat(new_hxs, dim=0)
        if self.rnn_type == 'lstm':
            new_state['cxs'] = torch.cat(new_cxs, dim=0)

        if self.glu is not None:
            if self.residual:
                residual = ys_emb
            ys_emb = self.glu(ys_emb)
            if self.residual:
                ys_emb = ys_emb + residual

        if self.adaptive_softmax is None:
            if self.output_proj is not None:
                ys_emb = self.output_proj(ys_emb)
            logits = self.output(ys_emb)
        else:
            logits = ys_emb

        return logits, ys_emb, new_state

    def zero_state(self, batch_size):
        """Initialize hidden state.

        Args:
            batch_size (int): batch size
        Returns:
            state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`

        """
        w = next(self.parameters())
        state = {'hxs': None, 'cxs': None}
        state['hxs'] = w.new_zeros(self.n_layers, batch_size, self.n_units)
        if self.rnn_type == 'lstm':
            state['cxs'] = w.new_zeros(self.n_layers, batch_size, self.n_units)
        return state

    def repackage_state(self, state):
        """Wraps hidden states in new Tensors, to detach them from their history.

        Args:
            state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`
        Returns:
            state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`

        """
        state['hxs'] = state['hxs'].detach()
        if self.rnn_type == 'lstm':
            state['cxs'] = state['cxs'].detach()
        return state
