#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN language model."""

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


class RNNLM(ModelBase):
    """RNN language model."""

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
                           bidirectional=self.bidirectional)
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
                    rnn_cell = nn.LSTMCell
                elif args.rnn_type == 'gru':
                    rnn_cell = nn.GRUCell

                self.rnn += [rnn_cell(rnn_idim, args.nunits, 1)]
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
        self.init_weights(args.param_init, dist=args.param_init_dist, ignore_keys=['bias'])

        # Initialize all biases with 0
        self.init_weights(0, dist='constant', keys=['bias'])

        # Recurrent weights are orthogonalized
        if args.rec_weight_orthogonal:
            self.init_weights(args.param_init, dist='orthogonal',
                              keys=[args.rnn_type, 'weight'], ignore_keys=['bias'])

        # Initialize bias in forget gate with 1
        self.init_forget_gate_bias_with_one()

    def forward(self, ys, state, reporter=None, is_eval=False):
        """Forward computation.

        Args:
            ys (np.array): A tensor of of size `[B, 2]`
            state (list): list of (hx_list, cx_list)
                hx_list (list of FloatTensor):
                cx_list (list of FloatTensor):
            reporter ():
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (FloatTensor): `[1]`
            state (list): list of (hx_list, cx_list)
                hx_list (list of FloatTensor):
                cx_list (list of FloatTensor):
            reporter ():

        """
        if is_eval:
            self.eval()
            with torch.no_grad():
                loss, observation = self._forward(ys, state)
        else:
            self.train()
            loss, observation = self._forward(ys, state)

        # Report here
        if reporter is not None:
            reporter.add(observation, is_eval)

        return loss, observation

    def _forward(self, ys, state):
        if self.backward:
            raise NotImplementedError()
            # TODO(hirofumi): reverse the order out of the model
        else:
            ys = [np2tensor(np.fromiter(y, dtype=np.int64), self.device_id).long() for y in ys]
            ys = pad_list(ys, self.pad)

            ys_in = ys[:, :-1]  # B*1
            ys_out = ys[:, 1:]  # B*1

        # Path through embedding
        ys_in = self.embed(ys_in)

        if state is None:
            hx_list, cx_list = self.initialize_hidden(bs=ys.shape[0])
        else:
            hx_list, cx_list = state

        # Path through RNN
        residual = None
        if self.fast_impl:
            if self.rnn_type == 'lstm':
                hx_list[-1], cx_list[-1] = self.rnn(ys_in, (hx_list[0], cx_list[0]))
            elif self.rnn_type == 'gru':
                hx_list[-1] = self.rnn(ys_in, hx_list[0])
            hx_list[-1] = self.dropout_top(hx_list[-1])
        else:
            for l in range(self.nlayers):
                if self.rnn_type == 'lstm':
                    if l == 0:
                        hx_list[0], cx_list[0] = self.rnn[l](ys_in, (hx_list[0], cx_list[0]))
                    else:
                        hx_list[l], cx_list[l] = self.rnn[l](hx_list[l - 1], (hx_list[l], cx_list[l]))
                elif self.rnn_type == 'gru':
                    if l == 0:
                        hx_list[0] = self.rnn[l](ys_in, hx_list[0])
                    else:
                        hx_list[l] = self.rnn[l](hx_list[l - 1], hx_list[l])
                hx_list[l] = self.dropout[l](hx_list[l])

                # Residual connection
                if self.residual and l > 0:
                    hx_list[l] += residual
                    residual = hx_list[l]
        output = hx_list[-1].unsqueeze(1)

        if self.use_glu:
            if self.residual:
                residual = output
            output = F.glu(self.fc_glu(output), dim=-1)
            if self.residual:
                output += residual
        logits = self.output(output)

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

        observation = {'loss': loss.item(),
                       'acc': acc,
                       'ppl': math.exp(loss.item())}

        return loss, (hx_list, cx_list), observation

    def predict(self, y, state):
        """

        Args:
            y (): `[B, emb_dim]`
            state (list):
        Returns:
            logits_step (FloatTensor):
            y ():
            state (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of FloatTensor):
                cx_list (list of FloatTensor):

        """
        if state[0] is None:
            hx_list, cx_list = self.initialize_hidden(bs=1)
        else:
            hx_list, cx_list = state

        # Path through RNN
        residual = None
        for l in range(self.nlayers):
            if self.rnn_type == 'lstm':
                if l == 0:
                    hx_list[0], cx_list[0] = self.rnn[l](
                        y, (hx_list[0], cx_list[0]))
                else:
                    hx_list[l], cx_list[l] = self.rnn[l](
                        hx_list[l - 1], (hx_list[l], cx_list[l]))
            elif self.rnn_type == 'gru':
                if l == 0:
                    hx_list[0] = self.rnn[l](y, hx_list[0])
                else:
                    hx_list[l] = self.rnn[l](hx_list[l - 1], hx_list[l])

            # Dropout for hidden-hidden or hidden-output connection
            hx_list[l] = self.dropout[l](hx_list[l])

            # Residual connection
            if self.residual and residual is not None:
                hx_list[l] += residual
            residual = hx_list[l]
        output = hx_list[-1].unsqueeze(1)

        if self.use_glu:
            if self.residual:
                residual = output
            output = F.glu(self.fc_glu(output), dim=-1)
            if self.residual:
                output += residual
        logits_step = self.output(output)

        return logits_step, y, (hx_list, cx_list)

    def initialize_hidden(self, bs):
        """Initialize hidden states.

        Args:
            bs (int): the size of mini-batch
        Returns:
            hx_list (list of FloatTensor):
            cx_list (list of FloatTensor):

        """
        zero_state = torch.zeros(bs, self.nunits).float()
        if self.device_id >= 0:
            zero_state = zero_state.cuda(self.device_id)

        if self.rnn_type == 'lstm':
            hx_list = [zero_state] * self.nlayers
            cx_list = [zero_state] * self.nlayers
        elif self.rnn_type == 'gru':
            hx_list = [zero_state] * self.nlayers
            cx_list = None

        return hx_list, cx_list

    def repackage_hidden(self, state):
        """Initialize hidden states.

        Args:
            state (list):
                hx_list (list of FloatTensor):
                cx_list (list of FloatTensor):
        Returns:
            state (list):
                hx_list (list of FloatTensor):
                cx_list (list of FloatTensor):

        """
        hx_list, cx_list = state
        if self.rnn_type == 'lstm':
            hx_list = [h.detach() for h in hx_list]
            cx_list = [c.detach() for c in cx_list]
        else:
            hx_list = [h.detach() for h in hx_list]
        return (hx_list, cx_list)

    def copy_from_seqrnnlm(self, rnnlm):
        """Copy parameters from sequene-level RNNLM.

        Args:
            rnnlm ():

        """
        for l in range(self.nlayers):
            self.rnn[l].weight_ih.data = rnnlm.rnn[l].weight_ih_l0.data
            self.rnn[l].weight_hh.data = rnnlm.rnn[l].weight_hh_l0.data
            self.rnn[l].bias_ih.data = rnnlm.rnn[l].bias_ih_l0.data
            self.rnn[l].bias_hh.data = rnnlm.rnn[l].bias_hh_l0.data
        self.output.fc.weight.data = rnnlm.output.fc.weight.data
        self.output.fc.bias.data = rnnlm.output.fc.bias.data
