#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN language model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.base import ModelBase
from neural_sp.models.linear import Embedding
from neural_sp.models.linear import LinearND
from neural_sp.models.utils import np2var
from neural_sp.models.utils import pad_list


class RNNLM(ModelBase):
    """RNN language model."""

    def __init__(self, args):

        super(ModelBase, self).__init__()

        self.emb_dim = args.emb_dim
        self.rnn_type = args.rnn_type
        assert args.rnn_type in ['lstm', 'gru']
        self.num_units = args.num_units
        self.num_layers = args.num_layers
        self.tie_weights = args.tie_weights
        self.residual = args.residual
        self.backward = args.backward

        self.num_classes = args.num_classes
        self.blank = 0
        self.unk = 1
        self.sos = 2
        self.eos = 3
        self.pad = 4
        # NOTE: these are reserved in advance

        # TODO(hirofumi): dropout for inputs

        self.emb = Embedding(num_classes=self.num_classes,
                             emb_dim=args.emb_dim,
                             dropout=args.dropout_emb,
                             ignore_index=self.pad)
        # TODO(hirofumi): remove dropout_emb

        for i_l in range(args.num_layers):
            if i_l == 0:
                enc_in_size = args.emb_dim
            else:
                enc_in_size = args.num_units

            if args.rnn_type == 'lstm':
                rnn_i = nn.LSTM(enc_in_size,
                                hidden_size=args.num_units,
                                num_layers=1,
                                bias=True,
                                batch_first=True,
                                dropout=0,
                                bidirectional=False)
            elif args.rnn_type == 'gru':
                rnn_i = nn.GRU(enc_in_size,
                               hidden_size=args.num_units,
                               num_layers=1,
                               bias=True,
                               batch_first=True,
                               dropout=0,
                               bidirectional=False)

            setattr(self, args.rnn_type + '_l' + str(i_l), rnn_i)

            # Dropout for hidden-hidden or hidden-output connection
            setattr(self, 'dropout_l' + str(i_l), nn.Dropout(p=args.dropout_hidden))

        self.output = LinearND(args.num_units, self.num_classes,
                               dropout=args.dropout_out)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if args.tie_weights:
            if args.num_units != args.emb_dim:
                raise ValueError('When using the tied flag, num_units must be equal to emb_dim.')
            self.output.fc.weight = self.emb.emb.weight

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

    def forward(self, ys, is_eval=False):
        """Forward computation.

        Args:
            ys (np.array): A tensor of of size `[B, T]`
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (torch.autograd.Variable(float)): A tensor of size `[1]`
            acc (float): Token-level accuracy in teacher-forcing

        """
        if is_eval:
            self.eval()
        else:
            self.train()

        if self.backward:
            raise NotImplementedError()
            # TODO(hirofumi): reverse the order out of the model
        else:
            ys = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long() for y in ys]
            ys = pad_list(ys, self.pad)

            ys_in = ys[:, :-1]
            ys_out = ys[:, 1:]

        # Path through embedding
        ys_in = self.emb(ys_in)

        res_out_prev = None
        for i_l in range(self.num_layers):
            # Path through RNN
            ys_in, _ = getattr(self, self.rnn_type + '_l' + str(i_l))(ys_in, hx=None)

            # Dropout for hidden-hidden or hidden-output connection
            ys_in = getattr(self, 'dropout_l' + str(i_l))(ys_in)

            # Residual connection
            if self.residual and res_out_prev is not None:
                ys_in += res_out_prev
            res_out_prev = ys_in

        logits = self.output(ys_in)

        # Compute XE sequence loss
        loss = F.cross_entropy(input=logits.view((-1, logits.size(2))),
                               target=ys_out.contiguous().view(-1),
                               ignore_index=-1, size_average=True)

        # Compute token-level accuracy in teacher-forcing
        pad_pred = logits.data.view(ys_out.size(0), ys_out.size(1), logits.size(-1)).max(2)[1]
        mask = ys_out.data != self.pad
        numerator = torch.sum(pad_pred.masked_select(mask) == ys_out.data.masked_select(mask))
        denominator = torch.sum(mask)
        acc = float(numerator) / float(denominator)

        # return loss, acc, (hx_list, cx_list)
        return loss, acc

    def predict(self, y, state):
        """

        Args:
            y (int):
            state (list):
        Returns:
            logits_step ():
            out ():
            state ():

        """
        # Start from None
        if state is None:
            state = [None] * self.num_layers

        # Path through RNN
        res_out_prev = None
        for i_l in range(self.num_layers):
            y, state[i_l] = getattr(self, self.rnn_type + '_l' + str(i_l))(y, hx=state[i_l])

            # Dropout for hidden-hidden or hidden-output connection
            y = getattr(self, 'dropout_l' + str(i_l))(y)

            # Residual connection
            if self.residual and res_out_prev is not None:
                y += res_out_prev
            res_out_prev = y

        logits_step = self.output(y)

        return logits_step, y, state

    def initialize_hidden(self, batch):
        """Initialize hidden states.

        Args:
            batch (int): the size of mini-batch
        Returns:
            hx_list (list of torch.autograd.Variable(float)):
            cx_list (list of torch.autograd.Variable(float)):

        """
        zero_state = Variable(torch.zeros(batch, self.num_units),
                              volatile=not self.training).float()

        if self.device_id >= 0:
            zero_state = zero_state.cuda(self.device_id)

        if self.rnn_type == 'lstm':
            hx_list = [zero_state] * self.num_layers
            cx_list = [zero_state] * self.num_layers
        elif self.rnn_type == 'gru':
            hx_list = [zero_state] * self.num_layers
            cx_list = None

        return hx_list, cx_list

    def repackage_hidden(self, state):
        """Initialize hidden states.

        Args:
            state (list):
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):
        Returns:
            state (list):
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):

        """
        hx_list, cx_list = state
        if self.rnn_type == 'lstm':
            hx_list = [h.detach() for h in hx_list]
            cx_list = [c.detach() for c in cx_list]
        elif self.rnn_type == 'gru':
            hx_list = [h.detach() for h in hx_list]
        return (hx_list, cx_list)
