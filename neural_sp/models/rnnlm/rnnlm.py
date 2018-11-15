#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN language model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.base import ModelBase
from neural_sp.models.linear import Embedding
from neural_sp.models.linear import LinearND
from neural_sp.models.torch_utils import np2var
from neural_sp.models.torch_utils import pad_list
from neural_sp.models.torch_utils import var2np


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
        self.sos = 2
        self.eos = 2
        self.pad = 3
        # NOTE: reserved in advance

        self.embed = Embedding(num_classes=self.num_classes,
                               emb_dim=args.emb_dim,
                               dropout=args.dropout_emb,
                               ignore_index=self.pad)

        for i_l in six.moves.range(args.num_layers):
            if i_l == 0:
                enc_in_size = args.emb_dim
            else:
                enc_in_size = args.num_units

            if args.rnn_type == 'lstm':
                rnn_i = nn.LSTMCell
            elif args.rnn_type == 'gru':
                rnn_i = nn.GRUCell

            setattr(self, args.rnn_type + '_l' + str(i_l), rnn_i(input_size=enc_in_size,
                                                                 hidden_size=args.num_units,
                                                                 bias=True))

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

    def forward(self, ys, state, is_eval=False):
        """Forward computation.

        Args:
            ys (np.array): A tensor of of size `[B, 2]`
            state (list): list of (hx_list, cx_list)
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):
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

            ys_in = ys[:, :-1]  # B*1
            ys_out = ys[:, 1:]  # B*1

        # Path through embedding
        ys_in = self.embed(ys_in)

        if state is None:
            hx_list, cx_list = self.initialize_hidden(batch_size=ys.shape[0])
        else:
            hx_list, cx_list = state

        # Path through RNN
        res_out_prev = None
        for i_l in six.moves.range(self.num_layers):
            name = self.rnn_type + '_l' + str(i_l)
            if self.rnn_type == 'lstm':
                if i_l == 0:
                    hx_list[0], cx_list[0] = getattr(self, name)(
                        ys_in, (hx_list[0], cx_list[0]))
                else:
                    hx_list[i_l], cx_list[i_l] = getattr(self, name)(
                        hx_list[i_l - 1], (hx_list[i_l], cx_list[i_l]))
            elif self.rnn_type == 'gru':
                if i_l == 0:
                    hx_list[0] = getattr(self, name)(ys_in, hx_list[0])
                else:
                    hx_list[i_l] = getattr(self, name)(hx_list[i_l - 1], hx_list[i_l])

            # Dropout for hidden-hidden or hidden-output connection
            hx_list[i_l] = getattr(self, 'dropout_l' + str(i_l))(hx_list[i_l])

            # Residual connection
            if self.residual and res_out_prev is not None:
                hx_list[i_l] += res_out_prev
            res_out_prev = hx_list[i_l]

        logits = self.output(hx_list[-1].unsqueeze(1))

        # Compute XE sequence loss
        loss = F.cross_entropy(input=logits.view((-1, logits.size(2))),
                               target=ys_out.contiguous().view(-1),
                               ignore_index=self.pad, size_average=True)

        # Compute token-level accuracy in teacher-forcing
        pad_pred = logits.view(ys_out.size(0), ys_out.size(1), logits.size(-1)).argmax(2)
        mask = ys_out != self.pad
        numerator = torch.sum(pad_pred.masked_select(mask) == ys_out.masked_select(mask))
        denominator = torch.sum(mask)
        acc = float(numerator) / float(denominator)

        return loss, acc, (hx_list, cx_list)

    def predict(self, y, state):
        """

        Args:
            y (): A tensor of size `[B, emb_dim]`
            state (list):
        Returns:
            logits_step ():
            out ():
            state (): A tuple of (hx_list, cx_list)
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):

        """
        if state[0] is None:
            hx_list, cx_list = self.initialize_hidden(batch_size=1)
        else:
            hx_list, cx_list = state

        # Path through RNN
        res_out_prev = None
        for i_l in six.moves.range(self.num_layers):
            name = self.rnn_type + '_l' + str(i_l)
            if self.rnn_type == 'lstm':
                if i_l == 0:
                    hx_list[0], cx_list[0] = getattr(self, name)(
                        y, (hx_list[0], cx_list[0]))
                else:
                    hx_list[i_l], cx_list[i_l] = getattr(self, name)(
                        hx_list[i_l - 1], (hx_list[i_l], cx_list[i_l]))
            elif self.rnn_type == 'gru':
                if i_l == 0:
                    hx_list[0] = getattr(self, name)(y, hx_list[0])
                else:
                    hx_list[i_l] = getattr(self, name)(hx_list[i_l - 1], hx_list[i_l])

            # Dropout for hidden-hidden or hidden-output connection
            hx_list[i_l] = getattr(self, 'dropout_l' + str(i_l))(hx_list[i_l])

            # Residual connection
            if self.residual and res_out_prev is not None:
                hx_list[i_l] += res_out_prev
            res_out_prev = hx_list[i_l]

        logits_step = self.output(hx_list[-1].unsqueeze(1))

        return logits_step, y, (hx_list, cx_list)

    def initialize_hidden(self, batch_size):
        """Initialize hidden states.

        Args:
            batch_size (int): the size of mini-batch
        Returns:
            hx_list (list of torch.autograd.Variable(float)):
            cx_list (list of torch.autograd.Variable(float)):

        """
        zero_state = Variable(torch.zeros(batch_size, self.num_units),
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
        else:
            hx_list = [h.detach() for h in hx_list]
        return (hx_list, cx_list)

    def copy_from_seqrnnlm(self, rnnlm):
        """Copy parameters from sequene-level RNNLM.

        Args:
            rnnlm ():

        """
        for i_l in six.moves.range(self.num_layers):
            name = self.rnn_type + '_l' + str(i_l)
            getattr(self, name).weight_ih.data = getattr(rnnlm, name).weight_ih_l0.data
            getattr(self, name).weight_hh.data = getattr(rnnlm, name).weight_hh_l0.data
            getattr(self, name).bias_ih.data = getattr(rnnlm, name).bias_ih_l0.data
            getattr(self, name).bias_hh.data = getattr(rnnlm, name).bias_hh_l0.data
        self.output.fc.weight.data = rnnlm.output.fc.weight.data
        self.output.fc.bias.data = rnnlm.output.fc.bias.data
