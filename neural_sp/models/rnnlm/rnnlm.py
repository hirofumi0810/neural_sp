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
import torch.nn.functional as F

from neural_sp.models.base import ModelBase
from neural_sp.models.criterion import cross_entropy_lsm
from neural_sp.models.linear import Embedding
from neural_sp.models.linear import LinearND
from neural_sp.models.utils import np2var
from neural_sp.models.utils import pad_list
from neural_sp.models.utils import var2np


class RNNLM(ModelBase):
    """RNN language model.

    Args:
        emb_dim (int): the dimension of the embedding
        rnn_type (str): lstm or gru
        bidirectional (bool): if True, create a bidirectional encoder
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers of the encoder
        drop_emb (float): the probability to drop nodes of the embedding layer
        drop_hidden (float): the probability to drop nodes of the outputs of RNN
        drop_output (float): the probability to drop nodes of the linear layer before the softmax layer
        num_classes (int): the number of classes of target labels
        param_init_dist (str): uniform or normal or orthogonal or constant
        param_init (float): Range of uniform distribution to initialize weight parameters
        rec_weight_orthogonal (bool): recurrent weights are orthogonalized
        lsm_prob (float): the probability of label smoothing
        lsm_type (str): uniform or unigram
        tie_weights (bool): input and output embeddings are tied
        residual (bool): add residual connections between RNN layers
        backward (bool): train RNNLM in the backward order

    """

    def __init__(self,
                 emb_dim,
                 rnn_type,
                 bidirectional,
                 num_units,
                 num_layers,
                 drop_emb,
                 drop_hidden,
                 drop_output,
                 num_classes,
                 param_init_dist,
                 param_init,
                 rec_weight_orthogonal,
                 lsm_prob,
                 lsm_type,
                 tie_weights,
                 residual,
                 backward):

        super(ModelBase, self).__init__()
        self.model_type = 'rnnlm'

        self.emb_dim = emb_dim
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_units = num_units
        self.num_layers = num_layers
        self.param_init = param_init
        self.tie_weights = tie_weights
        self.residual = residual
        self.backward = backward
        self.num_classes = num_classes + 1  # Add <EOS> class
        self.sos = num_classes
        self.eos = num_classes
        self.pad_index = -1024.0
        assert rnn_type in ['lstm', 'gru']
        assert not (bidirectional and backward)
        # NOTE: backward LM is only supported for unidirectional LM

        # Setting for regularization
        self.weight_noise_injection = False
        self.ls_prob = lsm_prob
        self.ls_type = lsm_type

        self.embed = Embedding(num_classes=self.num_classes,
                               emb_dim=emb_dim,
                               dropout=drop_emb,
                               ignore_index=self.eos)

        for i_l in range(num_layers):
            if i_l == 0:
                enc_in_size = emb_dim
            else:
                enc_in_size = num_units * self.num_directions

            if rnn_type == 'lstm':
                rnn_i = torch.nn.LSTM(enc_in_size,
                                      hidden_size=num_units,
                                      num_layers=1,
                                      bias=True,
                                      batch_first=True,
                                      dropout=0,
                                      bidirectional=False)
            elif rnn_type == 'gru':
                rnn_i = torch.nn.GRU(enc_in_size,
                                     hidden_size=num_units,
                                     num_layers=1,
                                     bias=True,
                                     batch_first=True,
                                     dropout=0,
                                     bidirectional=False)

            setattr(self, rnn_type + '_l' + str(i_l), rnn_i)

            # Dropout for hidden-hidden or hidden-output connection
            setattr(self, 'dropout_l' + str(i_l),
                    torch.nn.Dropout(p=drop_hidden))

        # for i_l in range(num_layers):
        #     if i_l == 0:
        #         enc_in_size = emb_dim
        #     else:
        #         enc_in_size = num_units * self.num_directions
        #
        #     if rnn_type == 'lstm':
        #         rnn_i = torch.nn.LSTMCell(input_size=enc_in_size,
        #                             hidden_size=num_units,
        #                             bias=True)
        #     elif rnn_type == 'gru':
        #         rnn_i = torch.nn.GRUCell(input_size=enc_in_size,
        #                            hidden_size=num_units,
        #                            bias=True)
        #     else:
        #         raise ValueError('rnn_type must be "lstm" or "gru".')
        #
        #     setattr(self, rnn_type + '_l' + str(i_l), rnn_i)
        #
        #     # Dropout for hidden-hidden or hidden-output connection
        #     setattr(self, 'dropout_l' + str(i_l), torch.nn.Dropout(p=drop_hidden))

        self.output = LinearND(
            num_units * self.num_directions, self.num_classes,
            dropout=drop_output)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if num_units != emb_dim:
                raise ValueError(
                    'When using the tied flag, num_units must be equal to emb_dim')
            self.output.fc.weight = self.embed.embed.weight

        # Initialize weight matrices
        self.init_weights(param_init,
                          distribution=param_init_dist,
                          ignore_keys=['bias'])

        # Initialize all biases with 0
        self.init_weights(0, distribution='constant', keys=['bias'])

        # Recurrent weights are orthogonalized
        if rec_weight_orthogonal:
            self.init_weights(param_init,
                              distribution='orthogonal',
                              keys=[rnn_type, 'weight'],
                              ignore_keys=['bias'])

        # Initialize bias in forget gate with 1
        self.init_forget_gate_bias_with_one()

    def forward(self, ys, is_eval=False):
        """Forward computation.

        Args:
            ys (np.array): A tensor of of size `[B, 2]`
            # state (list): list of (hx_list, cx_list)
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

            # Gaussian noise injection
            if self.weight_noise_injection:
                self.inject_weight_noise(mean=0, std=self.weight_noise_std)

        # Wrap by Variable
        if self.backward:
            raise NotImplementedError
            # TODO(hirofumi): reverse the order out of the model
        else:
            ys = np2var(ys, self.device_id, volatile=is_eval).long()

        ys_in = ys[:, :-1]
        ys_out = ys[:, 1:]
        # TODO(hirofumi): bpttより1小さくなってる

        # Path through embedding
        ys_in = self.embed(ys_in)

        # if state is None:
        #     hx_list, cx_list = self.initialize_hidden(batch=ys.shape[0])
        # else:
        #     hx_list, cx_list = state

        # Path through RNN
        res_outputs_prev = None
        for i_l in range(self.num_layers):
            # Path through RNN
            # getattr(self, self.rnn_type + '_l' +
            #         str(i_l)).flatten_parameters()
            ys_in, _ = getattr(self, self.rnn_type +
                               '_l' + str(i_l))(ys_in, hx=None)
            # if i_l == self.num_layers - 1:
            #     getattr(self, self.rnn_type + '_l' +
            #             str(i_l)).flatten_parameters()

            # Dropout for hidden-hidden or hidden-output connection
            ys_in_tmp = getattr(self, 'dropout_l' + str(i_l))(ys_in)

            # Residual connection
            if self.residual and res_outputs_prev is not None:
                ys_in = ys_in_tmp + res_outputs_prev
            else:
                ys_in = ys_in_tmp
            res_outputs_prev = ys_in_tmp

        logits = self.output(ys_in)

        # Path through RNN
        # for i_l in range(self.num_layers):
        #     if self.rnn_type == 'lstm':
        #         if i_l == 0:
        #             hx_list[0], cx_list[0] = getattr(self, 'lstm_l0')(
        #                 ys_in, (hx_list[0], cx_list[0]))
        #         else:
        #             hx_list[i_l], cx_list[i_l] = getattr(self, 'lstm_l' + str(i_l))(
        #                 hx_list[i_l - 1], (hx_list[i_l], cx_list[i_l]))
        #     elif self.rnn_type == 'gru':
        #         if i_l == 0:
        #             hx_list[0] = getattr(self, 'gru_l0')(ys_in, hx_list[0])
        #         else:
        #             hx_list[i_l] = getattr(self, 'gru_l' + str(i_l))(
        #                 hx_list[i_l - 1], hx_list[i_l])
        #
        #     # Dropout for hidden-hidden or hidden-output connection
        #     hx_list[i_l] = getattr(self, 'dropout_l' + str(i_l))(hx_list[i_l])
        #
        #     # Residual connection
        #     if i_l > 0 and self.residual:
        #         hx_list[i_l] += sum(hx_list[i_l - 1])
        #
        # logits = self.output(hx_list[-1].unsqueeze(1))

        # Compute XE sequence loss
        if self.ls_prob > 0:
            # Label smoothing (with uniform distribution)
            y_lens = [1 for y in ys]
            loss = cross_entropy_lsm(
                logits, ys=ys_out, y_lens=y_lens,
                lsm_prob=self.ls_prob,
                lsm_type=self.ls_type,
                size_average=False) / logits.size(0)
        else:
            loss = F.cross_entropy(
                input=logits.view((-1, logits.size(2))),
                target=ys_out.contiguous().view(-1),
                ignore_index=-1, size_average=True)

        # Compute token-level accuracy in teacher-forcing
        pad_pred = logits.data.view(ys_out.size(
            0), ys_out.size(1), logits.size(-1)).max(2)[1]
        mask = ys_out.data != self.pad_index
        numerator = torch.sum(pad_pred.masked_select(
            mask) == ys_out.data.masked_select(mask))
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
        if state is None:
            state = [None] * self.num_layers

        # Path through RNN
        res_outputs_prev = None
        for i_l in range(self.num_layers):
            # getattr(self, self.rnn_type + '_l' +
            #         str(i_l)).flatten_parameters()
            y, state[i_l] = getattr(self, self.rnn_type + '_l' + str(i_l))(y, hx=state[i_l])
            # if i_l == self.num_layers - 1:
            #     getattr(self, self.rnn_type + '_l' +
            #             str(i_l)).flatten_parameters()

            # Dropout for hidden-hidden or hidden-output connection
            ys_in_tmp = getattr(self, 'dropout_l' + str(i_l))(y)

            # Residual connection
            if self.residual and res_outputs_prev is not None:
                y = ys_in_tmp + res_outputs_prev
            else:
                y = ys_in_tmp
            res_outputs_prev = ys_in_tmp

        logits_step = self.output(y)

        # if state is None:
        #     hx_list, cx_list = self.initialize_hidden(batch=1)
        # else:
        #     hx_list, cx_list = state
        #
        # # Path through RNN
        # for i_l in range(self.num_layers):
        #     if self.rnn_type == 'lstm':
        #         if i_l == 0:
        #             hx_list[0], cx_list[0] = getattr(self, 'lstm_l0')(
        #                 y, (hx_list[0], cx_list[0]))
        #         else:
        #             hx_list[i_l], cx_list[i_l] = getattr(self, 'lstm_l' + str(i_l))(
        #                 hx_list[i_l - 1], (hx_list[i_l], cx_list[i_l]))
        #     elif self.rnn_type == 'gru':
        #         if i_l == 0:
        #             hx_list[0] = getattr(self, 'gru_l0')(y, hx_list[0])
        #         else:
        #             hx_list[i_l] = getattr(self, 'gru_l' + str(i_l))(
        #                 hx_list[i_l - 1], hx_list[i_l])
        #
        #     # Dropout for hidden-hidden or hidden-output connection
        #     hx_list[i_l] = getattr(self, 'dropout_l' + str(i_l))(hx_list[i_l])
        #
        #     # Residual connection
        #     if i_l > 0 and self.residual:
        #         hx_list[i_l] += sum(hx_list[i_l - 1])
        #
        # out = hx_list[-1].unsqueeze(1)
        # logits_step = self.output(out)

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
        else:
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

    def decode(self, start_tokens, max_len):
        """Decoding in the inference stage.

        Args:
            start_tokens (list): A list of length `[B]`
            max_len (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
        Returns:
            best_hyps (list): A list of length `[B]`
            perm_idx (list): A list of length `[B]`

        """
        self.eval()

        if self.bidirectional:
            raise NotImplementedError

        batch = len(start_tokens)

        # Wrap by Variable
        ys = [np2var(np.fromiter([self.sos] + y, dtype=np.int64), self.device_id, volatile=True).long()
              for y in start_tokens]
        y_lens = [0] * batch

        # Convert list to Variable
        y_in = pad_list(ys, -1)

        # Initialize hidden states
        # h = self._init_hidden(batch=batch,
        #                       use_cuda=self.use_cuda,
        #                       volatile=True)
        h = None

        _best_hyps = []
        eos_flag = [False] * batch
        res_outputs_prev = None
        for t in range(max_len):
            # Path through embedding
            if t < 2:
                y = self.embed(y_in[:, t: t + 1])
            else:
                y = self.embed(y)

            for i_l in range(self.num_layers):
                # Path through RNN
                y, _ = getattr(self, self.rnn_type + '_l' + str(i_l))(y, hx=h)

                # Dropout for hidden-hidden or hidden-output connection
                y_tmp = getattr(self, 'dropout_l' + str(i_l))(y)

                # Residual connection
                if res_outputs_prev is not None:
                    y = y_tmp + res_outputs_prev
                else:
                    y = y_tmp
                if self.residual:
                    res_outputs_prev = y_tmp
            res_outputs_prev = None

            logits_step = self.output(y)

            # Pick up 1-best
            y = torch.max(logits_step.squeeze(1), dim=1)[1].unsqueeze(1)
            _best_hyps += [y]

            # Count lengths of hypotheses
            for b in range(batch):
                if not eos_flag[b]:
                    if y.data.cpu().numpy()[b] == self.eos:
                        eos_flag[b] = True
                    y_lens[b] += 1
                    # NOTE: include <EOS>

            # Break if <EOS> is outputed in all mini-batch
            if sum(eos_flag) == batch:
                break

        # Concatenate in L dimension
        _best_hyps = torch.cat(_best_hyps, dim=1)

        # Convert to numpy
        _best_hyps = var2np(_best_hyps)

        # Truncate by <EOS>
        best_hyps = []
        for b in range(batch):
            if self.backward:
                best_hyps += [_best_hyps[b, :y_lens[b]][::-1]]
            else:
                best_hyps += [_best_hyps[b, :y_lens[b]]]

        return best_hyps
