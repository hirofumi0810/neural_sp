#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""RNN language model (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.pytorch_v3.base import ModelBase
from src.models.pytorch_v3.linear import LinearND, Embedding
from src.models.pytorch_v3.utils import np2var, var2np, pad_list
from src.models.pytorch_v3.criterion import cross_entropy_label_smoothing


class RNNLM(ModelBase):
    """RNN language model.
    Args:
        embedding_dim (int):
        rnn_type (string): lstm or gru
        bidirectional (bool): if True create a bidirectional encoder
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers of the encoder
        dropout_embedding (float): the probability to drop nodes of the embedding layer
        dropout_hidden (float): the probability to drop nodes in hidden-hidden connection
        dropout_output (float):
        num_classes (int): the number of classes of target labels
        parameter_init_distribution (string): uniform or normal or orthogonal
            or constant distribution
        parameter_init (float): Range of uniform distribution to initialize
            weight parameters
        recurrent_weight_orthogonal (bool): if True, recurrent weights are
            orthogonalized
        init_forget_gate_bias_with_one (bool): if True, initialize the forget
            gate bias with 1
        label_smoothing_prob (float):
        label_smoothing_type (string): uniform or unigram
        tie_weights (bool):
        backward (bool): if True, backward RNNLM
    """

    def __init__(self,
                 embedding_dim,
                 rnn_type,
                 bidirectional,
                 num_units,
                 num_layers,
                 dropout_embedding,
                 dropout_hidden,
                 dropout_output,
                 num_classes,
                 parameter_init_distribution='uniform',
                 parameter_init=0.1,
                 recurrent_weight_orthogonal=False,
                 init_forget_gate_bias_with_one=True,
                 label_smoothing_prob=0,
                 label_smoothing_type='unigram',
                 tie_weights=False,
                 backward=False):

        super(ModelBase, self).__init__()
        self.model_type = 'rnnlm'

        self.embedding_dim = embedding_dim
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_units = num_units
        self.num_layers = num_layers
        self.parameter_init = parameter_init
        self.tie_weights = tie_weights
        self.backward = backward
        self.num_classes = num_classes + 1  # Add <EOS> class
        self.sos = num_classes
        self.eos = num_classes
        self.pad_index = -1024

        # Setting for regularization
        self.weight_noise_injection = False
        self.ls_prob = label_smoothing_prob
        self.ls_type = label_smoothing_type

        self.embed = Embedding(num_classes=self.num_classes,
                               embedding_dim=embedding_dim,
                               dropout=dropout_embedding,
                               ignore_index=self.eos)

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim,
                               hidden_size=num_units,
                               num_layers=num_layers,
                               bias=True,
                               batch_first=True,
                               dropout=dropout_hidden,
                               bidirectional=bidirectional)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_dim,
                              hidden_size=num_units,
                              num_layers=num_layers,
                              bias=True,
                              batch_first=True,
                              dropout=dropout_hidden,
                              bidirectional=bidirectional)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(embedding_dim,
                              hidden_size=num_units,
                              num_layers=num_layers,
                              nonlinearity='tanh',
                              # nonlinearity='relu',
                              bias=True,
                              batch_first=True,
                              dropout=dropout_hidden,
                              bidirectional=bidirectional)
        else:
            raise ValueError('rnn_type must be "lstm" or "gru" or "rnn".')

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
            raise NotImplementedError

            if num_units != embedding_dim:
                raise ValueError(
                    'When using the tied flag, num_units must be equal to embedding_dim')
            self.output.fc.weight = self.embed.embed.weight

        # Initialize weight matrices
        self.init_weights(parameter_init,
                          distribution=parameter_init_distribution,
                          ignore_keys=['bias'])

        # Initialize all biases with 0
        self.init_weights(0, distribution='constant', keys=['bias'])

        # Recurrent weights are orthogonalized
        if recurrent_weight_orthogonal:
            self.init_weights(parameter_init,
                              distribution='orthogonal',
                              keys=[rnn_type, 'weight'],
                              ignore_keys=['bias'])

        # Initialize bias in forget gate with 1
        if init_forget_gate_bias_with_one:
            self.init_forget_gate_bias_with_one()

    def forward(self, ys, is_eval=False):
        """Forward computation.
        Args:
            ys (list): A list of length `[B]`, which contains arrays of size `[L]`
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
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

        # Sort by lenghts in the descending order
        perm_idx = sorted(list(range(0, len(ys), 1)),
                          key=lambda i: len(ys[i]), reverse=True)
        ys = [ys[i] for i in perm_idx]
        # NOTE: must be descending order for pack_padded_sequence

        # Wrap by Variable
        y_lens = [len(y) + 1 for y in ys]
        if self.backward:
            ys = [np2var(np.fromiter(y[::-1], dtype=np.int64), self.device_id).long()
                  for y in ys]
            # NOTE: reverse the order
        else:
            ys = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long()
                  for y in ys]

        sos = Variable(ys[0].data.new(1,).fill_(self.sos).long())
        eos = Variable(ys[0].data.new(1,).fill_(self.eos).long())

        # Append <SOS> and <EOS>
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # Convert list to Variable
        ys_in = pad_list(ys_in, self.eos)
        ys_out = pad_list(ys_out, -1)

        # Path through embedding
        ys_in = self.embed(ys_in)

        # Pack RNN inputs
        ys_in = pack_padded_sequence(ys_in, y_lens, batch_first=True)

        # Path through RNN
        ys_in, _ = self.rnn(ys_in, hx=None)

        # Unpack RNN outputs
        ys_in, unpacked_seq_len = pad_packed_sequence(
            ys_in, batch_first=True, padding_value=0)
        # assert y_lens - 1 == unpacked_seq_len

        logits = self.output(ys_in)

        # Compute XE sequence loss
        if self.ls_prob > 0:
            # Label smoothing (with uniform distribution)
            y_lens = [y.size(0) + 1 for y in ys]   # Add <EOS>
            loss = cross_entropy_label_smoothing(
                logits, ys=ys_out, y_lens=y_lens,
                label_smoothing_prob=self.ls_prob,
                label_smoothing_type=self.ls_type,
                size_average=False) / sum(logits.size()[:2])
        else:
            loss = F.cross_entropy(input=logits.view((-1, logits.size(2))),
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

        return loss, acc

    def _init_hidden(self, batch_size, use_cuda, volatile):
        """Initialize hidden states.
        Args:
            batch_size (int): the size of mini-batch
            use_cuda (bool, optional):
            volatile (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            if rnn_type is 'lstm', return a tuple of tensors (h_0, c_0).
                h_0 (torch.autograd.Variable, float): A tensor of size
                    `[num_layers * num_directions, batch_size, num_units]`
                c_0 (torch.autograd.Variable, float): A tensor of size
                    `[num_layers * num_directions, batch_size, num_units]`
            otherwise return h_0.
        """
        h_0 = Variable(torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.num_units))
        if volatile:
            h_0.volatile = True
        if use_cuda:
            h_0 = h_0.cuda()

        if self.rnn_type == 'lstm':
            c_0 = Variable(torch.zeros(
                self.num_layers * self.num_directions, batch_size, self.num_units))
            if volatile:
                c_0.volatile = True
            if use_cuda:
                c_0 = c_0.cuda()

            return (h_0, c_0)
        else:
            return h_0

    def decode(self, start_tokens, max_decode_len):
        """Decoding in the inference stage.
        Args:
            start_tokens (list): A list of length `[B]`
            max_decode_len (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
        Returns:
            best_hyps (list): A list of length `[B]`
            perm_idx (list): A list of length `[B]`
        """
        self.eval()

        batch_size = len(start_tokens)

        # Wrap by Variable
        ys = [np2var(np.fromiter([self.sos] + y, dtype=np.int64), self.device_id, volatile=True).long()
              for y in start_tokens]
        y_lens = [0] * batch_size

        # Convert list to Variable
        y_in = pad_list(ys, -1)

        # Initialize hidden states
        # h = self._init_hidden(batch_size=batch_size,
        #                       use_cuda=self.use_cuda,
        #                       volatile=True)
        h = None

        _best_hyps = []
        eos_flag = [False] * batch_size
        for t in range(max_decode_len):
            # Path through embedding
            if t < 2:
                y = self.embed(y_in[:, t:t + 1])
            else:
                y = self.embed(y)

            # Path through RNN
            y, h = self.rnn(y, hx=h)

            logits_step = self.output(y)

            # Pick up 1-best
            y = torch.max(logits_step.squeeze(1), dim=1)[1].unsqueeze(1)
            _best_hyps += [y]

            # Count lengths of hypotheses
            for b in range(batch_size):
                if not eos_flag[b]:
                    if y.data.cpu().numpy()[b] == self.eos:
                        eos_flag[b] = True
                    y_lens[b] += 1
                    # NOTE: include <EOS>

            # Break if <EOS> is outputed in all mini-batch
            if sum(eos_flag) == batch_size:
                break

        # Concatenate in L dimension
        _best_hyps = torch.cat(_best_hyps, dim=1)

        # Convert to numpy
        _best_hyps = var2np(_best_hyps)

        # Truncate by <EOS>
        best_hyps = []
        for b in range(batch_size):
            if self.backward:
                best_hyps += [_best_hyps[b, :y_lens[b]][::-1]]
            else:
                best_hyps += [_best_hyps[b, :y_lens[b]]]

        return best_hyps
