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

from models.pytorch_v3.base import ModelBase
from models.pytorch_v3.linear import LinearND, Embedding
from models.pytorch_v3.encoders.rnn import _init_hidden


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
        tie_weights (bool):
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
                 tie_weights=False):

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
        self.num_classes = num_classes + 1  # Add <EOS> class
        self.eos = num_classes

        self.embed = Embedding(num_classes=self.num_classes,
                               embedding_dim=embedding_dim,
                               dropout=dropout_embedding)
        # TODO: add label smoothing

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
        # if tie_weights:
        #     if num_units != embedding_dim:
        #         raise ValueError(
        #             'When using the tied flag, num_units must be equal to embedding_dim')
        #     self.output.fc.weight = self.embed.embed.weight

        ##################################################
        # Initialize parameters
        ##################################################
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
        if is_eval:
            self.eval()
        else:
            self.train()

            # Gaussian noise injection
            # if self.weight_noise_injection:
            #     self.inject_weight_noise(mean=0, std=self.weight_noise_std)

        # NOTE: ys is padded with -1 here
        # ys_in is padded with <EOS> in order to convert to one-hot vector,
        # and added <SOS> before the first token
        # ys_out_fwd is padded with -1, and added <EOS> after the last token
        ys_in = self._create_var((ys.shape[0], ys.shape[1]),
                                 fill_value=self.eos, dtype='long')
        ys_out = self._create_var((ys.shape[0], ys.shape[1]),
                                  fill_value=-1, dtype='long')
        for b in range(len(ys)):
            ys_in.data[b, :y_lens[b]] = torch.from_numpy(
                ys[b, :y_lens[b]])
            ys_out.data[b, :y_lens[b] - 1] = torch.from_numpy(
                ys[b, 1:y_lens[b]])
            ys_out.data[b, y_lens[b] - 1] = self.eos

        if self.use_cuda:
            ys_in = ys_in.cuda()
            ys_out = ys_out.cuda()

        # Wrap by Variable
        y_lens = self.np2var(y_lens, dtype='int')

        # Sort by lengths in descending order
        y_lens, perm_idx = y_lens.sort(dim=0, descending=True)
        ys_in = ys_in[perm_idx]
        ys_out = ys_out[perm_idx]

        # Path through embedding
        ys_in = self.embed(ys_in)

        # Initialize hidden states (and memory cells) per mini-batch
        h_0 = _init_hidden(batch_size=len(ys),
                           rnn_type=self.rnn_type,
                           num_units=self.num_units,
                           num_directions=self.num_directions,
                           num_layers=self.num_layers,
                           use_cuda=self.use_cuda,
                           volatile=is_eval)

        # Pack RNN inputs
        ys_in = pack_padded_sequence(
            ys_in, self.var2np(y_lens).tolist(), batch_first=True)

        # Path through RNN
        ys_in, _ = self.rnn(ys_in, hx=h_0)

        # Unpack RNN outputs
        ys_in, unpacked_seq_len = pad_packed_sequence(
            ys_in, batch_first=True, padding_value=0)
        # assert y_lens - 1 == unpacked_seq_len

        logits = self.output(ys_in)

        # TODO: add BPTT

        # Compute XE sequence loss
        loss = F.cross_entropy(
            input=logits.view((-1, logits.size(2))),
            target=ys_out.contiguous().view(-1),
            ignore_index=-1, size_average=False) / len(ys)

        if is_eval:
            loss = loss.data[0]

        return loss

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
