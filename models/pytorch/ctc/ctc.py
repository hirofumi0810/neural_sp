#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from warpctc_pytorch import CTCLoss
except:
    raise ImportError('Install warpctc_pytorch.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pytorch.base import ModelBase
from models.pytorch.encoders.load_encoder import load


class CTC(ModelBase):
    """The Connectionist Temporal Classification model.
    Args:
        input_size (int): the dimension of input features
        encoder_type (string): the type of the encoder. Set lstm or gru or rnn.
        bidirectional (bool): if True create a bidirectional encoder
        num_units (int): the number of units in each layer
        # num_proj (int): the number of nodes in recurrent projection layer
        num_layers (int): the number of layers of the encoder
        dropout (float): the probability to drop nodes
        num_classes (int): the number of classes of target labels
            (except for a blank label)
        splice (int, optional): frames to splice. Default is 1 frame.
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters
        # bottleneck_dim (int, optional): the dimensions of the bottleneck
        # layer
    """

    def __init__(self,
                 input_size,
                 encoder_type,
                 bidirectional,
                 num_units,
                 #  num_proj,
                 num_layers,
                 dropout,
                 num_classes,
                 splice=1,
                 parameter_init=0.1,
                 bottleneck_dim=None):

        super(ModelBase, self).__init__()

        self.input_size = input_size
        self.splice = splice
        self.encoder_type = encoder_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_units = num_units
        # self.num_proj = num_proj
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes + 1  # add blank class

        self.parameter_init = parameter_init
        self.bottleneck_dim = bottleneck_dim

        # Load encoder
        encoder = load(encoder_type=encoder_type)
        if encoder_type in ['lstm', 'gru', 'rnn']:
            self.encoder = encoder(input_size=input_size,
                                   rnn_type=encoder_type,
                                   bidirectional=bidirectional,
                                   num_units=num_units,
                                   num_layers=num_layers,
                                   dropout=dropout,
                                   parameter_init=parameter_init,
                                   use_cuda=self.use_cuda,
                                   batch_first=True)
        else:
            raise NotImplementedError

        if self.bottleneck_dim is not None:
            self.bottleneck = torch.nn.Linear(
                num_units * self.num_directions, bottleneck_dim)
            self.fc = torch.nn.Linear(bottleneck_dim, num_classes)
        else:
            self.fc = torch.nn.Linear(
                num_units * self.num_directions, num_classes)

        # GPU setting
        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            # TODO: Remove this??

    def forward(self, inputs):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T, input_size]`
        Returns:
            logits (FloatTensor): A tensor of size `[T, B, num_classes + 1]`
        """
        encoder_states, final_state = self.encoder(inputs)
        batch_size, max_time = encoder_states.size()[:2]
        encoder_states = encoder_states.contiguous()
        encoder_states = encoder_states.view(batch_size * max_time, -1)

        if self.bottleneck_dim is not None:
            logits = self.bottleneck(encoder_states)
            logits = self.fc(logits)
        else:
            logits = self.fc(encoder_states)

        logits = logits.view(batch_size, max_time, -1)
        return logits

    def loss(self, logits, labels, inputs_seq_len, labels_seq_len):
        """
        Args:
            logits (FloatTensor): A tensor of size `[B, T, num_classes]`
            labels (LongTensor): A tensor of size `[B, U]`
            inputs_seq_len (LongTensor): A tensor of size `[B]`
            labels_seq_len (LongTensor): A tensor of size `[B]`
        Returns:
            ctc_loss ():
        """
        batch_size, max_time, num_classes = logits.size()
        ctc_loss_fn = CTCLoss()

        print(logits)
        print(labels)
        print(inputs_seq_len)
        print(labels_seq_len)
        ctc_loss = ctc_loss_fn(logits, labels, inputs_seq_len, labels_seq_len)
        ctc_loss /= batch_size
        return ctc_loss

    def posteriors(self, logits, softmax_temperature=1, blank_prior=None,
                   log_scale=False):
        """
        Args:
            logits (FloatTensor): A tensor of size `[B, T, num_classes]`
            softmax_temperature (float, optional):
            blank_prior (float, optional):
            log_scale (bool, optional):
        Returns:
            probs ():
        """
        batch_size, max_time, num_classes = logits.size()
        logits = logits.view(batch_size, -1)
        probs = F.softmax(logits / softmax_temperature)

        # Divide by blank prior
        if blank_prior is not None:
            raise NotImplementedError

        probs = probs.view(batch_size, max_time, num_classes)
        return probs

    def decode(self, beam_width=1):
        """
        """
        if beam_width == 1:
            return self._decode_greedy()
        else:
            return self._decode_beam_search()

    def _decode_greedy(self, logprobs):
        """
        Args:
            logprobs (FloatTensor):
        Returns:

        """
        raise NotImplementedError

    def _decode_beam_search(self, logprobs, beam_width):
        """
        Args:
            logprobs (FloatTensor):
        Returns:

        """
        raise NotImplementedError
