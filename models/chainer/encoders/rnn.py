#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import chainer
# from chainer import functions as F
from chainer import links as L
from chainer import Chain
from chainer import Variable


class RNN_Encoder(Chain):
    """Bidirectional LSTM encoder.
    Args:
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        num_classes (int): the number of classes of target labels
        num_classes (int): the number of classes of target labels
            (except for a blank label). if 0, return hidden states before
            passing through the softmax layer
        rnn_type (string): lstm or gru or rnn_tanh or rnn_relu
        bidirectional (bool): if True, use the bidirectional model
        use_peephole (bool): if True, use peephole connections
        parameter_init (float): the range of uniform distribution to initialize
            weight parameters
        clip_activation (float): the range of activation clipping (> 0)
        num_proj (int): the number of nodes in recurrent projection layer
        bottleneck_dim (int): the dimensions of the bottleneck layer
    """

    def __init__(self,
                 input_size,
                 num_units,
                 num_layers,
                 num_classes,
                 rnn_type,
                 bidirectional,
                 #  use_peephole,
                 parameter_init):
                #  clip_activation,
                #  num_proj,
                #  bottleneck_dim):

        super(RNN_Encoder, self).__init__()
        if rnn_type not in ['lstm', 'gru', 'rnn_tanh', 'rnn_relu']:
            raise ValueError

        self.input_size = input_size
        self.num_units = num_units
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.parameter_init = parameter_init

        # self.use_peephole = use_peephole
        # self.clip_activation = clip_activation

        dropout = 0.9
        # TODO: make this argument

        with self.init_scope():
            if rnn_type == 'lstm':
                if bidirectional:
                    self.rnn = L.NStepBiLSTM(
                        num_layers, input_size, num_units, dropout=dropout)
                else:
                    self.rnn = L.NStepLSTM(
                        num_layers, input_size, num_units, dropout=dropout)

            elif rnn_type == 'gru':
                if bidirectional:
                    self.rnn = L.NStepBiGRU(
                        num_layers, input_size, num_units, dropout=dropout)
                else:
                    self.rnn = L.NStepGRU(
                        num_layers, input_size, num_units, dropout=dropout)

            elif rnn_type == 'rnn_tanh':
                if bidirectional:
                    self.rnn = L.NStepBiRNNTanh(
                        num_layers, input_size, num_units, dropout=dropout)
                else:
                    self.rnn = L.NStepRNNTanh(
                        num_layers, input_size, num_units, dropout=dropout)

            elif rnn_type == 'rnn_relu':
                if bidirectional:
                    self.rnn = L.NStepBiRNNReLU(
                        num_layers, input_size, num_units, dropout=dropout)
                else:
                    self.rnn = L.NStepRNNReLU(
                        num_layers, input_size, num_units, dropout=dropout)

            else:
                raise ValueError

            if self.num_classes != 0:
                self.fc = L.Linear(
                    num_units * self.num_directions, num_classes)

        # Initialize parameters
        for param in self.params():
            param.data[...] = np.random.uniform(
                -parameter_init, parameter_init, param.data.shape)
        # TODO: これはモデルのbase.pyで行う？？
        # TODO: xpにしなくていい？

    def _init_hidden(self, batch_size, xp):
        """Initialize hidden states.
        Args:
            batch_size (int): the size of mini-batch
            xp:
        Returns:
            if rnn_type is 'lstm', return a tuple of
                (initial_hidden_state, initial_memory_cell),
            otherwise return a tensor initial_hidden_state.
        """
        initial_hidden_state = Variable(xp.zeros(
            (self.num_layers * self.num_directions, batch_size, self.num_units),
            dtype=xp.float32))
        # volatile??

        if self.rnn_type == 'lstm':
            # hidden states & memory cells
            initial_memory_cell = Variable(xp.zeros(
                (self.num_layers * self.num_directions, batch_size, self.num_units),
                dtype=xp.float32))
            # volatile??
            return (initial_hidden_state, initial_memory_cell)
        else:  # gru or rnn
            # hidden states
            return initial_hidden_state

    def __call__(self, inputs):
        """
        Args:
            inputs (list): list of tensors of size `[B, input_size]`.
                Note that len(inputs) == max_time.
        Returns:
            logits (list): list of tensors of size `[B, num_classes]`.
                Note that len(logits) == max_time.
            final_state: A tensor of size
                `[num_layers * num_directions, B, num_units]`
        """
        # Initialize hidden states (and memory cells) per mini-batch
        initial_state = self._init_hidden(
            batch_size=len(inputs), xp=chainer.cuda.get_array_module(inputs))
        # TODO:Noneを渡すとゼロベクトルを用意
        # Encoder-DecoderのDecoderの時は初期ベクトルhxを渡す
        # h_0, c_0 = None, None

        if self.rnn_type == 'lstm':
            final_state, c_n, outputs = self.rnn(
                hx=initial_state[0], cx=initial_state[1], xs=inputs)

        else:
            # gru or rnn
            final_state, outputs = self.rnn(hx=initial_state, xs=inputs)

        # For attention-based models
        if self.num_classes == 0:
            return outputs, final_state

        logits = [self.fc(outputs[t]) for t in range(len(outputs))]

        return logits, final_state
