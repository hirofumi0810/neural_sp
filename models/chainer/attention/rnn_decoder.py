#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""RNN decoders (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import chainer
from chainer import functions as F
from chainer import links as L


class RNNDecoder(chainer.Chain):
    """RNN decoder.
    Args:
        input_size (int): the dimension of decoder inputs
        rnn_type (string): lstm or gru
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        dropout (float): the probability to drop nodes
        use_cuda (bool): if True, use GPUs
        batch_first (bool): if True, batch-major computation will be performed
        residual (bool, optional):
        dense_residual (bool, optional):
    """

    def __init__(self,
                 input_size,
                 rnn_type,
                 num_units,
                 num_layers,
                 dropout,
                 use_cuda,
                 residual=False,
                 dense_residual=False):

        super(RNNDecoder, self).__init__()

        self.input_size = input_size
        self.rnn_type = rnn_type
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.residual = residual
        self.dense_residual = dense_residual

        with self.init_scope():
            for l in range(num_layers):
                decoder_input_size = input_size if l == 0 else num_units

                if rnn_type == 'lstm':
                    rnn_i = L.StatelessLSTM(
                        in_size=decoder_input_size,
                        out_size=num_units)
                elif rnn_type == 'gru':
                    rnn_i = L.StatelessGRU(
                        in_size=decoder_input_size,
                        out_size=num_units)
                else:
                    raise ValueError('rnn_type must be "lstm" or "gru".')

                if use_cuda:
                    rnn_i.to_gpu()

                setattr(self, rnn_type + '_l' + str(l), rnn_i)

    def __call__(self, dec_in, dec_state):
        """Forward computation.
        Args:
            dec_in (chainer.Variable, float): A tensor of size
                `[B, 1, embedding_dim + encoder_num_units (decoder_num_units)]`
            dec_state (chainer.Variable(float) or tuple):
        Returns:
            dec_out (chainer.Variable, float): A tensor of size
                `[B, 1, num_units]`
            dec_state (chainer.Variable(float) or tuple):
        """
        if self.rnn_type == 'lstm':
            hx_list, cx_list = dec_state
        elif self.rnn_type == 'gru':
            hx_list = dec_state

        dec_in = F.squeeze(dec_in, axis=1)
        # NOTE: exclude residual connection from decoder's inputs
        for l in range(self.num_layers):
            if self.rnn_type == 'lstm':
                if l == 0:
                    cx_list[l], hx_list[l] = getattr(self, 'lstm_l0')(
                        cx_list[l], hx_list[l], dec_in)
                else:
                    cx_list[l], hx_list[l] = getattr(self, 'lstm_l' + str(l))(
                        cx_list[l], hx_list[l], hx_list[l - 1])
            elif self.rnn_type == 'gru':
                if l == 0:
                    hx_list[l] = getattr(self, 'gru_l0')(hx_list[l], dec_in)
                else:
                    hx_list[l] = getattr(self, 'gru_l' + str(l))(
                        hx_list[l], hx_list[l - 1])

            # Dropout for hidden-hidden or hidden-output connection
            if self.dropout > 0:
                hx_list[l] = F.dropout(hx_list[l], ratio=self.dropout)

            # Residual connection
            if l > 0 and self.residual or self.dense_residual:
                if self.residual:
                    hx_list[l] += sum(hx_list[l - 1])
                elif self.dense_residual:
                    hx_list[l] += sum(hx_list[:l])

        dec_out = F.expand_dims(hx_list[-1], axis=1)

        if self.rnn_type == 'lstm':
            dec_state = hx_list, cx_list
        elif self.rnn_type == 'gru':
            dec_state = hx_list

        return dec_out, dec_state
