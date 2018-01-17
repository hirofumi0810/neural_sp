#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""RNN decoders (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import chainer
from chainer import links as L


class RNNDecoder(chainer.Chain):
    """RNN decoder.
    Args:
        embedding_dim (int): the dimension of input features
        rnn_type (string): lstm or gru or rnn
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        dropout (float): the probability to drop nodes
        use_cuda (bool): if True, use GPUs
        batch_first (bool): if True, batch-major computation will be performed
        residual (bool, optional):
        dense_residual (bool, optional):
    """

    def __init__(self,
                 embedding_dim,
                 rnn_type,
                 num_units,
                 num_layers,
                 dropout,
                 use_cuda,
                 residual=False,
                 dense_residual=False):

        super(RNNDecoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.rnn_type = rnn_type
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.residual = residual
        self.dense_residual = dense_residual

        with self.init_scope():
            self.rnns = []
            for i_layer in range(num_layers):
                if i_layer == 0:
                    decoder_input_size = embedding_dim
                else:
                    decoder_input_size = num_units

                if rnn_type == 'lstm':
                    rnn_i = L.NStepLSTM(n_layers=1,
                                        in_size=decoder_input_size,
                                        out_size=num_units,
                                        dropout=dropout,
                                        initialW=None,
                                        initial_bias=None)
                elif rnn_type == 'gru':
                    rnn_i = L.NStepGRU(n_layers=1,
                                       in_size=decoder_input_size,
                                       out_size=num_units,
                                       dropout=dropout,
                                       initialW=None,
                                       initial_bias=None)
                elif rnn_type == 'rnn':
                    # rnn_i = L.NStepRNNReLU(
                    rnn_i = L.NStepRNNTanh(n_layers=1,
                                           in_size=decoder_input_size,
                                           out_size=num_units,
                                           dropout=dropout,
                                           initialW=None,
                                           initial_bias=None)
                else:
                    raise ValueError(
                        'rnn_type must be "lstm" or "gru" or "rnn".')

                setattr(self, rnn_type + '_l' + str(i_layer), rnn_i)
                if use_cuda:
                    rnn_i.to_gpu()
                self.rnns.append(rnn_i)

    def __call__(self, y, dec_state):
        """Forward computation.
        Args:
            y (): A tensor of size `[B, 1, embedding_dim]`
            dec_state (): A tensor of size
                `[num_layers, B, num_units]`
        Returns:
            dec_out: if batch_first is True, a tensor of size `[B, 1, num_units]`
                else `[1, B, num_units]`
            dec_state (Variable or tuple):
        """
        dec_out = y
        res_outputs_list = []
        # NOTE: exclude residual connection from decoder's inputs
        for i_layer in range(self.num_layers):
            if self.rnn_type == 'lstm':
                hx, cx, dec_out = self.rnns[i_layer](
                    hx=dec_state[0], cx=dec_state[1], xs=dec_out)
                dec_state = (hx, cx)
            else:
                dec_state, dec_out = self.rnns[i_layer](
                    hx=dec_state, xs=dec_out)

            # Residual connection
            if self.residual or self.dense_residual:
                if self.residual or self.dense_residual:
                    for outputs_lower in res_outputs_list:
                        dec_out = dec_out + outputs_lower
                    if self.residual:
                        res_outputs_list = [dec_out]
                    elif self.dense_residual:
                        res_outputs_list.append(dec_out)

        return dec_out, dec_state
