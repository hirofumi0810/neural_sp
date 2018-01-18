# ! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention layer (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import chainer
from chainer import functions as F
from chainer import links as L

from models.chainer.linear import LinearND

ATTENTION_TYPE = ['content', 'location', 'dot_product', 'rnn_attention']


class AttentionMechanism(chainer.Chain):
    """Attention layer.
    Args:
        decoder_num_units (int): the number of units in each layer of the
            decoder
        attention_type (string): the type of attention
        attention_dim: (int) the dimension of the attention layer
        use_cuda (bool):
        sharpening_factor (float, optional): a sharpening factor in the softmax
            layer for computing attention weights
        sigmoid_smoothing (bool, optional): if True, replace softmax function
            in computing attention weights with sigmoid function for smoothing
        out_channels (int, optional): the number of channles of conv outputs.
            This is used for location-based attention.
        kernel_size (int, optional): the size of kernel.
            This must be the odd number.
    """

    def __init__(self,
                 decoder_num_units,
                 attention_type,
                 attention_dim,
                 use_cuda,
                 sharpening_factor=1,
                 sigmoid_smoothing=False,
                 out_channels=10,
                 kernel_size=201):

        super(AttentionMechanism, self).__init__()

        self.decoder_num_units = decoder_num_units
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.sharpening_factor = sharpening_factor
        self.sigmoid_smoothing = sigmoid_smoothing

        with self.init_scope():
            if self.attention_type == 'content':
                self.W = LinearND(decoder_num_units * 2, attention_dim,
                                  bias=True, use_cuda=use_cuda)  # NOTE
                self.V = LinearND(attention_dim, 1,
                                  bias=False, use_cuda=use_cuda)

            elif self.attention_type == 'location':
                assert kernel_size % 2 == 1
                self.conv = L.Convolution2D(
                    in_channels=1,
                    out_channels=out_channels,
                    ksize=(1, kernel_size),
                    stride=1,
                    pad=(0, kernel_size // 2),
                    nobias=False,
                    initialW=None,
                    initial_bias=None)
                self.W = LinearND(decoder_num_units * 2, attention_dim,
                                  bias=True, use_cuda=use_cuda)  # NOTE
                self.W_conv = LinearND(out_channels, attention_dim,
                                       bias=False, use_cuda=use_cuda)
                self.V = LinearND(attention_dim, 1,
                                  bias=False, use_cuda=use_cuda)

            elif self.attention_type == 'dot_product':
                self.W_keys = LinearND(decoder_num_units, attention_dim,
                                       bias=False, use_cuda=use_cuda)
                self.W_query = LinearND(decoder_num_units, attention_dim,
                                        bias=False, use_cuda=use_cuda)

            elif self.attention_type == 'rnn_attention':
                raise NotImplementedError

            else:
                raise TypeError(
                    "attention_type should be one of [%s], you provided %s." %
                    (", ".join(ATTENTION_TYPE), attention_type))

            if use_cuda:
                for c in self.children():
                    c.to_gpu()

    def __call__(self, enc_out, dec_out, att_weights_step):
        """Forward computation.
        Args:
            enc_out (chainer.Variable): A tensor of size
                `[B, T_in, encoder_num_units]`
            dec_out (chainer.Variable): A tensor of size
                `[B, 1, decoder_num_units]`
            att_weights_step (chainer.Variable): A tensor of size `[B, T_in]`
        Returns:
            context_vec (chainer.Variable): A tensor of size
                `[B, 1, encoder_num_units]`
            att_weights_step (chainer.Variable): A tensor of size `[B, T_in]`
        """
        batch_size, max_time = enc_out.shape[:2]

        if self.attention_type == 'content':
            ###################################################################
            # energy = <v, tanh(W([h_de; h_en] + b))>
            ###################################################################
            concat = F.concat(
                [enc_out, F.broadcast_to(dec_out, enc_out.shape)], axis=2)
            energy = F.squeeze(self.V(F.tanh(self.W(concat))), axis=2)

        elif self.attention_type == 'location':
            ###################################################################
            # f = F * α_{i-1}
            # energy = <v, tanh(W([h_de; h_en] + W_conv(f) + b))>
            ###################################################################
            conv_feat = F.squeeze(self.conv(
                att_weights_step.reshape(batch_size, 1, 1, max_time)), axis=2)
            # -> `[B, out_channels, T_in]`
            conv_feat = conv_feat.transpose(0, 2, 1)
            # -> `[B, T_in, out_channels]`
            concat = F.concat(
                [enc_out, F.broadcast_to(dec_out, enc_out.shape)], axis=2)
            energy = F.squeeze(
                self.V(F.tanh(self.W(concat) + self.W_conv(conv_feat))), axis=2)

        elif self.attention_type == 'dot_product':
            ###################################################################
            # energy = <W_keys(h_en), W_query(h_de)>
            ###################################################################
            keys = self.W_keys(enc_out)
            query = self.W_query(dec_out).transpose(0, 2, 1)
            energy = F.squeeze(F.matmul(keys, query), axis=2)

        elif self.attention_type == 'rnn_attention':
            raise NotImplementedError

        else:
            raise NotImplementedError

        # Sharpening
        energy = energy * self.sharpening_factor
        # NOTE: energy: `[B, T_in]`

        # Compute attention weights
        if self.sigmoid_smoothing:
            att_weights_step = F.sigmoid(energy)
        else:
            att_weights_step = F.softmax(energy, axis=1)

        # Compute context vector (weighted sum of encoder outputs)
        batch_size, max_time = att_weights_step.shape
        context_vec = F.sum(enc_out * F.broadcast_to(
            F.expand_dims(att_weights_step, axis=2),
            (batch_size, max_time, enc_out.shape[-1])),
            axis=1, keepdims=True)

        return context_vec, att_weights_step
