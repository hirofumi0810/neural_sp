# ! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention layer (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import Variable
from chainer import cuda

from models.chainer.linear import LinearND

ATTENTION_TYPE = ['content', 'location', 'dot_product', 'rnn_attention']


class AttentionMechanism(chainer.Chain):
    """Attention layer.
    Args:
        encoder_num_units (int): the number of units in each layer of the
            encoder
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
        num_heads (int, optional): the number of heads in the multi-head attention
    """

    def __init__(self,
                 encoder_num_units,
                 decoder_num_units,
                 attention_type,
                 attention_dim,
                 use_cuda,
                 sharpening_factor=1,
                 sigmoid_smoothing=False,
                 out_channels=10,
                 kernel_size=201,
                 num_heads=1):

        super(AttentionMechanism, self).__init__()

        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.use_cuda = use_cuda
        self.sharpening_factor = sharpening_factor
        self.sigmoid_smoothing = sigmoid_smoothing
        self.num_heads = num_heads

        # Multi-head attention
        if num_heads > 1:
            setattr(self, 'W_mha', LinearND(
                encoder_num_units * num_heads, encoder_num_units,
                use_cuda=use_cuda))

        with self.init_scope():
            for h in range(num_heads):
                if self.attention_type == 'content':
                    setattr(self, 'W_enc_head' + str(h),
                            LinearND(encoder_num_units, attention_dim,
                                     bias=True, use_cuda=use_cuda))
                    setattr(self, 'W_dec_head' + str(h),
                            LinearND(decoder_num_units, attention_dim,
                                     bias=False, use_cuda=use_cuda))
                    setattr(self, 'V_head' + str(h),
                            LinearND(attention_dim, 1,
                                     bias=False, use_cuda=use_cuda))

                elif self.attention_type == 'location':
                    assert kernel_size % 2 == 1

                    setattr(self, 'W_enc_head' + str(h),
                            LinearND(encoder_num_units, attention_dim,
                                     bias=True, use_cuda=use_cuda))
                    setattr(self, 'W_dec_head' + str(h),
                            LinearND(decoder_num_units, attention_dim,
                                     bias=False, use_cuda=use_cuda))
                    setattr(self, 'W_conv_head' + str(h),
                            LinearND(out_channels, attention_dim,
                                     bias=False, use_cuda=use_cuda))

                    setattr(self, 'conv_head' + str(h),
                            L.ConvolutionND(ndim=1,
                                            in_channels=1,
                                            out_channels=out_channels,
                                            ksize=kernel_size,
                                            stride=1,
                                            pad=kernel_size // 2,
                                            nobias=True,
                                            initialW=None,
                                            initial_bias=None))
                    # setattr(self, 'conv_head' + str(h),
                    #         L.Convolution2D(in_channels=1,
                    #                         out_channels=out_channels,
                    #                         ksize=(1, kernel_size),
                    #                         stride=1,
                    #                         pad=(0, kernel_size // 2),
                    #                         nobias=True,
                    #                         initialW=None,
                    #                         initial_bias=None))
                    setattr(self, 'V_head' + str(h),
                            LinearND(attention_dim, 1,
                                     bias=False, use_cuda=use_cuda))

                elif self.attention_type == 'dot_product':
                    setattr(self, 'W_enc_head' + str(h),
                            LinearND(encoder_num_units, attention_dim,
                                     bias=False, use_cuda=use_cuda))
                    setattr(self, 'W_dec_head' + str(h),
                            LinearND(decoder_num_units, attention_dim,
                                     bias=False, use_cuda=use_cuda))

                elif self.attention_type == 'rnn_attention':
                    raise NotImplementedError

                else:
                    raise TypeError(
                        "attention_type should be one of [%s], you provided %s." %
                        (", ".join(ATTENTION_TYPE), attention_type))

            if use_cuda:
                for c in self.children():
                    c.to_gpu()

    def __call__(self, enc_out, x_lens, dec_out, aw_step):
        """Forward computation.
        Args:
            enc_out (chainer.Variable): A tensor of size
                `[B, T_in, encoder_num_units]`
            x_lens (chainer.Variable): A tensor of size `[B]`
            dec_out (chainer.Variable): A tensor of size
                `[B, 1, decoder_num_units]`
            aw_step (chainer.Variable): A tensor of size `[B, T_in, num_heads]`
        Returns:
            context_vec (chainer.Variable): A tensor of size
                `[B, 1, encoder_num_units]`
            aw_step (chainer.Variable): A tensor of size `[B, T_in, num_heads]`
        """
        batch_size, max_time = enc_out.shape[:2]

        energy = []
        for h in range(self.num_heads):
            if self.attention_type == 'content':
                ###################################################################
                # energy = <v, tanh(W([h_de; h_en] + b))>
                ###################################################################
                dec_out = F.broadcast_to(
                    dec_out, (dec_out.shape[0], enc_out.shape[1], dec_out.shape[2]))
                energy_head = F.squeeze(getattr(self, 'V_head' + str(h))(F.tanh(
                    getattr(self, 'W_enc_head' + str(h))(enc_out) +
                    getattr(self, 'W_dec_head' + str(h))(dec_out))), axis=2)
                energy.append(energy_head)

            elif self.attention_type == 'location':
                ###################################################################
                # f = F * α_{i-1}
                # energy = <v, tanh(W([h_de; h_en] + W_conv(f) + b))>
                ###################################################################
                # For 1D conv
                # conv_feat = getattr(self, 'conv_head' + str(h))(
                #     aw_step[:, :, h].reshape(batch_size, 1, max_time))

                # For 2D conv
                conv_feat = F.squeeze(getattr(self, 'conv_head' + str(h))(
                    aw_step[:, :, h].reshape(batch_size, 1, 1, max_time)), axis=2)
                # -> `[B, out_channels, T_in]`
                conv_feat = conv_feat.transpose(0, 2, 1)
                # -> `[B, T_in, out_channels]`

                dec_out = F.broadcast_to(
                    dec_out, (dec_out.shape[0], enc_out.shape[1], dec_out.shape[2]))
                energy_head = F.squeeze(getattr(self, 'V_head' + str(h))(F.tanh(
                    getattr(self, 'W_enc_head' + str(h))(enc_out) +
                    getattr(self, 'W_dec_head' + str(h))(dec_out) +
                    getattr(self, 'W_conv_head' + str(h))(conv_feat))), axis=2)
                energy.append(energy_head)

            elif self.attention_type == 'dot_product':
                ###################################################################
                # energy = <W_enc(h_en), W_dec(h_de)>
                ###################################################################
                energy_head = F.squeeze(F.matmul(
                    getattr(self, 'W_enc_head' + str(h))(enc_out),
                    getattr(self, 'W_dec_head' + str(h))(dec_out).transpose(0, 2, 1)), axis=2)
                energy.append(energy_head)

            elif self.attention_type == 'rnn_attention':
                raise NotImplementedError

            else:
                raise NotImplementedError

        context_vec = []
        aw_step = []
        for h in range(self.num_heads):
            # Mask attention distribution
            xp = cuda.get_array_module(enc_out)
            energy_mask = xp.ones((batch_size, max_time), dtype=np.float32)
            # energy_mask = xp.ones((batch_size, max_time), dtype=xp.float32)
            for b in range(batch_size):
                # TODO: fix bugs
                # if x_lens[b].data < max_time:
                if x_lens[b] < max_time:
                    energy_mask[b, x_lens[b]:] = 0
            energy_mask = Variable(energy_mask)
            energy[h] *= energy_mask
            # NOTE: energy[h] : `[B, T_in]`

            # Sharpening
            energy[h] = energy[h] * self.sharpening_factor

            # Compute attention weights
            if self.sigmoid_smoothing:
                aw_step_head = F.sigmoid(energy[h])
                # for b in range(batch_size):
                #     aw_step_head.data[b] /= aw_step_head.data[b].sum()
            else:
                aw_step_head = F.softmax(energy[h], axis=1)
            aw_step.append(aw_step_head)

            # Compute context vector (weighted sum of encoder outputs)
            batch_size, max_time = aw_step_head.shape
            context_vec_head = F.sum(enc_out * F.broadcast_to(
                F.expand_dims(aw_step_head, axis=2),
                (batch_size, max_time, enc_out.shape[-1])), axis=1, keepdims=True)
            context_vec.append(context_vec_head)

        # Concatenate all convtext vectors and attention distributions
        context_vec = F.concat(context_vec, axis=-1)
        aw_step = F.concat(aw_step, axis=-1).reshape(
            batch_size, -1, self.num_heads)

        if self.num_heads > 1:
            context_vec = getattr(self, 'W_mha')(context_vec)

        return context_vec, aw_step
