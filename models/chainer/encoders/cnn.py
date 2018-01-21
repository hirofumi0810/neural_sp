#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""CNN encoder (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import chainer
from chainer import functions as F
from chainer import links as L

from utils.io.variable import np2var


class CNNEncoder(chainer.Chain):
    """CNN encoder.
    Args:
        input_size (int): the dimension of input features
        conv_channels (list, optional): the number of channles in CNN layers
        conv_kernel_sizes (list, optional): the size of kernels in CNN layers
        conv_strides (list, optional): strides in CNN layers
        poolings (list, optional): the size of poolings in CNN layers
        dropout (float): the probability to drop nodes
        use_cuda (bool, optional): if True, use GPUs
        activation (string, optional): relu or prelu or hard_tanh or maxout
        batch_norm (bool, optional):
    """

    def __init__(self,
                 input_size,
                 conv_channels,
                 conv_kernel_sizes,
                 conv_strides,
                 poolings,
                 dropout,
                 use_cuda,
                 activation='relu',
                 batch_norm=False):

        super(CNNEncoder, self).__init__()

        if input_size % 3 == 0:
            self.input_freq = input_size // 3
            self.input_channels = 3
        else:
            self.input_freq = input_size
            self.input_channels = 1

        self.conv_channels = conv_channels
        self.poolings = poolings
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.activation = activation
        self.batch_norm = batch_norm

        assert len(conv_channels) > 0
        assert len(conv_channels) == len(conv_kernel_sizes)
        assert len(conv_kernel_sizes) == len(conv_strides)
        assert len(conv_strides) == len(poolings)

        with self.init_scope():
            in_c = self.input_channels
            in_freq = self.input_freq
            for i_layer in range(len(conv_channels)):

                # Conv
                conv = L.Convolution2D(in_channels=in_c,
                                       out_channels=conv_channels[i_layer],
                                       ksize=tuple(conv_kernel_sizes[i_layer]),
                                       stride=tuple(conv_strides[i_layer]),
                                       pad=tuple(conv_strides[i_layer]),
                                       nobias=batch_norm,
                                       initialW=None,
                                       initial_bias=None)
                setattr(self, 'conv_l' + str(i_layer), conv)
                in_freq = chainer.utils.get_conv_outsize(in_freq,
                                                         k=conv_kernel_sizes[i_layer][0],
                                                         s=conv_strides[i_layer][0],
                                                         p=conv_strides[i_layer][0])
                # NOTE: this is frequency-dimension

                # Max Pooling
                if len(poolings[i_layer]) > 0:
                    in_freq = chainer.utils.get_conv_outsize(in_freq,
                                                             k=poolings[i_layer][0],
                                                             s=poolings[i_layer][0],
                                                             p=0)

                # Batch Normalization
                if batch_norm:
                    setattr(self, 'bn_l' + str(i_layer),
                            L.BatchNormalization(conv_channels[i_layer]))

                in_c = conv_channels[i_layer]

        self.get_conv_out_size = ConvOutSize(conv_kernel_sizes,
                                             conv_strides,
                                             poolings)
        self.output_size = conv_channels[-1] * in_freq

    def __call__(self, xs, x_lens):
        """Forward computation.
        Args:
            xs (chainer.Variable): A tensor of size `[B, T, input_size]`
            x_lens (np.ndarray or chainer.Variable): A tensor of size `[B]`
        Returns:
            xs (chainer.Variable): A tensor of size `[B, T', feature_dim]`
            x_lens (np.ndarray or chainer.Variable): A tensor of size `[B]`
        """
        wrap_by_var = isinstance(x_lens, chainer.Variable)

        # Convert to Variable
        if isinstance(xs, list):
            xs = F.pad_sequence(xs, padding=0)

        batch_size, max_time, input_size = xs.shape

        # assert input_size == self.input_freq * self.input_channels

        # Reshape to 4D tensor
        xs = F.swapaxes(xs, 1, 2)
        if self.input_channels == 3:
            xs = xs.reshape(batch_size, 3, input_size // 3, max_time)
            # NOTE: xs: `[B, in_ch (3), freq // 3, max_time]`
        else:
            xs = F.expand_dims(xs, axis=1)
            # NOTE: xs: `[B, in_ch (1), freq, max_time]`

        for i_layer in range(len(self.conv_channels)):

            # Conv
            xs = getattr(self, 'conv_l' + str(i_layer))(xs)

            # Activation
            if self.activation == 'relu':
                xs = F.relu(xs)
            elif self.activation == 'prelu':
                raise NotImplementedError
                # xs = F.prelu(xs, W)
                # layers.append(F.PReLU(num_parameters=1, init=0.2))
            elif self.activation == 'hard_tanh':
                raise NotImplementedError
            elif self.activation == 'maxout':
                raise NotImplementedError
                # xs = F.maxout(xs, pool_size=1, axis=1)
            else:
                raise NotImplementedError

            # Max Pooling
            if len(self.poolings[i_layer]) > 0:
                xs = F.max_pooling_2d(xs,
                                      ksize=tuple(self.poolings[i_layer]),
                                      stride=tuple(self.poolings[i_layer]),
                                      # pad=(1, 1),
                                      pad=(0, 0),  # default
                                      cover_all=False)
                # NOTE: If cover_all is False, remove last feature when the
                # dimension of features are odd.

            # Batch Normalization
            if self.batch_norm:
                xs = getattr(self, 'bn_l' + str(i_layer))(xs)

            # Dropout
            if self.dropout > 0:
                xs = F.dropout(xs, ratio=self.dropout)
            # TODO: compare BN before ReLU and after ReLU

            # print(xs.shape)
            # NOTE: xs: `[B, out_ch, new_freq, new_time]`

        # Collapse feature dimension
        output_channels, freq, time = xs.shape[1:]
        xs = xs.transpose(0, 3, 2, 1)
        xs = xs.reshape(batch_size, time, freq * output_channels)

        if wrap_by_var:
            # Update x_lens
            x_lens = [self.get_conv_out_size(x.data, 1) for x in x_lens]

            # Wrap by Variable again
            x_lens = np2var(x_lens, use_cuda=self.use_cuda, backend='chainer')
        else:
            # Update x_lens
            x_lens = [self.get_conv_out_size(x, 1) for x in x_lens]

            # Convert to list again
        xs = F.separate(xs, axis=0)

        return xs, x_lens


class ConvOutSize(object):
    """TODO."""

    def __init__(self, conv_kernel_sizes, conv_strides, poolings):
        super(ConvOutSize, self).__init__()

        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_strides = conv_strides
        self.poolings = poolings

    def __call__(self, size, dim):
        """
        Args:
            size (int):
            dim (int): dim == 0 means frequency dimension, dim == 1 means
                time dimension.
        Returns:
            size (int):
        """
        for i_layer in range(len(self.conv_kernel_sizes)):
            size = chainer.utils.get_conv_outsize(
                size,
                k=self.conv_kernel_sizes[i_layer][dim],
                s=self.conv_strides[i_layer][dim],
                p=self.conv_strides[i_layer][dim])

            if len(self.poolings[i_layer]) > 0:
                size = chainer.utils.get_conv_outsize(
                    size,
                    k=self.poolings[i_layer][dim],
                    s=self.poolings[i_layer][dim],
                    p=0)

        # assert size >= 1
        return size
