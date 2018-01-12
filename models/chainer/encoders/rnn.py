#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""(Hierarchical) RNN encoders (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import initializers

from models.chainer.linear import LinearND
# from models.chainer.encoders.cnn import CNNEncoder
# from models.chainer.encoders.cnn_v2 import CNNEncoder
# from models.chainer.encoders.cnn_utils import ConvOutSize


class RNNEncoder(chainer.Chain):
    """RNN encoder.
    Args:
        input_size (int): the dimension of input features
        rnn_type (string): lstm or gru or rnn
        bidirectional (bool): if True, use the bidirectional encoder
        num_units (int): the number of units in each layer
        num_proj (int): the number of nodes in the projection layer
        num_layers (int): the number of layers
        dropout (float): the probability to drop nodes
        parameter_init (float): the range of uniform distribution to
            initialize weight parameters (>= 0)
        subsample_list (list): subsample in the corresponding layers (True)
            ex.) [False, True, True, False] means that downsample is conducted
                in the 2nd and 3rd layers.
        subsample_type (string, optional): drop or concat
        use_cuda (bool, optional): if True, use GPUs
        merge_bidirectional (bool, optional): if True, sum bidirectional outputs
        num_stack (int, optional): the number of frames to stack
        splice (int, optional): frames to splice. Default is 1 frame.
        conv_channels (list, optional): the number of channles in CNN layers
        conv_kernel_sizes (list, optional): the size of kernels in CNN layers
        conv_strides (list, optional): strides in CNN layers
        poolings (list, optional): the size of poolings in CNN layers
        activation (string, optional): The activation function of CNN layers.
            Choose from relu or prelu or hard_tanh or maxout
        batch_norm (bool, optional):
        residual (bool, optional):
        dense_residual (bool, optional):
        num_layers_sub (int): the number of layers in the sub task

        # use_peephole (bool): if True, use peephole connections
        # clip_activation (float): the range of activation clipping (> 0)
    """

    def __init__(self,
                 input_size,
                 rnn_type,
                 bidirectional,
                 num_units,
                 num_proj,
                 num_layers,
                 dropout,
                 parameter_init,
                 subsample_list=[],
                 subsample_type='concat',
                 use_cuda=False,
                 merge_bidirectional=False,
                 num_stack=1,
                 splice=1,
                 conv_channels=[],
                 conv_kernel_sizes=[],
                 conv_strides=[],
                 poolings=[],
                 activation='relu',
                 batch_norm=False,
                 residual=False,
                 dense_residual=False,
                 num_layers_sub=0):

        super(RNNEncoder, self).__init__()

        if len(subsample_list) > 0 and len(subsample_list) != num_layers:
            raise ValueError(
                'subsample_list must be the same size as num_layers.')
        if subsample_type not in ['drop', 'concat']:
            raise TypeError('subsample_type must be "drop" or "concat".')
        if num_layers_sub < 0 or (num_layers_sub > 1 and num_layers < num_layers_sub):
            raise ValueError(
                'Set num_layers_sub between 1 to num_layers.')

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_units = num_units
        self.num_proj = num_proj if num_proj is not None else 0
        self.num_layers = num_layers
        self.merge_bidirectional = merge_bidirectional

        # TODO:
        # self.use_peephole = use_peephole
        # self.clip_activation = clip_activation

        # Setting for hierarchical encoder
        self.num_layers_sub = num_layers_sub

        # Setting for subsampling
        if len(subsample_list) == 0:
            self.subsample_list = [False] * num_layers
        else:
            self.subsample_list = subsample_list
        self.subsample_type = subsample_type
        # This implementation is bases on
        # https://arxiv.org/abs/1508.01211
        #     Chan, William, et al. "Listen, attend and spell."
        #         arXiv preprint arXiv:1508.01211 (2015).

        # Setting for residual connection
        assert not (residual and dense_residual)
        self.residual = residual
        self.dense_residual = dense_residual
        subsample_last_layer = 0
        for i_layer_reverse, is_subsample in enumerate(subsample_list[::-1]):
            if is_subsample:
                subsample_last_layer = num_layers - i_layer_reverse
                break
        self.residual_start_layer = subsample_last_layer + 1
        # NOTE: このレイヤの出力からres_outputs_listに入れていく

        with self.init_scope():
            initializer = initializers.Uniform(scale=parameter_init)

            # Setting for CNNs before RNNs# Setting for CNNs before RNNs
            if len(conv_channels) > 0 and len(conv_channels) == len(conv_kernel_sizes) and len(conv_kernel_sizes) == len(conv_strides):
                raise NotImplementedError

                # assert num_stack == 1
                # assert splice == 1
                # self.conv = CNNEncoder(
                #     input_size,
                #     conv_channels=conv_channels,
                #     conv_kernel_sizes=conv_kernel_sizes,
                #     conv_strides=conv_strides,
                #     poolings=poolings,
                #     dropout=dropout,
                #     parameter_init=parameter_init,
                #     activation=activation,
                #     use_cuda=use_cuda,
                #     batch_norm=batch_norm)
                # input_size = self.conv.output_size
                # self.get_conv_out_size = ConvOutSize(self.conv.conv)
            else:
                input_size = input_size * splice * num_stack
                self.conv = None

            self.rnns = []
            self.projections = []

            for i_layer in range(num_layers):
                if i_layer == 0:
                    encoder_input_size = input_size
                elif self.num_proj > 0:
                    encoder_input_size = num_proj
                    if subsample_type == 'concat' and i_layer > 0 and self.subsample_list[i_layer - 1]:
                        encoder_input_size *= 2
                else:
                    encoder_input_size = num_units * self.num_directions
                    if subsample_type == 'concat' and i_layer > 0 and self.subsample_list[i_layer - 1]:
                        encoder_input_size *= 2

                if rnn_type == 'lstm':
                    if bidirectional:
                        rnn_i = L.NStepBiLSTM(
                            n_layers=1,
                            in_size=encoder_input_size,
                            out_size=num_units,
                            dropout=dropout,
                            initialW=initializer,
                            initial_bias=None)
                    else:
                        rnn_i = L.NStepLSTM(
                            n_layers=1,
                            in_size=encoder_input_size,
                            out_size=num_units,
                            dropout=dropout,
                            initialW=initializer,
                            initial_bias=None)

                elif rnn_type == 'gru':
                    if bidirectional:
                        rnn_i = L.NStepBiGRU(
                            n_layers=1,
                            in_size=encoder_input_size,
                            out_size=num_units,
                            dropout=dropout,
                            initialW=initializer,
                            initial_bias=None)
                    else:
                        rnn_i = L.NStepGRU(
                            n_layers=1,
                            in_size=encoder_input_size,
                            out_size=num_units,
                            dropout=dropout,
                            initialW=initializer,
                            initial_bias=None)

                elif rnn_type == 'rnn':
                    if bidirectional:
                        # rnn_i = L.NStepBiRNNReLU(
                        rnn_i = L.NStepBiRNNTanh(
                            n_layers=1,
                            in_size=encoder_input_size,
                            out_size=num_units,
                            dropout=dropout,
                            initialW=initializer,
                            initial_bias=None)
                    else:
                        # rnn_i = L.NStepRNNReLU(
                        rnn_i = L.NStepRNNTanh(
                            n_layers=1,
                            in_size=encoder_input_size,
                            out_size=num_units,
                            dropout=dropout,
                            initialW=initializer,
                            initial_bias=None)
                else:
                    raise ValueError(
                        'rnn_type must be "lstm" or "gru" or "rnn".')

                if self.subsample_list[i_layer]:
                    setattr(self, 'p' + rnn_type + '_l' + str(i_layer), rnn_i)
                else:
                    setattr(self, rnn_type + '_l' + str(i_layer), rnn_i)
                if use_cuda:
                    rnn_i.to_gpu()
                self.rnns.append(rnn_i)

                if i_layer != self.num_layers - 1 and self.num_proj > 0:
                    proj_i = LinearND(
                        num_units * self.num_directions, num_proj,
                        dropout=dropout,
                        initializer=initializers.Uniform(scale=parameter_init))
                    setattr(self, 'proj_l' + str(i_layer), proj_i)
                    if use_cuda:
                        proj_i.to_gpu()
                    self.projections.append(proj_i)

    def __call__(self, xs, x_lens):
        """Forward computation.
        Args:
            xs (list of chainer.Variable): A list of tensors of size
                `[T, input_size]`, of length '[B]'
            x_lens (list): A list of length `[B]`
        Returns:
            xs (list of chainer.Variable):
                A list of tensors of size
                    `[T // sum(subsample_list), num_units (* num_directions)]`,
                    of length '[B]'
            OPTION:
            xs_sub (list of chainer.Variable):
                A list of tensor of size
                    `[T // sum(subsample_list), num_units (* num_directions)]`,
                    of length `[B]`
        """
        # NOTE: automatically sort xs in descending order by length,
        # and transpose the sequence

        # Path through CNN layers before RNN layers
        # if self.conv is not None:
        #     xs = self.conv(xs)

        # Modify x_lens for reducing time resolution by CNN layers
        # if self.conv is not None:
        #     x_lens = [self.get_conv_out_size(
        #         x, 1) for x in x_lens]
        #     max_time = self.get_conv_out_size(max_time, 1)

        res_outputs_list = []
        # NOTE: exclude residual connection from the raw inputs
        for i_layer in range(self.num_layers):

            if self.rnn_type == 'lstm':
                _, _, xs = self.rnns[i_layer](hx=None, cx=None, xs=xs)
            else:
                _, xs = self.rnns[i_layer](hx=None, xs=xs)

            # Pick up outputs in the sub task before the projection layer
            if self.num_layers_sub >= 1 and i_layer == self.num_layers_sub - 1:
                xs_sub = xs

            # NOTE: Exclude the last layer
            if i_layer != self.num_layers - 1:
                if self.residual or self.dense_residual or self.num_proj > 0 or self.subsample_list[i_layer]:

                    # Projection layer (affine transformation)
                    if self.num_proj > 0:
                        # Convert to 2D tensor
                        xs = F.vstack(xs)
                        xs = F.tanh(self.projections[i_layer](xs))
                        # Reshape back to 3D tensor
                        xs = F.split_axis(xs, np.cumsum(x_lens)[:-1], axis=0)

                    # Subsampling
                    if self.subsample_list[i_layer]:
                        # Pick up features at odd time step
                        if self.subsample_type == 'drop':
                            xs = [x[::2, :] for x in xs]

                        # Concatenate the successive frames
                        elif self.subsample_type == 'concat':
                            xs = [F.vstack([F.concat([x[t - 1:t, :], x[t:t + 1, :]], axis=-1)
                                            for t in range(x.shape[0]) if (t + 1) % 2 == 0])
                                  for x in xs]

                        # Update x_lens
                        x_lens = [x.shape[0] for x in xs]

                    # Residual connection
                    elif self.residual or self.dense_residual:
                        if i_layer >= self.residual_start_layer - 1:
                            for xs_lower in res_outputs_list:
                                xs = [x + x_l for x,
                                      x_l in zip(xs, xs_lower)]
                            if self.residual:
                                res_outputs_list = [xs]
                            elif self.dense_residual:
                                res_outputs_list.append(xs)

        # Sum bidirectional outputs
        if self.bidirectional and self.merge_bidirectional:
            xs = [x[:, :self.num_units] + x[:, self.num_units:] for x in xs]

        # sub task (optional)
        if self.num_layers_sub >= 1:
            # Sum bidirectional outputs
            if self.bidirectional and self.merge_bidirectional:
                xs_sub = [x[:, :self.num_units] + x[:, self.num_units:]
                          for x in xs_sub]
            return xs, xs_sub
        else:
            return xs
