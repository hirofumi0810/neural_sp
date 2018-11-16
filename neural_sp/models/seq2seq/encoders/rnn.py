#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""(Hierarchical) RNN encoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from neural_sp.models.linear import LinearND
from neural_sp.models.seq2seq.encoders.cnn import CNNEncoder


class RNNEncoder(nn.Module):
    """RNN encoder.

    Args:
        input_dim (int): the dimension of input features  (freq * channel)
        rnn_type (str): blstm or lstm or bgru or gru
        num_units (int): the number of units in each RNN layer
        num_projs (int): the number of units in each projection layer after RNN layer
        num_layers (int): the number of RNN layers
        dropout_in (float): the probability to drop nodes in input-hidden connection
        dropout_hidden (float): the probability to drop nodes in hidden-hidden connection
        subsample (list): subsample in the corresponding RNN layers
            ex.) [False, True, True, False] means that subsample is conducted
                in the 2nd and 3rd layers.
        subsample_type (str): drop or concat
        batch_first (bool): if True, batch-major computation will be performed
        num_stack (int): the number of frames to stack
        num_skip (int): the number of frames to skip
        num_splice (int): frames to splice. Default is 1 frame.
        conv_in_channel (int): the number of channels of input features
        conv_channels (int): the number of channles in the CNN layers
        conv_kernel_sizes (list): the size of kernels in the CNN layers
        conv_strides (list): the number of strides in the CNN layers
        conv_poolings (list): the size of poolings in the CNN layers
        conv_batch_norm (bool): apply batch normalization only in the CNN layers
        residual (bool): add residual connections between RNN layers
        num_layers_sub (int): the number of layers in the sub task
        nin (int): if larger than 0, insert 1*1 conv (filter size: nin)
            and ReLU activation between each LSTM layer
        layer_norm (bool): layer normalization

    """

    def __init__(self,
                 input_dim,
                 rnn_type,
                 num_units,
                 num_projs,
                 num_layers,
                 dropout_in,
                 dropout_hidden,
                 subsample,
                 subsample_type,
                 batch_first,
                 num_stack,
                 num_splice,
                 conv_in_channel,
                 conv_channels,
                 conv_kernel_sizes,
                 conv_strides,
                 conv_poolings,
                 conv_batch_norm,
                 residual,
                 num_layers_sub=0,
                 nin=0,
                 layer_norm=False):

        super(RNNEncoder, self).__init__()

        if len(subsample) > 0 and len(subsample) != num_layers:
            raise ValueError('subsample must be the same size as num_layers.')
        if subsample_type not in ['drop', 'concat']:
            raise TypeError('subsample_type must be "drop" or "concat".')
        if num_layers_sub < 0 or (num_layers_sub > 1 and num_layers < num_layers_sub):
            raise ValueError('Set num_layers_sub between 1 to num_layers.')

        self.rnn_type = rnn_type
        self.bidirectional = True if rnn_type in ['blstm', 'bgru'] else False
        self.num_units = num_units
        self.num_directions = 2 if self.bidirectional else 1
        self.num_projs = num_projs if num_projs is not None else 0
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.layer_norm = layer_norm

        # Setting for hierarchical encoder
        self.num_layers_sub = num_layers_sub

        # Setting for subsampling
        if len(subsample) == 0:
            self.subsample = [False] * num_layers
        else:
            self.subsample = subsample
        self.subsample_type = subsample_type

        # Setting for residual connection
        self.residual = residual
        subsample_last_layer = 0
        for l_reverse, is_subsample in enumerate(subsample[::-1]):
            if is_subsample:
                subsample_last_layer = num_layers - l_reverse
                break
        self.residual_start_layer = subsample_last_layer + 1
        # NOTE: residual connection starts from the last subsampling layer

        # Setting for the NiN
        self.conv_batch_norm = conv_batch_norm
        self.nin = nin

        # Dropout for input-hidden connection
        self.dropout_in = nn.Dropout(p=dropout_in)

        # Setting for CNNs before RNNs
        if len(conv_channels) > 0 and len(conv_channels) == len(conv_kernel_sizes) and len(conv_kernel_sizes) == len(conv_strides):
            assert num_stack == 1 and num_splice == 1
            self.conv = CNNEncoder(input_dim,
                                   in_channel=conv_in_channel,
                                   channels=conv_channels,
                                   kernel_sizes=conv_kernel_sizes,
                                   strides=conv_strides,
                                   poolings=conv_poolings,
                                   dropout_in=0,
                                   dropout_hidden=dropout_hidden,
                                   activation='relu',
                                   batch_norm=conv_batch_norm)
            input_dim = self.conv.output_dim
        else:
            input_dim *= num_splice * num_stack
            self.conv = None

        self.fast_impl = False
        # Fast implementation without processes between each layer
        if sum(self.subsample) == 0 and self.num_projs == 0 and not residual and num_layers_sub == 0 and (not conv_batch_norm) and nin == 0:
            self.fast_impl = True
            if 'lstm' in rnn_type:
                rnn = nn.LSTM(input_dim,
                              hidden_size=num_units,
                              num_layers=num_layers,
                              bias=True,
                              batch_first=batch_first,
                              dropout=dropout_hidden,
                              bidirectional=self.bidirectional)
            elif 'gru' in rnn_type:
                rnn = nn.GRU(input_dim,
                             hidden_size=num_units,
                             num_layers=num_layers,
                             bias=True,
                             batch_first=batch_first,
                             dropout=dropout_hidden,
                             bidirectional=self.bidirectional)
            else:
                raise ValueError('rnn_type must be "(b)lstm" or "(b)gru".')

            setattr(self, rnn_type, rnn)
            # NOTE: pytorch introduces a dropout layer on the outputs of
            # each RNN layer EXCEPT the last layer

            # Dropout for hidden-output connection
            self.dropout_top = nn.Dropout(p=dropout_hidden)

        else:
            for i_l in six.moves.range(num_layers):
                if i_l == 0:
                    enc_in_dim = input_dim
                elif nin > 0:
                    enc_in_dim = nin
                elif self.num_projs > 0:
                    enc_in_dim = num_projs
                    if subsample_type == 'concat' and i_l > 0 and self.subsample[i_l - 1]:
                        enc_in_dim *= 2
                else:
                    enc_in_dim = num_units * self.num_directions
                    if subsample_type == 'concat' and i_l > 0 and self.subsample[i_l - 1]:
                        enc_in_dim *= 2

                if 'lstm' in rnn_type:
                    rnn_i = nn.LSTM(enc_in_dim,
                                    hidden_size=num_units,
                                    num_layers=1,
                                    bias=True,
                                    batch_first=batch_first,
                                    dropout=0,
                                    bidirectional=self.bidirectional)
                elif 'gru' in rnn_type:
                    rnn_i = nn.GRU(enc_in_dim,
                                   hidden_size=num_units,
                                   num_layers=1,
                                   bias=True,
                                   batch_first=batch_first,
                                   dropout=0,
                                   bidirectional=self.bidirectional)
                else:
                    raise ValueError('rnn_type must be "lstm" or "gru".')

                setattr(self, rnn_type + '_l' + str(i_l), rnn_i)
                enc_out_dim = num_units * self.num_directions
                # TODO(hirofumi): check this

                # Dropout for hidden-hidden or hidden-output connection
                setattr(self, 'dropout_l' + str(i_l), nn.Dropout(p=dropout_hidden))

                if i_l != self.num_layers - 1 and self.num_projs > 0:
                    proj_i = LinearND(num_units * self.num_directions, num_projs)
                    setattr(self, 'proj_l' + str(i_l), proj_i)
                    enc_out_dim = num_projs

                # Network in network (1*1 conv)
                if nin > 0:
                    setattr(self, 'nin_l' + str(i_l),
                            nn.Conv1d(in_channels=enc_out_dim,
                                      out_channels=nin,
                                      kernel_size=1,
                                      stride=1,
                                      padding=1,
                                      bias=not conv_batch_norm))

                    # Batch normalization
                    if conv_batch_norm:
                        if nin:
                            setattr(self, 'bn_0_l' + str(i_l), nn.BatchNorm1d(enc_out_dim))
                            setattr(self, 'bn_l' + str(i_l), nn.BatchNorm1d(nin))
                        elif subsample_type == 'concat' and self.subsample[i_l]:
                            setattr(self, 'bn_l' + str(i_l), nn.BatchNorm1d(enc_out_dim * 2))
                        else:
                            setattr(self, 'bn_l' + str(i_l), nn.BatchNorm1d(enc_out_dim))
                    # NOTE* BN in RNN models is applied only after NiN

    def forward(self, xs, x_lens):
        """Forward computation.

        Args:
            xs (torch.autograd.Variable, float): `[B, T, input_dim]`
            x_lens (list): `[B]`
        Returns:
            xs (torch.autograd.Variable, float):
                if batch_first is True
                    `[B, T // sum(subsample), num_units (* num_directions)]`
                else
                    `[T // sum(subsample), B, num_units (* num_directions)]`
            x_lens (list): `[B]`
            OPTION:
                xs_sub (torch.autograd.Variable, float):
                    if batch_first is True
                        `[B, T // sum(subsample), num_units (* num_directions)]`
                    else
                        `[T // sum(subsample), B, num_units (* num_directions)]`
                x_lens_sub (list): `[B]`

        """
        # Dropout for inputs-hidden connection
        xs = self.dropout_in(xs)

        # Path through CNN layers before RNN layers
        if self.conv is not None:
            xs, x_lens = self.conv(xs, x_lens)

        if not self.batch_first:
            # Convert to the time-major
            xs = xs.transpose(0, 1).contiguous()

        if self.fast_impl:
            getattr(self, self.rnn_type).flatten_parameters()
            # NOTE: this is necessary for multi-GPUs setting

            # Path through RNN
            xs = pack_padded_sequence(xs, x_lens, batch_first=self.batch_first)
            xs, _ = getattr(self, self.rnn_type)(xs, hx=None)
            xs, unpacked_seq_len = pad_packed_sequence(
                xs, batch_first=self.batch_first, padding_value=0)
            # assert x_lens == unpacked_seq_len

            # Dropout for hidden-output connection
            xs = self.dropout_top(xs)
        else:
            res_outputs = []
            for i_l in six.moves.range(self.num_layers):
                getattr(self, self.rnn_type + '_l' + str(i_l)).flatten_parameters()
                # NOTE: this is necessary for multi-GPUs setting

                # Path through RNN
                xs = pack_padded_sequence(xs, x_lens, batch_first=self.batch_first)
                xs, _ = getattr(self, self.rnn_type + '_l' + str(i_l))(xs, hx=None)
                xs, unpacked_seq_len = pad_packed_sequence(
                    xs, batch_first=self.batch_first, padding_value=0)
                # assert x_lens == unpacked_seq_len

                # Dropout for hidden-hidden or hidden-output connection
                xs = getattr(self, 'dropout_l' + str(i_l))(xs)

                # Pick up outputs in the sub task before the projection layer
                if self.num_layers_sub >= 1 and i_l == self.num_layers_sub - 1:
                    xs_sub = xs.clone()
                    x_lens_sub = copy.deepcopy(x_lens)

                # NOTE: Exclude the last layer
                if i_l != self.num_layers - 1:
                    # Subsampling
                    if self.subsample[i_l]:
                        if self.subsample_type == 'drop':
                            if self.batch_first:
                                xs = xs[:, 1::2, :]
                            else:
                                xs = xs[1::2, :, :]
                            # NOTE: Pick up features at EVEN time step

                        # Concatenate the successive frames
                        elif self.subsample_type == 'concat':
                            if self.batch_first:
                                xs = [torch.cat([xs[:, t - 1:t, :], xs[:, t:t + 1, :]], dim=2)
                                      for t in six.moves.range(xs.size(1)) if (t + 1) % 2 == 0]
                                xs = torch.cat(xs, dim=1)
                            else:
                                xs = [torch.cat([xs[t - 1:t, :, :], xs[t:t + 1, :, :]], dim=2)
                                      for t in six.moves.range(xs.size(0)) if (t + 1) % 2 == 0]
                                xs = torch.cat(xs, dim=0)
                            # NOTE: Exclude the last frame if the length of xs is odd

                        # Update x_lens
                        if self.batch_first:
                            x_lens = [x.size(0) for x in xs]
                        else:
                            x_lens = [xs[:, i].size(0) for i in six.moves.range(xs.size(1))]

                    # Projection layer (affine transformation)
                    if self.num_projs > 0:
                        xs = F.tanh(getattr(self, 'proj_l' + str(i_l))(xs))

                    # NiN
                    if self.nin > 0:
                        raise NotImplementedError()

                        # Batch normalization befor NiN
                        if self.conv_batch_norm:
                            size = list(xs.size())
                            xs = to2d(xs, size)
                            xs = getattr(self, 'bn_0_l' + str(i_l))(xs)
                            xs = F.relu(xs)
                            xs = to3d(xs, size)
                            # NOTE: mean and var are computed along all timesteps in the mini-batch

                        xs = getattr(self, 'nin_l' + str(i_l))(xs)

                    # Residual connection
                    if (not self.subsample[i_l]) and self.residual:
                        if i_l >= self.residual_start_layer - 1:
                            for xs_lower in res_outputs:
                                xs = xs + xs_lower
                            if self.residual:
                                res_outputs = [xs]
                    # NOTE: Exclude residual connection from the raw inputs

        if not self.batch_first:
            # Convert to the time-major
            xs = xs.transpose(0, 1).contiguous()

        # For the sub task
        if self.num_layers_sub >= 1:
            if not self.batch_first:
                # Convert to the time-major
                xs_sub = xs_sub.transpose(0, 1).contiguous()
            return xs, x_lens, xs_sub, x_lens_sub
        else:
            return xs, x_lens


def to2d(xs, size):
    return xs.contiguous().view((int(np.prod(size[:-1])), int(size[-1])))


def to3d(xs, size):
    return xs.view(size)
