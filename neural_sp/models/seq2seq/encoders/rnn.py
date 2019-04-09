#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""(Hierarchical) RNN encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from neural_sp.models.model_utils import LinearND
from neural_sp.models.seq2seq.encoders.cnn import CNNEncoder


class RNNEncoder(nn.Module):
    """RNN encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        rnn_type (str): blstm or lstm or bgru or gru
        n_units (int): number of units in each layer
        n_projs (int): number of units in each projection layer
        n_layers (int): number of layers
        dropout_in (float): probability to drop nodes in input-hidden connection
        dropout (float): probability to drop nodes in hidden-hidden connection
        subsample (list): subsample in the corresponding RNN layers
            ex.) [False, True, True, False] means that subsample is conducted in the 2nd and 3rd layers.
        subsample_type (str): drop or concat or max_pool
        n_stacks (int): number of frames to stack
        n_splices (int): frames to splice. Default is 1 frame.
        conv_in_channel (int): number of channels of input features
        conv_channels (int): number of channles in the CNN layers
        conv_kernel_sizes (list): size of kernels in the CNN layers
        conv_strides (list): number of strides in the CNN layers
        conv_poolings (list): size of poolings in the CNN layers
        conv_batch_norm (bool): apply batch normalization only in the CNN layers
        conv_bottleneck_dim (int): dimension of the bottleneck layer between CNN and RNN layers
        residual (bool): add residual connections between the consecutive layers
        n_layers_sub1 (int): number of layers in the 1st auxiliary task
        n_layers_sub2 (int): number of layers in the 2nd auxiliary task
        n_layers_sub3 (int): number of layers in the 3rd auxiliary task
        nin (int): if larger than 0, insert 1*1 conv (filter size: nin)
            and ReLU activation between each LSTM layer
        layer_norm (bool): layer normalization
        task_specific_layer (bool):

    """

    def __init__(self,
                 input_dim,
                 rnn_type,
                 n_units,
                 n_projs,
                 n_layers,
                 dropout_in,
                 dropout,
                 subsample,
                 subsample_type,
                 n_stacks,
                 n_splices,
                 conv_in_channel,
                 conv_channels,
                 conv_kernel_sizes,
                 conv_strides,
                 conv_poolings,
                 conv_batch_norm,
                 conv_bottleneck_dim,
                 residual,
                 n_layers_sub1=0,
                 n_layers_sub2=0,
                 n_layers_sub3=0,
                 nin=0,
                 layer_norm=False,
                 task_specific_layer=False):

        super(RNNEncoder, self).__init__()

        if len(subsample) > 0 and len(subsample) != n_layers:
            raise ValueError('subsample must be the same size as n_layers.')
        if n_layers_sub1 < 0 or (n_layers_sub1 > 1 and n_layers < n_layers_sub1):
            raise ValueError('Set n_layers_sub1 between 1 to n_layers.')
        if n_layers_sub2 < 0 or (n_layers_sub2 > 1 and n_layers_sub1 < n_layers_sub2):
            raise ValueError('Set n_layers_sub2 between 1 to n_layers_sub1.')
        if n_layers_sub3 < 0 or (n_layers_sub3 > 1 and n_layers_sub2 < n_layers_sub3):
            raise ValueError('Set n_layers_sub3 between 1 to n_layers_sub2.')
        if rnn_type == 'cnn':
            assert n_stacks == 1 and n_splices == 1

        self.rnn_type = rnn_type
        self.bidirectional = True if rnn_type in ['blstm', 'bgru'] else False
        self.n_units = n_units
        self.n_dirs = 2 if self.bidirectional else 1
        self.n_projs = n_projs
        self.n_layers = n_layers
        self.layer_norm = layer_norm
        self.residual = residual

        # Setting for hierarchical encoder
        self.n_layers_sub1 = n_layers_sub1
        self.n_layers_sub2 = n_layers_sub2
        self.n_layers_sub3 = n_layers_sub3
        self.task_specific_layer = task_specific_layer

        # Setting for subsampling
        if len(subsample) == 0:
            self.subsample = [1] * n_layers
        else:
            self.subsample = subsample
        self.subsample_type = subsample_type

        # Setting for residual connection
        subsample_last = 0
        for l_reverse, is_subsample in enumerate(subsample[::-1]):
            if is_subsample:
                subsample_last = n_layers - l_reverse
                break
        self.residual_start_layer = subsample_last + 1
        # NOTE: residual connection starts from the last subsampling layer

        # Setting for the NiN
        self.conv_batch_norm = conv_batch_norm
        self.nin = nin

        # Dropout for input-hidden connection
        self.dropout_in = nn.Dropout(p=dropout_in)

        # Setting for CNNs before RNNs
        if conv_poolings:
            channels = [int(c) for c in conv_channels.split('_')] if len(conv_channels) > 0 else []
            kernel_sizes = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                            for c in conv_kernel_sizes.split('_')] if len(conv_kernel_sizes) > 0 else []
            strides = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                       for c in conv_strides.split('_')] if len(conv_strides) > 0 else []
            poolings = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                        for c in conv_poolings.split('_')] if len(conv_poolings) > 0 else []
        else:
            channels = []
            kernel_sizes = []
            strides = []
            poolings = []

        if len(channels) > 0 and len(channels) == len(kernel_sizes) and len(kernel_sizes) == len(strides):
            # assert n_stacks == 1 and n_splices == 1
            self.conv = CNNEncoder(input_dim * n_stacks,
                                   in_channel=conv_in_channel * n_stacks,
                                   channels=channels,
                                   kernel_sizes=kernel_sizes,
                                   strides=strides,
                                   poolings=poolings,
                                   dropout=dropout,
                                   batch_norm=conv_batch_norm,
                                   bottleneck_dim=conv_bottleneck_dim)
            self._output_dim = self.conv.output_dim
        else:
            self._output_dim = input_dim * n_splices * n_stacks
            self.conv = None

        if rnn_type != 'cnn':
            self.fast_impl = False
            # Fast implementation without processes between each layer
            if np.prod(self.subsample) == 1 and self.n_projs == 0 and not residual and n_layers_sub1 == 0 and (not conv_batch_norm) and nin == 0:
                self.fast_impl = True
                if 'lstm' in rnn_type:
                    rnn = nn.LSTM
                elif 'gru' in rnn_type:
                    rnn = nn.GRU
                else:
                    raise ValueError('rnn_type must be "(b)lstm" or "(b)gru".')

                self.rnn = rnn(self._output_dim, n_units, n_layers,
                               bias=True,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=self.bidirectional)
                # NOTE: pytorch introduces a dropout layer on the outputs of each layer EXCEPT the last layer
                self._output_dim = n_units
                self.dropout_top = nn.Dropout(p=dropout)
            else:
                self.rnn = nn.ModuleList()
                self.dropout = nn.ModuleList()
                if self.n_projs > 0:
                    self.proj = nn.ModuleList()
                if subsample_type == 'max_pool' and np.prod(self.subsample) > 1:
                    self.max_pool = nn.ModuleList()
                    for l in range(n_layers):
                        if self.subsample[l] > 1:
                            self.max_pool += [nn.MaxPool2d((1, 1),
                                                           stride=(self.subsample[l], 1),
                                                           ceil_mode=True)]
                        else:
                            self.max_pool += [None]
                if subsample_type == 'concat' and np.prod(self.subsample) > 1:
                    self.concat = nn.ModuleList()
                    for l in range(n_layers):
                        if self.subsample[l] > 1:
                            self.concat += [LinearND(n_units * self.n_dirs * self.subsample[l], n_units * self.n_dirs)]
                        else:
                            self.concat += [None]

                for l in range(n_layers):
                    if 'lstm' in rnn_type:
                        rnn_i = nn.LSTM
                    elif 'gru' in rnn_type:
                        rnn_i = nn.GRU
                    else:
                        raise ValueError('rnn_type must be "(b)lstm" or "(b)gru".')

                    self.rnn += [rnn_i(self._output_dim, n_units, 1,
                                       bias=True,
                                       batch_first=True,
                                       dropout=0,
                                       bidirectional=self.bidirectional)]
                    self.dropout += [nn.Dropout(p=dropout)]
                    self._output_dim = n_units * self.n_dirs

                    # Projection layer
                    if n_projs > 0:
                        self.proj += [LinearND(n_units * self.n_dirs, n_projs)]
                        self._output_dim = n_projs

                    # Task specific layer
                    if l == n_layers_sub1 - 1 and task_specific_layer:
                        self.rnn_sub1_tsl = rnn_i(self._output_dim, n_units, 1,
                                                  bias=True,
                                                  batch_first=True,
                                                  dropout=0,
                                                  bidirectional=self.bidirectional)
                        self.dropout_sub1_tsl = nn.Dropout(p=dropout)
                    if l == n_layers_sub2 - 1 and task_specific_layer:
                        self.rnn_sub2_tsl = rnn_i(self._output_dim, n_units, 1,
                                                  bias=True,
                                                  batch_first=True,
                                                  dropout=0,
                                                  bidirectional=self.bidirectional)
                        self.dropout_sub2_tsl = nn.Dropout(p=dropout)
                    if l == n_layers_sub3 - 1 and task_specific_layer:
                        self.rnn_sub3_tsl = rnn_i(self._output_dim, n_units, 1,
                                                  bias=True,
                                                  batch_first=True,
                                                  dropout=0,
                                                  bidirectional=self.bidirectional)
                        self.dropout_sub3_tsl = nn.Dropout(p=dropout)

                    # Network in network (1*1 conv)
                    if nin > 0:
                        setattr(self, 'nin_l' + str(l),
                                nn.Conv1d(in_channels=self._output_dim,
                                          out_channels=nin,
                                          kernel_size=1,
                                          stride=1,
                                          padding=1,
                                          bias=not conv_batch_norm))

                        # Batch normalization
                        if conv_batch_norm:
                            if nin:
                                setattr(self, 'bn_0_l' + str(l), nn.BatchNorm1d(self._output_dim))
                                setattr(self, 'bn_l' + str(l), nn.BatchNorm1d(nin))
                            else:
                                setattr(self, 'bn_l' + str(l), nn.BatchNorm1d(self._output_dim))
                        # NOTE* BN in RNN models is applied only after NiN

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, xs, xlens, task):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim]`
            xlens (list): `[B]`
            task (str): all or ys or ys_sub1 or ys_sub2 or ys_sub3
        Returns:
            eouts (dict):
                xs (FloatTensor): `[B, T // prod(subsample), n_units (*n_dirs)]`
                xlens (list): `[B]`
                xs_sub1 (FloatTensor): `[B, T // prod(subsample), n_units (*n_dirs)]`
                xlens_sub1 (list): `[B]`
                xs_sub2 (FloatTensor): `[B, T // prod(subsample), n_units (*n_dirs)]`
                xlens_sub2 (list): `[B]`

        """
        eouts = {'ys': {'xs': None, 'xlens': None},
                 'ys_sub1': {'xs': None, 'xlens': None},
                 'ys_sub2': {'xs': None, 'xlens': None},
                 'ys_sub3': {'xs': None, 'xlens': None}}

        # Dropout for inputs-hidden connection
        xs = self.dropout_in(xs)

        # Path through CNN layers before RNN layers
        if self.conv is not None:
            xs, xlens = self.conv(xs, xlens)
            if self.rnn_type == 'cnn':
                eouts['ys']['xs'] = xs
                eouts['ys']['xlens'] = xlens
                return eouts

        if self.fast_impl:
            self.rnn.flatten_parameters()
            # NOTE: this is necessary for multi-GPUs setting

            # Path through RNN
            xs = pack_padded_sequence(xs, xlens, batch_first=True)
            xs, _ = self.rnn(xs, hx=None)
            xs = pad_packed_sequence(xs, batch_first=True)[0]
            xs = self.dropout_top(xs)
        else:
            xs_lower = None
            for l in range(len(self.rnn)):
                self.rnn[l].flatten_parameters()
                # NOTE: this is necessary for multi-GPUs setting

                # Path through RNN
                xs = pack_padded_sequence(xs, xlens, batch_first=True)
                xs, _ = self.rnn[l](xs, hx=None)
                xs = pad_packed_sequence(xs, batch_first=True)[0]
                xs = self.dropout[l](xs)

                # Pick up outputs in the sub task before the projection layer
                if l == self.n_layers_sub1 - 1:
                    if self.task_specific_layer:
                        self.rnn_sub1_tsl.flatten_parameters()
                        xs_sub1 = pack_padded_sequence(xs, xlens, batch_first=True)
                        xs_sub1, _ = self.rnn_sub1_tsl(xs_sub1, hx=None)
                        xs_sub1 = pad_packed_sequence(xs_sub1, batch_first=True)[0]
                        xs_sub1 = self.dropout_sub1_tsl(xs_sub1)
                    else:
                        xs_sub1 = xs.clone()
                    xlens_sub1 = xlens[:]

                    if task == 'ys_sub1':
                        eouts[task]['xs'] = xs_sub1
                        eouts[task]['xlens'] = xlens_sub1
                        return eouts

                if l == self.n_layers_sub2 - 1:
                    if self.task_specific_layer:
                        self.rnn_sub2_tsl.flatten_parameters()
                        xs_sub2 = pack_padded_sequence(xs, xlens, batch_first=True)
                        xs_sub2, _ = self.rnn_sub2_tsl(xs_sub2, hx=None)
                        xs_sub2 = pad_packed_sequence(xs_sub2, batch_first=True)[0]
                        xs_sub2 = self.dropout_sub2_tsl(xs_sub2)
                    else:
                        xs_sub2 = xs.clone()
                    xlens_sub2 = xlens[:]

                    if task == 'ys_sub2':
                        eouts[task]['xs'] = xs_sub2
                        eouts[task]['xlens'] = xlens_sub2
                        return eouts

                if l == self.n_layers_sub3 - 1:
                    if self.task_specific_layer:
                        self.rnn_sub3_tsl.flatten_parameters()
                        xs_sub3 = pack_padded_sequence(xs, xlens, batch_first=True)
                        xs_sub3, _ = self.rnn_sub3_tsl(xs_sub3, hx=None)
                        xs_sub3 = pad_packed_sequence(xs_sub3, batch_first=True)[0]
                        xs_sub3 = self.dropout_sub3_tsl(xs_sub3)
                    else:
                        xs_sub3 = xs.clone()
                    xlens_sub3 = xlens[:]

                    if task == 'ys_sub3':
                        eouts[task]['xs'] = xs_sub3
                        eouts[task]['xlens'] = xlens_sub3
                        return eouts

                # Projection layer
                if self.n_projs > 0:
                    # xs = torch.tanh(self.proj[l](xs))
                    xs = self.proj[l](xs)

                # NOTE: Exclude the last layer
                if l != len(self.rnn) - 1:
                    # Subsampling
                    if self.subsample[l] > 1:
                        if self.subsample_type == 'drop':
                            xs = xs[:, 1::self.subsample[l], :]
                            # NOTE: Pick up features at even time step
                        elif self.subsample_type == 'concat':
                            # Concatenate the successive frames
                            xs = xs.transpose(1, 0).contiguous()
                            xs = [torch.cat([xs[t - r:t - r + 1] for r in range(self.subsample[l] - 1, -1, -1)], dim=-1)
                                  for t in range(xs.size(0)) if (t + 1) % self.subsample[l] == 0]
                            # NOTE: Exclude the last frame if the length of xs is odd
                            xs = torch.cat(xs, dim=0).transpose(1, 0)
                            xs = torch.tanh(self.concat[l](xs))
                        elif self.subsample_type == 'max_pool':
                            xs = xs.transpose(1, 0).contiguous()
                            xs = [torch.max(xs[t - self.subsample[l] + 1:t + 1], dim=0)[0].unsqueeze(0)
                                  for t in range(xs.size(0)) if (t + 1) % self.subsample[l] == 0]
                            xs = torch.cat(xs, dim=0).transpose(1, 0)

                        # Update xlens
                        xlens = [x.size(0) for x in xs]

                    # NiN
                    if self.nin > 0:
                        raise NotImplementedError()

                        # Batch normalization befor NiN
                        if self.conv_batch_norm:
                            size = list(xs.size())
                            xs = to2d(xs, size)
                            xs = getattr(self, 'bn_0_l' + str(l))(xs)
                            xs = F.relu(xs)
                            xs = to3d(xs, size)
                            # NOTE: mean and var are computed along all timesteps in the mini-batch

                        xs = getattr(self, 'nin_l' + str(l))(xs)

                    # Residual connection
                    if (not self.subsample[l]) and self.residual:
                        if l >= self.residual_start_layer - 1:
                            xs = xs + xs_lower
                        xs_lower = xs
                    # NOTE: Exclude residual connection from the raw inputs

        if task in ['all', 'ys']:
            eouts['ys']['xs'] = xs
            eouts['ys']['xlens'] = xlens
        if self.n_layers_sub1 >= 1 and task == 'all':
            eouts['ys_sub1']['xs'] = xs_sub1
            eouts['ys_sub1']['xlens'] = xlens_sub1
        if self.n_layers_sub2 >= 1 and task == 'all':
            eouts['ys_sub2']['xs'] = xs_sub2
            eouts['ys_sub2']['xlens'] = xlens_sub2
        if self.n_layers_sub3 >= 1 and task == 'all':
            eouts['ys_sub3']['xs'] = xs_sub3
            eouts['ys_sub3']['xlens'] = xlens_sub3

        return eouts


def to2d(xs, size):
    return xs.contiguous().view((int(np.prod(size[: -1])), int(size[-1])))


def to3d(xs, size):
    return xs.view(size)
