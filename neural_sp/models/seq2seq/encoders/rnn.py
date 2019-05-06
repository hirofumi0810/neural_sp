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

from neural_sp.models.modules.linear import LinearND
from neural_sp.models.seq2seq.encoders.cnn import ConvEncoder
from neural_sp.models.seq2seq.encoders.time_depth_separable_conv import TDSEncoder


class RNNEncoder(nn.Module):
    """RNN encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        rnn_type (str): blstm or lstm or bgru or gru or cnn or tds
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
        conv_channels (int): number of channles in the CNN blocks
        conv_kernel_sizes (list): size of kernels in the CNN blocks
        conv_strides (list): number of strides in the CNN blocks
        conv_poolings (list): size of poolings in the CNN blocks
        conv_batch_norm (bool): apply batch normalization only in the CNN blocks
        conv_residual (bool): add residual connection between each CNN block
        conv_bottleneck_dim (int): dimension of the bottleneck layer between CNN and RNN layers
        residual (bool): add residual connections between the consecutive layers
        n_layers_sub1 (int): number of layers in the 1st auxiliary task
        n_layers_sub2 (int): number of layers in the 2nd auxiliary task
        nin (bool): insert 1*1 conv + batch normalization + ReLU
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
                 subsample_type='drop',
                 n_stacks=1,
                 n_splices=1,
                 conv_in_channel=1,
                 conv_channels=0,
                 conv_kernel_sizes=[],
                 conv_strides=[],
                 conv_poolings=[],
                 conv_batch_norm=False,
                 conv_residual=False,
                 conv_bottleneck_dim=0,
                 residual=False,
                 n_layers_sub1=0,
                 n_layers_sub2=0,
                 nin=False,
                 layer_norm=False,
                 task_specific_layer=False):

        super(RNNEncoder, self).__init__()

        if len(subsample) > 0 and len(subsample) != n_layers:
            raise ValueError('subsample must be the same size as n_layers.')
        if n_layers_sub1 < 0 or (n_layers_sub1 > 1 and n_layers < n_layers_sub1):
            raise ValueError('Set n_layers_sub1 between 1 to n_layers.')
        if n_layers_sub2 < 0 or (n_layers_sub2 > 1 and n_layers_sub1 < n_layers_sub2):
            raise ValueError('Set n_layers_sub2 between 1 to n_layers_sub1.')

        self.rnn_type = rnn_type
        self.bidirectional = True if rnn_type in ['blstm', 'bgru'] else False
        self.n_units = n_units
        self.n_dirs = 2 if self.bidirectional else 1
        self.n_projs = n_projs
        self.n_layers = n_layers
        self.layer_norm = layer_norm

        # Setting for hierarchical encoder
        self.n_layers_sub1 = n_layers_sub1
        self.n_layers_sub2 = n_layers_sub2
        self.task_specific_layer = task_specific_layer

        # Setting for subsampling
        self.subsample = subsample
        self.subsample_type = subsample_type

        # Setting for residual connections
        self.residual = residual
        if residual:
            assert np.prod(subsample) == 1

        # Setting for the NiN (Network in Network)
        self.nin = nin

        # Dropout for input-hidden connection
        self.dropout_in = nn.Dropout(p=dropout_in)

        # Setting for CNNs before RNNs
        if conv_channels:
            channels = [int(c) for c in conv_channels.split('_')] if len(conv_channels) > 0 else []
            kernel_sizes = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                            for c in conv_kernel_sizes.split('_')] if len(conv_kernel_sizes) > 0 else []
            if rnn_type == 'tds':
                strides = []
                poolings = []
            else:
                strides = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                           for c in conv_strides.split('_')] if len(conv_strides) > 0 else []
                poolings = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                            for c in conv_poolings.split('_')] if len(conv_poolings) > 0 else []
        else:
            channels = []
            kernel_sizes = []
            strides = []
            poolings = []

        if len(channels) > 0:
            assert n_stacks == 1 and n_splices == 1
            if rnn_type == 'tds':
                self.conv = TDSEncoder(input_dim=input_dim,
                                       in_channel=conv_in_channel,
                                       channels=channels,
                                       kernel_sizes=kernel_sizes,
                                       dropout=dropout)
            else:
                self.conv = ConvEncoder(input_dim,
                                        in_channel=conv_in_channel,
                                        channels=channels,
                                        kernel_sizes=kernel_sizes,
                                        strides=strides,
                                        poolings=poolings,
                                        dropout=0,
                                        batch_norm=conv_batch_norm,
                                        residual=conv_residual,
                                        bottleneck_dim=conv_bottleneck_dim)
            self._output_dim = self.conv.output_dim
        else:
            self._output_dim = input_dim * n_splices * n_stacks
            self.conv = None

        if rnn_type not in ['cnn', 'tds']:
            # Fast implementation without processes between each layer
            self.fast_impl = False
            if np.prod(self.subsample) == 1 and self.n_projs == 0 and not residual and n_layers_sub1 == 0 and not nin:
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
                self._output_dim = n_units * self.n_dirs
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
                    self.concat_proj = nn.ModuleList()
                    self.concat_bn = nn.ModuleList()
                    for l in range(n_layers):
                        if self.subsample[l] > 1:
                            self.concat_proj += [LinearND(n_units * self.n_dirs
                                                          * self.subsample[l], n_units * self.n_dirs)]
                            self.concat_bn += [nn.BatchNorm2d(n_units * self.n_dirs)]
                        else:
                            self.concat_proj += [None]
                            self.concat_bn += [None]
                if nin:
                    self.nin_conv = nn.ModuleList()
                    self.nin_bn = nn.ModuleList()

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

                    # Network in network (1*1 conv + batch normalization + ReLU)
                    # NOTE: exclude the last layer
                    if nin and l != n_layers - 1:
                        self.nin_conv += [nn.Conv2d(in_channels=self._output_dim,
                                                    out_channels=self._output_dim,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)]
                        self.nin_bn += [nn.BatchNorm2d(self._output_dim)]
                        if n_layers_sub1 > 0 or n_layers_sub2 > 0:
                            assert task_specific_layer

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, xs, xlens, task):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim]`
            xlens (list): `[B]`
            task (str): all or ys or ys_sub1 or ys_sub2
        Returns:
            eouts (dict):
                xs (FloatTensor): `[B, T // prod(subsample), n_units (*2)]`
                xlens (list): `[B]`
                xs_sub1 (FloatTensor): `[B, T // prod(subsample), n_units (*2)]`
                xlens_sub1 (list): `[B]`
                xs_sub2 (FloatTensor): `[B, T // prod(subsample), n_units (*2)]`
                xlens_sub2 (list): `[B]`

        """
        eouts = {'ys': {'xs': None, 'xlens': None},
                 'ys_sub1': {'xs': None, 'xlens': None},
                 'ys_sub2': {'xs': None, 'xlens': None}}

        # Sort by lenghts in the descending order for pack_padded_sequence
        xlens, perm_ids = torch.LongTensor(xlens).sort(0, descending=True)
        xs = xs[perm_ids]
        _, perm_ids_unsort = perm_ids.sort()

        # Dropout for inputs-hidden connection
        xs = self.dropout_in(xs)

        # Path through CNN blocks before RNN layers
        if self.conv is not None:
            xs, xlens = self.conv(xs, xlens)
            if self.rnn_type in ['cnn', 'tds']:
                eouts['ys']['xs'] = xs
                eouts['ys']['xlens'] = xlens
                return eouts

        if self.fast_impl:
            self.rnn.flatten_parameters()
            # NOTE: this is necessary for multi-GPUs setting

            # Path through RNN
            xs = pack_padded_sequence(xs, xlens.tolist(), batch_first=True)
            xs, _ = self.rnn(xs, hx=None)
            xs = pad_packed_sequence(xs, batch_first=True)[0]
            xs = self.dropout_top(xs)
        else:
            residual = None
            for l in range(len(self.rnn)):
                self.rnn[l].flatten_parameters()
                # NOTE: this is necessary for multi-GPUs setting

                # Path through RNN
                xs = pack_padded_sequence(xs, xlens.tolist(), batch_first=True)
                xs, _ = self.rnn[l](xs, hx=None)
                xs = pad_packed_sequence(xs, batch_first=True)[0]
                xs = self.dropout[l](xs)

                # Pick up outputs in the sub task before the projection layer
                if l == self.n_layers_sub1 - 1:
                    if self.task_specific_layer:
                        self.rnn_sub1_tsl.flatten_parameters()
                        xs_sub1 = pack_padded_sequence(xs, xlens.tolist(), batch_first=True)
                        xs_sub1, _ = self.rnn_sub1_tsl(xs_sub1, hx=None)
                        xs_sub1 = pad_packed_sequence(xs_sub1, batch_first=True)[0]
                        xs_sub1 = self.dropout_sub1_tsl(xs_sub1)
                    else:
                        xs_sub1 = xs.clone()[perm_ids_unsort]
                    xlens_sub1 = xlens[perm_ids_unsort].tolist()

                    if task == 'ys_sub1':
                        eouts[task]['xs'] = xs_sub1
                        eouts[task]['xlens'] = xlens_sub1
                        return eouts

                if l == self.n_layers_sub2 - 1:
                    if self.task_specific_layer:
                        self.rnn_sub2_tsl.flatten_parameters()
                        xs_sub2 = pack_padded_sequence(xs, xlens.tolist(), batch_first=True)
                        xs_sub2, _ = self.rnn_sub2_tsl(xs_sub2, hx=None)
                        xs_sub2 = pad_packed_sequence(xs_sub2, batch_first=True)[0]
                        xs_sub2 = self.dropout_sub2_tsl(xs_sub2)
                    else:
                        xs_sub2 = xs.clone()[perm_ids_unsort]
                    xlens_sub2 = xlens[perm_ids_unsort].tolist()

                    if task == 'ys_sub2':
                        eouts[task]['xs'] = xs_sub2
                        eouts[task]['xlens'] = xlens_sub2
                        return eouts

                # Projection layer
                if self.n_projs > 0:
                    xs = torch.tanh(self.proj[l](xs))

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
                            # xs = torch.tanh(self.concat[l](xs))

                            # Projection + batch normalization, ReLU
                            xs = self.concat_proj[l](xs)
                            xs = xs.contiguous().transpose(2, 1).unsqueeze(3)  # `[B, n_unis (*2), T, 1]`
                            # NOTE: consider feature dimension as input channel
                            xs = self.nin_bn[l](xs)
                            xs = F.relu(xs)  # `[B, n_unis (*2), T, 1]`
                            xs = xs.transpose(2, 1).squeeze(3)  # `[B, T, n_unis (*2)]`

                        elif self.subsample_type == 'max_pool':
                            xs = xs.transpose(1, 0).contiguous()
                            xs = [torch.max(xs[t - self.subsample[l] + 1:t + 1], dim=0)[0].unsqueeze(0)
                                  for t in range(xs.size(0)) if (t + 1) % self.subsample[l] == 0]
                            # NOTE: Exclude the last frame if the length of xs is odd
                            xs = torch.cat(xs, dim=0).transpose(1, 0)

                        # Update xlens
                        xlens //= self.subsample[l]

                    # NiN (1*1 conv + batch normalization + ReLU)
                    if self.nin:
                        xs = xs.contiguous().transpose(2, 1).unsqueeze(3)  # `[B, n_unis (*2), T, 1]`
                        # NOTE: consider feature dimension as input channel
                        xs = self.nin_conv[l](xs)
                        xs = self.nin_bn[l](xs)
                        xs = F.relu(xs)  # `[B, n_unis (*2), T, 1]`
                        xs = xs.transpose(2, 1).squeeze(3)  # `[B, T, n_unis (*2)]`

                    # Residual connection
                    if self.residual and residual is not None:
                        xs += residual
                    residual = xs

        # Unsort
        xs = xs[perm_ids_unsort]
        xlens = xlens[perm_ids_unsort].tolist()

        if task in ['all', 'ys']:
            eouts['ys']['xs'] = xs
            eouts['ys']['xlens'] = xlens
        if self.n_layers_sub1 >= 1 and task == 'all':
            eouts['ys_sub1']['xs'] = xs_sub1
            eouts['ys_sub1']['xlens'] = xlens_sub1
        if self.n_layers_sub2 >= 1 and task == 'all':
            eouts['ys_sub2']['xs'] = xs_sub2
            eouts['ys_sub2']['xlens'] = xlens_sub2

        return eouts


def to2d(xs, size):
    return xs.contiguous().view((int(np.prod(size[: -1])), int(size[-1])))


def to3d(xs, size):
    return xs.view(size)
