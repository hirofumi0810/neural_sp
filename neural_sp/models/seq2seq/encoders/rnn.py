#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""(Hierarchical) RNN encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from neural_sp.models.modules.linear import Linear
from neural_sp.models.seq2seq.encoders.conv import ConvEncoder
from neural_sp.models.seq2seq.encoders.encoder_base import EncoderBase
from neural_sp.models.seq2seq.encoders.gated_conv import GatedConvEncoder
from neural_sp.models.seq2seq.encoders.tds import TDSEncoder


class RNNEncoder(EncoderBase):
    """RNN encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        rnn_type (str): type of encoder (including pure CNN layers)
        n_units (int): number of units in each layer
        n_projs (int): number of units in each projection layer
        n_layers (int): number of layers
        dropout_in (float): dropout probability for input-hidden connection
        dropout (float): dropout probability for hidden-hidden connection
        subsample (list): subsample in the corresponding RNN layers
            ex.) [False, True, True, False] means that subsample is conducted in the 2nd and 3rd layers.
        subsample_type (str): drop/concat/max_pool
        last_proj_dim (int): dimension of the last projection layer
        n_stacks (int): number of frames to stack
        n_splices (int): number of frames to splice
        conv_in_channel (int): number of channels of input features
        conv_channels (int): number of channles in the CNN blocks
        conv_kernel_sizes (list): size of kernels in the CNN blocks
        conv_strides (list): number of strides in the CNN blocks
        conv_poolings (list): size of poolings in the CNN blocks
        conv_batch_norm (bool): apply batch normalization only in the CNN blocks
        conv_bottleneck_dim (int): dimension of the bottleneck layer between CNN and RNN layers
        n_layers_sub1 (int): number of layers in the 1st auxiliary task
        n_layers_sub2 (int): number of layers in the 2nd auxiliary task
        nin (bool): insert 1*1 conv + batch normalization + ReLU
        task_specific_layer (bool):
        param_init (float):

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
                 last_proj_dim=0,
                 conv_in_channel=1,
                 conv_channels=0,
                 conv_kernel_sizes=[],
                 conv_strides=[],
                 conv_poolings=[],
                 conv_batch_norm=False,
                 conv_bottleneck_dim=0,
                 n_layers_sub1=0,
                 n_layers_sub2=0,
                 nin=False,
                 task_specific_layer=False,
                 param_init=0.1):

        super(RNNEncoder, self).__init__()
        logger = logging.getLogger("training")

        if len(subsample) > 0 and len(subsample) != n_layers:
            raise ValueError('subsample must be the same size as n_layers.')
        if n_layers_sub1 < 0 or (n_layers_sub1 > 1 and n_layers < n_layers_sub1):
            raise ValueError('Set n_layers_sub1 between 1 to n_layers.')
        if n_layers_sub2 < 0 or (n_layers_sub2 > 1 and n_layers_sub1 < n_layers_sub2):
            raise ValueError('Set n_layers_sub2 between 1 to n_layers_sub1.')

        self.rnn_type = rnn_type
        self.bidirectional = True if rnn_type in ['blstm', 'bgru', 'conv_blstm', 'conv_bgru'] else False
        self.n_units = n_units
        self.n_dirs = 2 if self.bidirectional else 1
        self.n_layers = n_layers

        # Setting for hierarchical encoder
        self.n_layers_sub1 = n_layers_sub1
        self.n_layers_sub2 = n_layers_sub2
        self.task_specific_layer = task_specific_layer

        # Setting for bridge layers
        self.bridge = None
        self.bridge_sub1 = None
        self.bridge_sub2 = None

        # Dropout for input-hidden connection
        self.dropout_in = nn.Dropout(p=dropout_in)

        # Setting for CNNs before RNNs
        if conv_channels and rnn_type not in ['blstm', 'lstm', 'bgru', 'gru']:
            channels = [int(c) for c in conv_channels.split('_')] if len(conv_channels) > 0 else []
            kernel_sizes = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                            for c in conv_kernel_sizes.split('_')] if len(conv_kernel_sizes) > 0 else []
            if rnn_type in ['tds', 'gated_conv']:
                strides = []
                poolings = []
            else:
                strides = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                           for c in conv_strides.split('_')] if len(conv_strides) > 0 else []
                poolings = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                            for c in conv_poolings.split('_')] if len(conv_poolings) > 0 else []
            if 'conv_' in rnn_type:
                subsample = [1] * self.n_layers
                logger.warning('Subsampling is automatically ignored because CNN layers are used before RNN layers.')
        else:
            channels = []
            kernel_sizes = []
            strides = []
            poolings = []

        if len(channels) > 0:
            if rnn_type == 'tds':
                self.conv = TDSEncoder(input_dim=input_dim * n_stacks,
                                       in_channel=conv_in_channel,
                                       channels=channels,
                                       kernel_sizes=kernel_sizes,
                                       dropout=dropout,
                                       bottleneck_dim=last_proj_dim)
            elif rnn_type == 'gated_conv':
                self.conv = GatedConvEncoder(input_dim=input_dim * n_stacks,
                                             in_channel=conv_in_channel,
                                             channels=channels,
                                             kernel_sizes=kernel_sizes,
                                             dropout=dropout,
                                             bottleneck_dim=last_proj_dim,
                                             param_init=param_init)
            else:
                assert n_stacks == 1 and n_splices == 1
                self.conv = ConvEncoder(input_dim,
                                        in_channel=conv_in_channel,
                                        channels=channels,
                                        kernel_sizes=kernel_sizes,
                                        strides=strides,
                                        poolings=poolings,
                                        dropout=0,
                                        batch_norm=conv_batch_norm,
                                        bottleneck_dim=conv_bottleneck_dim,
                                        param_init=param_init)
            self._output_dim = self.conv.output_dim
        else:
            self._output_dim = input_dim * n_splices * n_stacks
            self.conv = None

        self.padding = Padding()

        if rnn_type not in ['conv', 'tds', 'gated_conv']:
            self.rnn = nn.ModuleList()
            self.dropout = nn.ModuleList()
            self.proj = None
            if n_projs > 0:
                self.proj = nn.ModuleList()

            # subsample
            self.subsample = None
            if subsample_type == 'max_pool' and np.prod(subsample) > 1:
                self.subsample = nn.ModuleList([MaxpoolSubsampler(subsample[l])
                                                for l in range(n_layers)])
            elif subsample_type == 'concat' and np.prod(subsample) > 1:
                self.subsample = nn.ModuleList([ConcatSubsampler(subsample[l], n_units, self.n_dirs)
                                                for l in range(n_layers)])
            elif subsample_type == 'drop' and np.prod(subsample) > 1:
                self.subsample = nn.ModuleList([DropSubsampler(subsample[l])
                                                for l in range(n_layers)])

            # NiN
            self.nin = None
            if nin:
                self.nin = nn.ModuleList()

            for l in range(n_layers):
                if 'lstm' in rnn_type:
                    rnn_i = nn.LSTM
                elif 'gru' in rnn_type:
                    rnn_i = nn.GRU
                else:
                    raise ValueError('rnn_type must be "(conv_)(b)lstm" or "(conv_)(b)gru".')

                self.rnn += [rnn_i(self._output_dim, n_units, 1,
                                   bias=True, batch_first=True, dropout=0,
                                   bidirectional=self.bidirectional)]
                self.dropout += [nn.Dropout(p=dropout)]
                self._output_dim = n_units * self.n_dirs

                # Projection layer
                if self.proj is not None:
                    if l != n_layers - 1:
                        self.proj += [Linear(n_units * self.n_dirs, n_projs)]
                        self._output_dim = n_projs

                # Task specific layer
                if l == n_layers_sub1 - 1 and task_specific_layer:
                    self.rnn_sub1 = rnn_i(self._output_dim, n_units, 1,
                                          bias=True, batch_first=True, dropout=0,
                                          bidirectional=self.bidirectional)
                    self.dropout_sub1 = nn.Dropout(p=dropout)
                    if last_proj_dim != self.output_dim:
                        self.bridge_sub1 = Linear(n_units, last_proj_dim)
                if l == n_layers_sub2 - 1 and task_specific_layer:
                    self.rnn_sub2 = rnn_i(self._output_dim, n_units, 1,
                                          bias=True, batch_first=True, dropout=0,
                                          bidirectional=self.bidirectional)
                    self.dropout_sub2 = nn.Dropout(p=dropout)
                    if last_proj_dim != self.output_dim:
                        self.bridge_sub2 = Linear(n_units, last_proj_dim)

                # Network in network
                if self.nin is not None:
                    if l != n_layers - 1:
                        self.nin += [NiN(self._output_dim)]
                    # if n_layers_sub1 > 0 or n_layers_sub2 > 0:
                    #     assert task_specific_layer

            if last_proj_dim != self.output_dim:
                self.bridge = Linear(self._output_dim, last_proj_dim)
                self._output_dim = last_proj_dim

        # Initialize parameters
        self.reset_parameters(param_init)

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger = logging.getLogger('training')
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if 'conv' in n or 'tds' in n or 'gated_conv' in n:
                continue  # for CNN layers before RNN layers
            if p.dim() == 1:
                nn.init.constant_(p, val=0)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0))
            elif p.dim() in [2, 4]:
                nn.init.uniform_(p, a=-param_init, b=param_init)
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', param_init))
            else:
                raise ValueError

    def forward(self, xs, xlens, task):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim]`
            xlens (list): A list of length `[B]`
            task (str): all or ys or ys_sub1 or ys_sub2
        Returns:
            eouts (dict):
                xs (FloatTensor): `[B, T // prod(subsample), n_units (*2)]`
                xlens (IntTensor): `[B]`
                xs_sub1 (FloatTensor): `[B, T // prod(subsample), n_units (*2)]`
                xlens_sub1 (IntTensor): `[B]`
                xs_sub2 (FloatTensor): `[B, T // prod(subsample), n_units (*2)]`
                xlens_sub2 (IntTensor): `[B]`

        """
        eouts = {'ys': {'xs': None, 'xlens': None},
                 'ys_sub1': {'xs': None, 'xlens': None},
                 'ys_sub2': {'xs': None, 'xlens': None}}

        # Sort by lenghts in the descending order for pack_padded_sequence
        xlens, perm_ids = torch.IntTensor(xlens).sort(0, descending=True)
        xs = xs[perm_ids]
        _, perm_ids_unsort = perm_ids.sort()

        # Dropout for inputs-hidden connection
        xs = self.dropout_in(xs)

        # Path through CNN blocks before RNN layers
        if self.conv is not None:
            xs, xlens = self.conv(xs, xlens)
            if self.rnn_type in ['conv', 'tds', 'gated_conv']:
                eouts['ys']['xs'] = xs
                eouts['ys']['xlens'] = xlens
                return eouts

        for l in range(self.n_layers):
            self.rnn[l].flatten_parameters()  # for multi-GPUs
            xs = self.padding(xs, xlens, self.rnn[l])
            xs = self.dropout[l](xs)

            # Pick up outputs in the sub task before the projection layer
            if l == self.n_layers_sub1 - 1:
                if self.task_specific_layer:
                    self.rnn_sub1.flatten_parameters()  # for multi-GPUs
                    xs_sub1 = self.padding(xs, xlens, self.rnn_sub1)
                    xs_sub1 = self.dropout_sub1(xs_sub1)
                else:
                    xs_sub1 = xs.clone()[perm_ids_unsort]
                if self.bridge_sub1 is not None:
                    xs_sub1 = self.bridge_sub1(xs_sub1)
                xlens_sub1 = xlens[perm_ids_unsort]

                if task == 'ys_sub1':
                    eouts[task]['xs'] = xs_sub1
                    eouts[task]['xlens'] = xlens_sub1
                    return eouts

            if l == self.n_layers_sub2 - 1:
                if self.task_specific_layer:
                    self.rnn_sub2.flatten_parameters()  # for multi-GPUs
                    xs_sub2 = self.padding(xs, xlens, self.rnn_sub2)
                    xs_sub2 = self.dropout_sub2(xs_sub2)
                else:
                    xs_sub2 = xs.clone()[perm_ids_unsort]
                if self.bridge_sub2 is not None:
                    xs_sub2 = self.bridge_sub2(xs_sub2)
                xlens_sub2 = xlens[perm_ids_unsort]

                if task == 'ys_sub2':
                    eouts[task]['xs'] = xs_sub2
                    eouts[task]['xlens'] = xlens_sub2
                    return eouts

            # NOTE: Exclude the last layer
            if l != self.n_layers - 1:
                # Projection layer
                if self.proj is not None:
                    xs = torch.tanh(self.proj[l](xs))

                # Subsampling
                if self.subsample is not None:
                    xs, xlens = self.subsample[l](xs, xlens)

                # NiN
                if self.nin is not None:
                    xs = self.nin[l](xs)

        # Bridge layer
        if self.bridge is not None:
            xs = self.bridge(xs)

        # Unsort
        xs = xs[perm_ids_unsort]
        xlens = xlens[perm_ids_unsort]

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


class Padding(nn.Module):
    """Padding variable length of sequences."""

    def __init__(self):
        super(Padding, self).__init__()

    def forward(self, xs, xlens, rnn):
        xs = pack_padded_sequence(xs, xlens.tolist(), batch_first=True)
        xs, _ = rnn(xs, hx=None)
        xs = pad_packed_sequence(xs, batch_first=True)[0]
        return xs


class MaxpoolSubsampler(nn.Module):
    """Subsample by max-pooling input frames."""

    def __init__(self, factor):
        super(MaxpoolSubsampler, self).__init__()

        self.factor = factor
        if factor > 1:
            self.max_pool = nn.MaxPool2d((1, 1), stride=(factor, 1), ceil_mode=True)

    def forward(self, xs, xlens):
        if self.factor == 1:
            return xs, xlens

        xs = xs.transpose(1, 0).contiguous()
        xs = [torch.max(xs[t - self.factor + 1:t + 1], dim=0)[0].unsqueeze(0)
              for t in range(xs.size(0)) if (t + 1) % self.factor == 0]
        # NOTE: Exclude the last frames if the length is not divisible
        xs = torch.cat(xs, dim=0).transpose(1, 0)

        xlens /= self.factor
        return xs, xlens


class DropSubsampler(nn.Module):
    """Subsample by droping input frames."""

    def __init__(self, factor):
        super(DropSubsampler, self).__init__()

        self.factor = factor

    def forward(self, xs, xlens):
        if self.factor == 1:
            return xs, xlens

        xs = xs[:, ::self.factor, :]
        xlens = [max(1, (i + self.factor - 1) // self.factor) for i in xlens]
        xlens = torch.IntTensor(xlens)
        return xs, xlens


class ConcatSubsampler(nn.Module):
    """Subsample by concatenating input frames."""

    def __init__(self, factor, n_units, n_dirs):
        super(ConcatSubsampler, self).__init__()

        self.factor = factor
        if factor > 1:
            self.proj = Linear(n_units * n_dirs * factor, n_units * n_dirs)
            self.batch_norm = nn.BatchNorm1d(n_units * n_dirs)

    def forward(self, xs, xlens):
        if self.factor == 1:
            return xs, xlens

        # Concatenate the successive frames
        xs = xs.transpose(1, 0).contiguous()
        xs = [torch.cat([xs[t - r:t - r + 1] for r in range(self.factor - 1, -1, -1)], dim=-1)
              for t in range(xs.size(0)) if (t + 1) % self.factor == 0]
        xs = torch.cat(xs, dim=0).transpose(1, 0)
        # NOTE: Exclude the last frames if the length is not divisible

        # Projection
        xs = self.proj(xs)
        # xs = torch.tanh(proj(xs))

        # Batch normalization, ReLU
        bs, time = xs.size()[:2]
        xs = self.batch_norm(xs.view(bs * time, -1)).view(bs, time, -1)
        # xs = self.batch_norm[l](xs)
        xs = torch.relu(xs)

        xlens /= self.factor
        return xs, xlens


class NiN(nn.Module):
    """Network in network."""

    def __init__(self, dim):
        super(NiN, self).__init__()

        self.conv = nn.Conv2d(in_channels=dim,
                              out_channels=dim,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.batch_norm = nn.BatchNorm2d(dim)

    def forward(self, xs):
        # 1*1 conv + batch normalization + ReLU
        xs = xs.contiguous().transpose(2, 1).unsqueeze(3)  # `[B, n_unis (*2), T, 1]`
        # NOTE: consider feature dimension as input channel
        xs = torch.relu(self.batch_norm(self.conv(xs)))  # `[B, n_unis (*2), T, 1]`
        xs = xs.transpose(2, 1).squeeze(3)  # `[B, T, n_unis (*2)]`
        return xs
