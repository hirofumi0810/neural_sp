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
        bidirectional_sum_fwd_bwd (bool):

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
                 param_init=0.1,
                 bidirectional_sum_fwd_bwd=False):

        super(RNNEncoder, self).__init__()
        logger = logging.getLogger("training")

        if len(subsample) > 0 and len(subsample) != n_layers:
            raise ValueError('subsample must be the same size as n_layers.')
        if n_layers_sub1 < 0 or (n_layers_sub1 > 1 and n_layers < n_layers_sub1):
            raise ValueError('Set n_layers_sub1 between 1 to n_layers.')
        if n_layers_sub2 < 0 or (n_layers_sub2 > 1 and n_layers_sub1 < n_layers_sub2):
            raise ValueError('Set n_layers_sub2 between 1 to n_layers_sub1.')

        self.rnn_type = rnn_type
        self.bidirectional = True if ('blstm' in rnn_type or 'bgru' in rnn_type) else False
        self.latency_controlled = True if ('lcblstm' in rnn_type or 'lcbrgu' in rnn_type) else False
        self.chunk_size = 20  # for latency-controlled BRNN
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

        if 'conv' in rnn_type:
            subsample = [1] * self.n_layers
            logger.warning('Subsampling is automatically ignored because CNN layers are used before RNN layers.')

        if conv_channels:
            if rnn_type == 'tds':
                self.conv = TDSEncoder(input_dim=input_dim * n_stacks,
                                       in_channel=conv_in_channel,
                                       channels=conv_channels,
                                       kernel_sizes=conv_kernel_sizes,
                                       dropout=dropout,
                                       bottleneck_dim=last_proj_dim)
            elif rnn_type == 'gated_conv':
                self.conv = GatedConvEncoder(input_dim=input_dim * n_stacks,
                                             in_channel=conv_in_channel,
                                             channels=conv_channels,
                                             kernel_sizes=conv_kernel_sizes,
                                             dropout=dropout,
                                             bottleneck_dim=last_proj_dim,
                                             param_init=param_init)
            else:
                assert n_stacks == 1 and n_splices == 1
                self.conv = ConvEncoder(input_dim,
                                        in_channel=conv_in_channel,
                                        channels=conv_channels,
                                        kernel_sizes=conv_kernel_sizes,
                                        strides=conv_strides,
                                        poolings=conv_poolings,
                                        dropout=0.,
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
            if self.latency_controlled:
                self.rnn_bwd = nn.ModuleList()
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
                self.subsample = nn.ModuleList([ConcatSubsampler(subsample[l], n_units * self.n_dirs)
                                                for l in range(n_layers)])
            elif subsample_type == 'drop' and np.prod(subsample) > 1:
                self.subsample = nn.ModuleList([DropSubsampler(subsample[l])
                                                for l in range(n_layers)])
            elif subsample_type == '1dconv' and np.prod(subsample) > 1:
                self.subsample = nn.ModuleList([Conv1dSubsampler(subsample[l], n_units * self.n_dirs)
                                                for l in range(n_layers)])

            # NiN
            self.nin = nn.ModuleList() if nin else None

            for l in range(n_layers):
                if 'lstm' in rnn_type:
                    rnn_i = nn.LSTM
                elif 'gru' in rnn_type:
                    rnn_i = nn.GRU
                else:
                    raise ValueError('rnn_type must be "(conv_)(b/lcb)lstm" or "(conv_)(b/lcb)gru".')

                if self.latency_controlled:
                    self.rnn += [rnn_i(self._output_dim, n_units, 1, batch_first=True)]
                    self.rnn_bwd += [rnn_i(self._output_dim, n_units, 1, batch_first=True)]
                else:
                    self.rnn += [rnn_i(self._output_dim, n_units, 1, batch_first=True,
                                       bidirectional=self.bidirectional)]
                self.dropout += [nn.Dropout(p=dropout)]
                self._output_dim = n_units if bidirectional_sum_fwd_bwd else n_units * self.n_dirs
                self.bidirectional_sum_fwd_bwd = bidirectional_sum_fwd_bwd

                # Projection layer
                if self.proj is not None:
                    if l != n_layers - 1:
                        self.proj += [nn.Linear(n_units * self.n_dirs, n_projs)]
                        self._output_dim = n_projs

                # Task specific layer
                if l == n_layers_sub1 - 1 and task_specific_layer:
                    self.rnn_sub1 = rnn_i(self._output_dim, n_units, 1,
                                          batch_first=True,
                                          bidirectional=self.bidirectional)
                    self.dropout_sub1 = nn.Dropout(p=dropout)
                    if last_proj_dim != self.output_dim:
                        self.bridge_sub1 = nn.Linear(n_units, last_proj_dim)
                if l == n_layers_sub2 - 1 and task_specific_layer:
                    self.rnn_sub2 = rnn_i(self._output_dim, n_units, 1,
                                          batch_first=True,
                                          bidirectional=self.bidirectional)
                    self.dropout_sub2 = nn.Dropout(p=dropout)
                    if last_proj_dim != self.output_dim:
                        self.bridge_sub2 = nn.Linear(n_units, last_proj_dim)

                # Network in network
                if self.nin is not None:
                    if l != n_layers - 1:
                        self.nin += [NiN(self._output_dim)]
                    # if n_layers_sub1 > 0 or n_layers_sub2 > 0:
                    #     assert task_specific_layer

            if last_proj_dim != self.output_dim:
                self.bridge = nn.Linear(self._output_dim, last_proj_dim)
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
                nn.init.constant_(p, 0.)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
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
            if self.latency_controlled:
                self.rnn_bwd[l].flatten_parameters()  # for multi-GPUs
                xs_chunks = []
                fwd_state = None
                for t in range(0, xs.size(1), self.chunk_size):
                    # bwd
                    xs_chunk_bwd = torch.flip(xs[:, t:t + self.chunk_size * 2], [1])
                    xs_chunk_bwd, _ = self.rnn_bwd[l](xs_chunk_bwd, hx=None)
                    xs_chunk_bwd = torch.flip(xs_chunk_bwd, [1])[:, :self.chunk_size]
                    # fwd
                    xs_chunk_fwd = xs[:, t:t + self.chunk_size]
                    xs_chunk_fwd, fwd_state = self.rnn[l](xs_chunk_fwd, hx=fwd_state)
                    if self.bidirectional_sum_fwd_bwd:
                        xs_chunk = xs_chunk_fwd + xs_chunk_bwd
                    else:
                        xs_chunk = torch.cat([xs_chunk_fwd, xs_chunk_bwd], dim=-1)
                    xs_chunks.append(xs_chunk)
                xs = torch.cat(xs_chunks, dim=1)
            else:
                xs = self.padding(xs, xlens, self.rnn[l], self.bidirectional_sum_fwd_bwd)
            xs = self.dropout[l](xs)

            # Pick up outputs in the sub task before the projection layer
            if l == self.n_layers_sub1 - 1:
                xs_sub1, xlens_sub1 = self.sub_module(xs, xlens, perm_ids_unsort, 'sub1')
                if task == 'ys_sub1':
                    eouts[task]['xs'] = xs_sub1
                    eouts[task]['xlens'] = xlens_sub1
                    return eouts
            if l == self.n_layers_sub2 - 1:
                xs_sub2, xlens_sub2 = self.sub_module(xs, xlens, perm_ids_unsort, 'sub2')
                if task == 'ys_sub2':
                    eouts[task]['xs'] = xs_sub2
                    eouts[task]['xlens'] = xlens_sub2
                    return eouts

            # NOTE: Exclude the last layer
            if l != self.n_layers - 1:
                # Projection layer -> Subsampling -> NiN
                if self.proj is not None:
                    xs = torch.tanh(self.proj[l](xs))
                if self.subsample is not None:
                    xs, xlens = self.subsample[l](xs, xlens)
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

    def sub_module(self, xs, xlens, perm_ids_unsort, module='sub1'):
        if self.task_specific_layer:
            getattr(self, 'rnn_' + module).flatten_parameters()  # for multi-GPUs
            xs_sub = self.padding(xs, xlens, getattr(self, 'rnn_' + module))
            xs_sub = getattr(self, 'dropout_' + module)(xs_sub)
        else:
            xs_sub = xs.clone()[perm_ids_unsort]
        if getattr(self, 'bridge_' + module) is not None:
            xs_sub = getattr(self, 'bridge_' + module)(xs_sub)
        xlens_sub = xlens[perm_ids_unsort]
        return xs_sub, xlens_sub


class Padding(nn.Module):
    """Padding variable length of sequences."""

    def __init__(self):
        super(Padding, self).__init__()

    def forward(self, xs, xlens, rnn, bidirectional_sum_fwd_bwd=False):
        xs = pack_padded_sequence(xs, xlens.tolist(), batch_first=True)
        xs, _ = rnn(xs, hx=None)
        xs = pad_packed_sequence(xs, batch_first=True)[0]
        if bidirectional_sum_fwd_bwd and rnn.bidirectional:
            half = xs.size(-1) // 2
            xs = xs[:, :, :half] + xs[:, :, half:]
        return xs


class MaxpoolSubsampler(nn.Module):
    """Subsample by max-pooling input frames."""

    def __init__(self, factor):
        super(MaxpoolSubsampler, self).__init__()

        self.factor = factor
        if factor > 1:
            self.max_pool = nn.MaxPool1d(1, stride=factor, ceil_mode=True)

    def forward(self, xs, xlens):
        if self.factor == 1:
            return xs, xlens

        # subsample
        xs = self.max_pool(xs.transpose(2, 1)).transpose(2, 1).contiguous()

        xlens //= self.factor
        return xs, xlens


class Conv1dSubsampler(nn.Module):
    """Subsample by 1d convolution and max-pooling."""

    def __init__(self, factor, n_units, conv_kernel_size=5):
        super(Conv1dSubsampler, self).__init__()

        assert conv_kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        self.factor = factor
        if factor > 1:
            self.conv1d = nn.Conv1d(in_channels=n_units,
                                    out_channels=n_units,
                                    kernel_size=conv_kernel_size,
                                    stride=1,
                                    padding=(conv_kernel_size - 1) // 2)
            self.max_pool = nn.MaxPool1d(1, stride=factor, ceil_mode=True)

    def forward(self, xs, xlens):
        if self.factor == 1:
            return xs, xlens

        # subsample
        xs = torch.relu(self.conv1d(xs.transpose(2, 1)))
        xs = self.max_pool(xs).transpose(2, 1).contiguous()

        xlens //= self.factor
        return xs, xlens


class DropSubsampler(nn.Module):
    """Subsample by droping input frames."""

    def __init__(self, factor):
        super(DropSubsampler, self).__init__()

        self.factor = factor

    def forward(self, xs, xlens):
        if self.factor == 1:
            return xs, xlens

        # subsample
        xs = xs[:, ::self.factor, :]

        xlens = [max(1, (i + self.factor - 1) // self.factor) for i in xlens]
        xlens = torch.IntTensor(xlens)
        return xs, xlens


class ConcatSubsampler(nn.Module):
    """Subsample by concatenating input frames."""

    def __init__(self, factor, n_units):
        super(ConcatSubsampler, self).__init__()

        self.factor = factor
        if factor > 1:
            self.proj = nn.Linear(n_units * factor, n_units)

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
        xs = torch.relu(self.proj(xs))

        xlens //= self.factor
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
