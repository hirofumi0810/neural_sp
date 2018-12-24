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
        input_dim (int): the dimension of input features (freq * channel)
        rnn_type (str): blstm or lstm or bgru or gru
        nunits (int): the number of units in each layer
        nprojs (int): the number of units in each projection layer
        nlayers (int): the number of layers
        dropout_in (float): the probability to drop nodes in input-hidden connection
        dropout (float): the probability to drop nodes in hidden-hidden connection
        subsample (list): subsample in the corresponding RNN layers
            ex.) [False, True, True, False] means that subsample is conducted in the 2nd and 3rd layers.
        subsample_type (str): drop or concat or max_pool
        nstacks (int): the number of frames to stack
        nskips (int): the number of frames to skip
        nsplices (int): frames to splice. Default is 1 frame.
        conv_in_channel (int): the number of channels of input features
        conv_channels (int): the number of channles in the CNN layers
        conv_kernel_sizes (list): the size of kernels in the CNN layers
        conv_strides (list): the number of strides in the CNN layers
        conv_poolings (list): the size of poolings in the CNN layers
        conv_batch_norm (bool): apply batch normalization only in the CNN layers
        residual (bool): add residual connections between the consecutive layers
        nlayers_sub (int): the number of layers in the sub task
        nin (int): if larger than 0, insert 1*1 conv (filter size: nin)
            and ReLU activation between each LSTM layer
        layer_norm (bool): layer normalization

    """

    def __init__(self,
                 input_dim,
                 rnn_type,
                 nunits,
                 nprojs,
                 nlayers,
                 dropout_in,
                 dropout,
                 subsample,
                 subsample_type,
                 nstacks,
                 nsplices,
                 conv_in_channel,
                 conv_channels,
                 conv_kernel_sizes,
                 conv_strides,
                 conv_poolings,
                 conv_batch_norm,
                 residual,
                 nlayers_sub=0,
                 nin=0,
                 layer_norm=False):

        super(RNNEncoder, self).__init__()

        if len(subsample) > 0 and len(subsample) != nlayers:
            raise ValueError('subsample must be the same size as nlayers.')
        if nlayers_sub < 0 or (nlayers_sub > 1 and nlayers < nlayers_sub):
            raise ValueError('Set nlayers_sub between 1 to nlayers.')
        if rnn_type == 'cnn':
            assert nstacks == 1 and nsplices == 1

        self.rnn_type = rnn_type
        self.bidirectional = True if rnn_type in ['blstm', 'bgru'] else False
        self.nunits = nunits
        self.ndirs = 2 if self.bidirectional else 1
        self.nprojs = nprojs
        self.nlayers = nlayers
        self.layer_norm = layer_norm

        # Setting for hierarchical encoder
        self.nlayers_sub = nlayers_sub

        # Setting for subsampling
        if len(subsample) == 0:
            self.subsample = [1] * nlayers
        else:
            self.subsample = subsample
        self.subsample_type = subsample_type

        # Setting for residual connection
        self.residual = residual
        subsample_last = 0
        for l_reverse, is_subsample in enumerate(subsample[::-1]):
            if is_subsample:
                subsample_last = nlayers - l_reverse
                break
        self.residual_start_layer = subsample_last + 1
        # NOTE: residual connection starts from the last subsampling layer

        # Setting for the NiN
        self.conv_batch_norm = conv_batch_norm
        self.nin = nin

        # Dropout for input-hidden connection
        self.dropout_in = nn.Dropout(p=dropout_in)

        # Setting for CNNs before RNNs
        if len(conv_channels) > 0 and len(conv_channels) == len(conv_kernel_sizes) and len(conv_kernel_sizes) == len(conv_strides):
            assert nstacks == 1 and nsplices == 1
            self.conv = CNNEncoder(input_dim,
                                   in_channel=conv_in_channel,
                                   channels=conv_channels,
                                   kernel_sizes=conv_kernel_sizes,
                                   strides=conv_strides,
                                   poolings=conv_poolings,
                                   dropout=dropout,
                                   activation='relu',
                                   batch_norm=conv_batch_norm)
            input_dim = self.conv.output_dim
        else:
            input_dim *= nsplices * nstacks
            self.conv = None

        if rnn_type != 'cnn':
            self.fast_impl = False
            # Fast implementation without processes between each layer
            if np.prod(self.subsample) == 1 and self.nprojs == 0 and not residual and nlayers_sub == 0 and (not conv_batch_norm) and nin == 0:
                self.fast_impl = True
                if 'lstm' in rnn_type:
                    rnn = nn.LSTM
                elif 'gru' in rnn_type:
                    rnn = nn.GRU
                else:
                    raise ValueError('rnn_type must be "(b)lstm" or "(b)gru".')

                self.rnn = rnn(input_dim, nunits, nlayers,
                               bias=True,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=self.bidirectional)
                # NOTE: pytorch introduces a dropout layer on the outputs of each layer EXCEPT the last layer
                self.dropout_top = nn.Dropout(p=dropout)
            else:
                self.rnn = torch.nn.ModuleList()
                self.dropout = torch.nn.ModuleList()
                if self.nprojs > 0:
                    self.proj = torch.nn.ModuleList()
                if subsample_type == 'max_pool' and np.prod(self.subsample) > 1:
                    self.max_pool = torch.nn.ModuleList()
                    for l in range(nlayers):
                        if self.subsample[l] > 1:
                            self.max_pool += [nn.MaxPool2d((1, 1),
                                                           stride=(self.subsample[l], 1),
                                                           ceil_mode=True)]
                        else:
                            self.max_pool += [None]
                if subsample_type == 'concat' and np.prod(self.subsample) > 1:
                    self.concat = torch.nn.ModuleList()
                    for l in range(nlayers):
                        if self.subsample[l] > 1:
                            self.concat += [LinearND(nunits * self.ndirs * self.subsample[l], nunits * self.ndirs)]
                        else:
                            self.concat += [None]

                for l in range(nlayers):
                    if l == 0:
                        enc_in_dim = input_dim
                    elif nin > 0:
                        enc_in_dim = nin
                    elif self.nprojs > 0:
                        enc_in_dim = self.nprojs
                    else:
                        enc_in_dim = nunits * self.ndirs

                    if 'lstm' in rnn_type:
                        rnn_i = nn.LSTM
                    elif 'gru' in rnn_type:
                        rnn_i = nn.GRU
                    else:
                        raise ValueError('rnn_type must be "lstm" or "gru".')

                    self.rnn += [rnn_i(enc_in_dim, nunits, 1,
                                       bias=True,
                                       batch_first=True,
                                       dropout=0,
                                       bidirectional=self.bidirectional)]
                    self.dropout += [nn.Dropout(p=dropout)]
                    enc_out_dim = nunits * self.ndirs

                    if l != self.nlayers - 1 and self.nprojs > 0:
                        self.proj += [LinearND(nunits * self.ndirs, self.nprojs)]
                        enc_out_dim = self.nprojs

                    # Network in network (1*1 conv)
                    if nin > 0:
                        setattr(self, 'nin_l' + str(l),
                                nn.Conv1d(in_channels=enc_out_dim,
                                          out_channels=nin,
                                          kernel_size=1,
                                          stride=1,
                                          padding=1,
                                          bias=not conv_batch_norm))

                        # Batch normalization
                        if conv_batch_norm:
                            if nin:
                                setattr(self, 'bn_0_l' + str(l), nn.BatchNorm1d(enc_out_dim))
                                setattr(self, 'bn_l' + str(l), nn.BatchNorm1d(nin))
                            else:
                                setattr(self, 'bn_l' + str(l), nn.BatchNorm1d(enc_out_dim))
                        # NOTE* BN in RNN models is applied only after NiN

    def forward(self, xs, x_lens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim]`
            x_lens (list): `[B]`
        Returns:
            xs (FloatTensor): `[B, T // prod(subsample), nunits (* ndirs)]`
            x_lens (list): `[B]`
            OPTION:
                xs_sub (FloatTensor): `[B, T // prod(subsample), nunits (* ndirs)]`
                x_lens_sub (list): `[B]`

        """
        # Dropout for inputs-hidden connection
        xs = self.dropout_in(xs)

        # Path through CNN layers before RNN layers
        if self.conv is not None:
            xs, x_lens = self.conv(xs, x_lens)
            if self.rnn_type == 'cnn':
                return xs, x_lens

        if self.fast_impl:
            self.rnn.flatten_parameters()
            # NOTE: this is necessary for multi-GPUs setting

            # Path through RNN
            xs = pack_padded_sequence(xs, x_lens, batch_first=True)
            xs, _ = self.rnn(xs, hx=None)
            xs = pad_packed_sequence(xs, batch_first=True)[0]
            xs = self.dropout_top(xs)
        else:
            res_outputs = []
            for l in range(self.nlayers):
                self.rnn[l].flatten_parameters()
                # NOTE: this is necessary for multi-GPUs setting

                # Path through RNN
                xs = pack_padded_sequence(xs, x_lens, batch_first=True)
                xs, _ = self.rnn[l](xs, hx=None)
                xs = pad_packed_sequence(xs, batch_first=True)[0]
                xs = self.dropout[l](xs)

                # Pick up outputs in the sub task before the projection layer
                if self.nlayers_sub >= 1 and l == self.nlayers_sub - 1:
                    xs_sub = xs.clone()
                    x_lens_sub = copy.deepcopy(x_lens)

                # NOTE: Exclude the last layer
                if l != self.nlayers - 1:
                    # Subsampling
                    if self.subsample[l] > 1:
                        if self.subsample_type == 'drop':
                            xs = xs[:, 1::self.subsample[l], :]
                            # NOTE: Pick up features at EVEN time step
                        elif self.subsample_type == 'concat':
                            # Concatenate the successive frames
                            xs = xs.transpose(0, 1).contiguous()
                            xs = [torch.cat([xs[t - r:t - r + 1] for r in range(self.subsample[l] - 1, -1, -1)], dim=-1)
                                  for t in range(xs.size(0)) if (t + 1) % self.subsample[l] == 0]
                            # NOTE: Exclude the last frame if the length of xs is odd
                            xs = torch.cat(xs, dim=0).transpose(0, 1)
                            xs = torch.tanh(self.concat[l](xs))
                        elif self.subsample_type == 'max_pool':
                            xs = xs.transpose(0, 1).contiguous()
                            xs = [torch.max(xs[t - self.subsample[l] + 1:t + 1], dim=0)[0].unsqueeze(0)
                                  for t in range(xs.size(0)) if (t + 1) % self.subsample[l] == 0]
                            xs = torch.cat(xs, dim=0).transpose(0, 1)

                        # Update x_lens
                        x_lens = [x.size(0) for x in xs]

                    # Projection layer
                    if self.nprojs > 0:
                        xs = torch.tanh(self.proj[l](xs))

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
                            for xs_lower in res_outputs:
                                xs = xs + xs_lower
                            if self.residual:
                                res_outputs = [xs]
                    # NOTE: Exclude residual connection from the raw inputs

        if self.nlayers_sub >= 1:
            # For the sub task
            return xs, x_lens, xs_sub, x_lens_sub
        else:
            return xs, x_lens


def to2d(xs, size):
    return xs.contiguous().view((int(np.prod(size[: -1])), int(size[-1])))


def to3d(xs, size):
    return xs.view(size)
