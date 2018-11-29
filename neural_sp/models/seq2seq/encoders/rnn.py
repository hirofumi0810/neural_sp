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
from neural_sp.models.seq2seq.encoders.cnn import VGG2LBlock
from neural_sp.models.seq2seq.encoders.cnn import get_vgg2l_odim


class RNNEncoder(nn.Module):
    """RNN encoder.

    Args:
        input_dim (int): the dimension of input features  (freq * channel)
        rnn_type (str): blstm or lstm or bgru or gru
        nunits (int): the number of units in each layer
        nprojs (int): the number of units in each projection layer
        nlayers (int): the number of layers
        dropout_in (float): the probability to drop nodes in input-hidden connection
        dropout_hidden (float): the probability to drop nodes in hidden-hidden connection
        subsample (list): subsample in the corresponding RNN layers
            ex.) [False, True, True, False] means that subsample is conducted
                in the 2nd and 3rd layers.
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
                 dropout_hidden,
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
        if subsample_type not in ['drop', 'concat']:
            raise TypeError('subsample_type must be "drop" or "concat".')
        if nlayers_sub < 0 or (nlayers_sub > 1 and nlayers < nlayers_sub):
            raise ValueError('Set nlayers_sub between 1 to nlayers.')

        self.rnn_type = rnn_type
        self.bidirectional = True if rnn_type in ['blstm', 'bgru'] else False
        self.nunits = nunits
        self.ndirs = 2 if self.bidirectional else 1
        self.nprojs = nprojs if nprojs is not None else 0
        self.nlayers = nlayers
        self.layer_norm = layer_norm

        # Setting for hierarchical encoder
        self.nlayers_sub = nlayers_sub

        # Setting for subsampling
        if len(subsample) == 0:
            self.subsample = [False] * nlayers
        else:
            self.subsample = subsample
        self.subsample_type = subsample_type

        # Setting for residual connection
        self.residual = residual
        subsample_last_layer = 0
        for l_reverse, is_subsample in enumerate(subsample[::-1]):
            if is_subsample:
                subsample_last_layer = nlayers - l_reverse
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
            assert nstacks == 1 and nsplices == 1
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

            # espnet
            # self.conv = VGG2LBlock(in_channel=conv_in_channel)
            # input_dim = get_vgg2l_odim(input_dim, in_channel=conv_in_channel)
        else:
            input_dim *= nsplices * nstacks
            self.conv = None

        self._fast_impl = False
        # Fast implementation without processes between each layer
        if sum(self.subsample) == 0 and self.nprojs == 0 and not residual and nlayers_sub == 0 and (not conv_batch_norm) and nin == 0:
            self._fast_impl = True
            if 'lstm' in rnn_type:
                rnn = nn.LSTM
            elif 'gru' in rnn_type:
                rnn = nn.GRU
            else:
                raise ValueError('rnn_type must be "(b)lstm" or "(b)gru".')

            self.rnn = rnn(input_dim, nunits, nlayers,
                           bias=True,
                           batch_first=True,
                           dropout=dropout_hidden,
                           bidirectional=self.bidirectional)
            # NOTE: pytorch introduces a dropout layer on the outputs of each layer EXCEPT the last layer

            # Dropout for the outputs of the top layer
            self.dropout_top = nn.Dropout(p=dropout_hidden)

        else:
            self.rnn = torch.nn.ModuleList()
            self.proj = torch.nn.ModuleList()
            for i_l in range(nlayers):
                if i_l == 0:
                    enc_in_dim = input_dim
                elif nin > 0:
                    enc_in_dim = nin
                elif self.nprojs > 0:
                    enc_in_dim = nprojs
                    if subsample_type == 'concat' and i_l > 0 and self.subsample[i_l - 1]:
                        enc_in_dim *= 2
                else:
                    enc_in_dim = nunits * self.ndirs
                    if subsample_type == 'concat' and i_l > 0 and self.subsample[i_l - 1]:
                        enc_in_dim *= 2

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
                enc_out_dim = nunits * self.ndirs

                # Dropout for hidden-hidden or hidden-output connection
                setattr(self, 'dropout_l' + str(i_l), nn.Dropout(p=dropout_hidden))

                if i_l != self.nlayers - 1 and self.nprojs > 0:
                    self.proj += [LinearND(nunits * self.ndirs, nprojs)]
                    enc_out_dim = nprojs

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
            xs (FloatTensor): `[B, T, input_dim]`
            x_lens (list): `[B]`
        Returns:
            xs (FloatTensor):
                `[B, T // sum(subsample), nunits (* ndirs)]`
            x_lens (list): `[B]`
            OPTION:
                xs_sub (FloatTensor):
                    `[B, T // sum(subsample), nunits (* ndirs)]`
                x_lens_sub (list): `[B]`

        """
        # Dropout for inputs-hidden connection
        xs = self.dropout_in(xs)

        # Path through CNN layers before RNN layers
        if self.conv is not None:
            xs, x_lens = self.conv(xs, x_lens)

        if self._fast_impl:
            getattr(self, self.rnn_type).flatten_parameters()
            # NOTE: this is necessary for multi-GPUs setting

            # Path through RNN
            xs = pack_padded_sequence(xs, x_lens, batch_first=True)
            xs, _ = getattr(self, self.rnn_type)(xs, hx=None)
            xs, unpacked_seq_len = pad_packed_sequence(xs, batch_first=True, padding_value=0)
            # assert x_lens == unpacked_seq_len

            # Dropout for the outputs of the top layer
            xs = self.dropout_top(xs)
        else:
            res_outputs = []
            for i_l in range(self.nlayers):
                self.rnn[i_l].flatten_parameters()
                # NOTE: this is necessary for multi-GPUs setting

                # Path through RNN
                xs = pack_padded_sequence(xs, x_lens, batch_first=True)
                xs, _ = self.rnn[i_l](xs, hx=None)
                xs, unpacked_seq_len = pad_packed_sequence(xs, batch_first=True, padding_value=0)
                # assert x_lens == unpacked_seq_len

                # Dropout for hidden-hidden or hidden-output connection
                xs = getattr(self, 'dropout_l' + str(i_l))(xs)

                # Pick up outputs in the sub task before the projection layer
                if self.nlayers_sub >= 1 and i_l == self.nlayers_sub - 1:
                    xs_sub = xs.clone()
                    x_lens_sub = copy.deepcopy(x_lens)

                # NOTE: Exclude the last layer
                if i_l != self.nlayers - 1:
                    # Subsampling
                    if self.subsample[i_l]:
                        if self.subsample_type == 'drop':
                            xs = xs[:, 1::2, :]
                            # NOTE: Pick up features at EVEN time step

                        # Concatenate the successive frames
                        elif self.subsample_type == 'concat':
                            xs = [torch.cat([xs[:, t - 1:t, :], xs[:, t:t + 1, :]], dim=2)
                                  for t in range(xs.size(1)) if (t + 1) % 2 == 0]
                            xs = torch.cat(xs, dim=1)
                            # NOTE: Exclude the last frame if the length of xs is odd

                        # Update x_lens
                        x_lens = [x.size(0) for x in xs]

                    # Projection layer (affine transformation)
                    if self.nprojs > 0:
                        xs = F.tanh(self.proj[i_l](xs))

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

        # For the sub task
        if self.nlayers_sub >= 1:
            return xs, x_lens, xs_sub, x_lens_sub
        else:
            return xs, x_lens


def to2d(xs, size):
    return xs.contiguous().view((int(np.prod(size[: -1])), int(size[-1])))


def to3d(xs, size):
    return xs.view(size)
