#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""(Hierarchical) RNN encoders (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from src.models.pytorch_v3.encoders.cnn import CNNEncoder
from src.models.pytorch_v3.linear import LinearND


class RNNEncoder(torch.nn.Module):
    """RNN encoder.

    Args:
        input_size (int): the dimension of input features
        rnn_type (str): lstm or gru or rnn
        bidirectional (bool): if True, use the bidirectional encoder
        n_units (int): the number of units in each layer
        n_projs (int): the number of nodes in the projection layer
        n_layers (int): the number of layers
        dropout_in (float): the probability to drop nodes in input-hidden connection
        dropout_hidden (float): the probability to drop nodes in hidden-hidden connection
        subsample_list (list): subsample in the corresponding layers (True)
            ex.) [False, True, True, False] means that downsample is conducted
                in the 2nd and 3rd layers.
        subsample_type (str): drop or concat
        batch_first (bool): if True, batch-major computation will be performed
        merge_bidirectional (bool): if True, sum bidirectional outputs
        pack_sequence (bool):
        n_stack (int): the number of frames to stack
        n_splice (int): frames to splice. Default is 1 frame.
        conv_in_channel (int): the number of channels of input features
        conv_channels (list): the number of channles in CNN layers
        conv_kernel_sizes (list): the size of kernels in CNN layers
        conv_strides (list): strides in CNN layers
        conv_poolings (list): the size of poolings in CNN layers
        conv_batch_norm (bool): if True, apply batch normalization
        residual (bool): if True, apply residual connection between each layer
        n_layers_sub (int): the number of layers in the sub task
        nin (int): if larger than 0, insert 1*1 conv (filter size: nin)
            and ReLU activation between each LSTM layer

    """

    def __init__(self,
                 input_size,
                 rnn_type,
                 bidirectional,
                 n_units,
                 n_projs,
                 n_layers,
                 dropout_in,
                 dropout_hidden,
                 subsample_list,
                 subsample_type,
                 batch_first,
                 merge_bidirectional,
                 pack_sequence,
                 n_stack,
                 n_splice,
                 conv_in_channel,
                 conv_channels,
                 conv_kernel_sizes,
                 conv_strides,
                 conv_poolings,
                 conv_batch_norm,
                 residual,
                 n_layers_sub=0,
                 nin=0):

        super(RNNEncoder, self).__init__()

        if len(subsample_list) > 0 and len(subsample_list) != n_layers:
            raise ValueError('subsample_list must be the same size as n_layers.')
        if subsample_type not in ['drop', 'concat']:
            raise TypeError('subsample_type must be "drop" or "concat".')
        if n_layers_sub < 0 or (n_layers_sub > 1 and n_layers < n_layers_sub):
            raise ValueError('Set n_layers_sub between 1 to n_layers.')

        self.rnn_type = rnn_type
        self.n_units = n_units
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1
        self.n_projs = n_projs if n_projs is not None else 0
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.merge_bidirectional = merge_bidirectional
        self.pack_sequence = pack_sequence

        # Setting for hierarchical encoder
        self.n_layers_sub = n_layers_sub

        # Setting for subsampling
        if len(subsample_list) == 0:
            self.subsample_list = [False] * n_layers
        else:
            self.subsample_list = subsample_list
        self.subsample_type = subsample_type
        # This implementation is bases on
        # https://arxiv.org/abs/1508.01211
        #     Chan, William, et al. "Listen, attend and spell."
        #         arXiv preprint arXiv:1508.01211 (2015).

        # Setting for residual connection
        self.residual = residual
        subsample_last_layer = 0
        for l_reverse, is_subsample in enumerate(subsample_list[::-1]):
            if is_subsample:
                subsample_last_layer = n_layers - l_reverse
                break
        self.residual_start_layer = subsample_last_layer + 1
        # NOTE: residual connection starts from the last subsampling layer

        # Setting for the NiN
        self.conv_batch_norm = conv_batch_norm
        self.nin = nin

        # Dropout for input-hidden connection
        self.dropout_in = torch.nn.Dropout(p=dropout_in)

        # Setting for CNNs before RNNs
        if len(conv_channels) > 0 and len(conv_channels) == len(conv_kernel_sizes) and len(conv_kernel_sizes) == len(conv_strides):
            assert n_stack == 1 and n_splice == 1
            self.conv = CNNEncoder(input_size,
                                   in_channel=conv_in_channel,
                                   channels=conv_channels,
                                   kernel_sizes=conv_kernel_sizes,
                                   strides=conv_strides,
                                   poolings=conv_poolings,
                                   dropout_in=0,
                                   dropout_hidden=dropout_hidden,
                                   activation='relu',
                                   batch_norm=conv_batch_norm)
            input_size = self.conv.output_size
        else:
            input_size = input_size * n_splice * n_stack
            self.conv = None

        # Fast implementation without processes between each layer
        if sum(self.subsample_list) == 0 and self.n_projs == 0 and not residual and n_layers_sub == 0 and (not conv_batch_norm) and nin == 0:
            self.fast_impl = True

            if rnn_type == 'lstm':
                rnn = torch.nn.LSTM(input_size,
                                    hidden_size=n_units,
                                    num_layers=n_layers,
                                    bias=True,
                                    batch_first=batch_first,
                                    dropout=dropout_hidden,
                                    bidirectional=bidirectional)
            elif rnn_type == 'gru':
                rnn = torch.nn.GRU(input_size,
                                   hidden_size=n_units,
                                   num_layers=n_layers,
                                   bias=True,
                                   batch_first=batch_first,
                                   dropout=dropout_hidden,
                                   bidirectional=bidirectional)
            else:
                raise ValueError('rnn_type must be "lstm" or "gru".')

            setattr(self, rnn_type, rnn)
            # NOTE: pytorch introduces a dropout layer on the outputs of
            # each RNN layer EXCEPT the last layer

            # Dropout for hidden-output connection
            self.dropout_last = torch.nn.Dropout(p=dropout_hidden)

        else:
            self.fast_impl = False

            for i_l in range(n_layers):
                if i_l == 0:
                    enc_in_size = input_size
                elif nin > 0:
                    enc_in_size = nin
                elif self.n_projs > 0:
                    enc_in_size = n_projs
                    if subsample_type == 'concat' and i_l > 0 and self.subsample_list[i_l - 1]:
                        enc_in_size *= 2
                else:
                    enc_in_size = n_units * self.n_directions
                    if subsample_type == 'concat' and i_l > 0 and self.subsample_list[i_l - 1]:
                        enc_in_size *= 2

                if rnn_type == 'lstm':
                    rnn_i = torch.nn.LSTM(enc_in_size,
                                          hidden_size=n_units,
                                          num_layers=1,
                                          bias=True,
                                          batch_first=batch_first,
                                          dropout=0,
                                          bidirectional=bidirectional)

                elif rnn_type == 'gru':
                    rnn_i = torch.nn.GRU(enc_in_size,
                                         hidden_size=n_units,
                                         num_layers=1,
                                         bias=True,
                                         batch_first=batch_first,
                                         dropout=0,
                                         bidirectional=bidirectional)
                else:
                    raise ValueError('rnn_type must be "lstm" or "gru".')

                setattr(self, rnn_type + '_l' + str(i_l), rnn_i)
                enc_out_size = n_units * self.n_directions
                # TODO(hirofumi): check this

                # Dropout for hidden-hidden or hidden-output connection
                setattr(self, 'dropout_l' + str(i_l), torch.nn.Dropout(p=dropout_hidden))

                if i_l != self.n_layers - 1 and self.n_projs > 0:
                    proj_i = LinearND(n_units * self.n_directions, n_projs)
                    setattr(self, 'proj_l' + str(i_l), proj_i)
                    enc_out_size = n_projs

                # Network in network (1*1 conv)
                if nin > 0:
                    setattr(self, 'nin_l' + str(i_l),
                            torch.nn.Conv1d(in_channels=enc_out_size,
                                            out_channels=nin,
                                            kernel_size=1,
                                            stride=1,
                                            padding=1,
                                            bias=not conv_batch_norm))

                    # Batch normalization
                    if conv_batch_norm:
                        if nin:
                            setattr(self, 'bn_0_l' + str(i_l), torch.nn.BatchNorm1d(enc_out_size))
                            setattr(self, 'bn_l' + str(i_l), torch.nn.BatchNorm1d(nin))
                        elif subsample_type == 'concat' and self.subsample_list[i_l]:
                            setattr(self, 'bn_l' + str(i_l), torch.nn.BatchNorm1d(enc_out_size * 2))
                        else:
                            setattr(self, 'bn_l' + str(i_l), torch.nn.BatchNorm1d(enc_out_size))
                    # NOTE* BN in RNN models is applied only after NiN

    def forward(self, xs, x_lens, volatile=False):
        """Forward computation.

        Args:
            xs (torch.autograd.Variable, float): A tensor of size
                `[B, T, input_size]`
            x_lens (list): A tensor of size `[B]`
            volatile (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            xs (torch.autograd.Variable, float):
                if batch_first is True, a tensor of size
                    `[B, T // sum(subsample_list), n_units (* n_directions)]`
                else
                    `[T // sum(subsample_list), B, n_units (* n_directions)]`
            x_lens (list): A tensor of size `[B]`
            OPTION:
                xs_sub (torch.autograd.Variable, float):
                    if batch_first is True, a tensor of size
                        `[B, T // sum(subsample_list), n_units (* n_directions)]`
                    else
                        `[T // sum(subsample_list), B, n_units (* n_directions)]`
                x_lens_sub (list): A tensor of size `[B]`

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
            if xs.is_cuda:
                torch.cuda.empty_cache()

            # Pack encoder inputs
            if self.pack_sequence:
                xs = pack_padded_sequence(xs, x_lens, batch_first=self.batch_first)

            # Path through RNN
            getattr(self, self.rnn_type).flatten_parameters()
            xs, _ = getattr(self, self.rnn_type)(xs, hx=None)
            getattr(self, self.rnn_type).flatten_parameters()

            # Unpack encoder outputs
            if self.pack_sequence:
                xs, unpacked_seq_len = pad_packed_sequence(
                    xs, batch_first=self.batch_first, padding_value=0)
                # assert x_lens == unpacked_seq_len

            # Dropout for hidden-output connection
            xs = self.dropout_last(xs)
        else:
            res_outputs = []
            for i_l in range(self.n_layers):
                if xs.is_cuda:
                    torch.cuda.empty_cache()

                # Pack i_l-th encoder xs
                if self.pack_sequence:
                    xs = pack_padded_sequence(xs, x_lens, batch_first=self.batch_first)

                # Path through RNN
                getattr(self, self.rnn_type + '_l' + str(i_l)).flatten_parameters()
                xs, _ = getattr(self, self.rnn_type + '_l' + str(i_l))(xs, hx=None)
                if i_l == self.n_layers - 1:
                    getattr(self, self.rnn_type + '_l' + str(i_l)).flatten_parameters()

                # Unpack i_l-th encoder outputs
                if self.pack_sequence:
                    xs, unpacked_seq_len = pad_packed_sequence(
                        xs, batch_first=self.batch_first, padding_value=0)
                    # assert x_lens == unpacked_seq_len

                # Dropout for hidden-hidden or hidden-output connection
                xs = getattr(self, 'dropout_l' + str(i_l))(xs)

                # Pick up outputs in the sub task before the projection layer
                if self.n_layers_sub >= 1 and i_l == self.n_layers_sub - 1:
                    xs_sub = xs.clone()
                    x_lens_sub = copy.deepcopy(x_lens)

                # NOTE: Exclude the last layer
                if i_l != self.n_layers - 1:
                    # Subsampling
                    if self.subsample_list[i_l]:
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
                                      for t in range(xs.size(1)) if (t + 1) % 2 == 0]
                                xs = torch.cat(xs, dim=1)
                            else:
                                xs = [torch.cat([xs[t - 1:t, :, :], xs[t:t + 1, :, :]], dim=2)
                                      for t in range(xs.size(0)) if (t + 1) % 2 == 0]
                                xs = torch.cat(xs, dim=0)
                            # NOTE: Exclude the last frame if the length of xs is odd

                        # Update x_lens
                        if self.batch_first:
                            x_lens = [x.size(0) for x in xs]
                        else:
                            x_lens = [xs[:, i].size(0) for i in range(xs.size(1))]

                    # Projection layer (affine transformation)
                    if self.n_projs > 0:
                        xs = F.tanh(getattr(self, 'proj_l' + str(i_l))(xs))

                    # NiN
                    if self.nin > 0:
                        raise NotImplementedError

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
                    if (not self.subsample_list[i_l]) and self.residual:
                        if i_l >= self.residual_start_layer - 1:
                            for xs_lower in res_outputs:
                                xs = xs + xs_lower
                            if self.residual:
                                res_outputs = [xs]
                    # NOTE: Exclude residual connection from the raw inputs

        # Sum bidirectional outputs
        if self.bidirectional and self.merge_bidirectional:
            xs = xs[:, :, :self.n_units] + xs[:, :, self.n_units:]

        if not self.batch_first:
            # Convert to the time-major
            xs = xs.transpose(0, 1).contiguous()

        # For the sub task
        if self.n_layers_sub >= 1:
            # Sum bidirectional outputs
            if self.bidirectional and self.merge_bidirectional:
                xs_sub = xs_sub[:, :, :self.n_units] + xs_sub[:, :, self.n_units:]
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
