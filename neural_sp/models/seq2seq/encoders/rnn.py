# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""(Hierarchical) RNN encoder."""

from distutils.util import strtobool
import logging
import math
import numpy as np
import random
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from neural_sp.models.modules.initialization import init_with_uniform
from neural_sp.models.seq2seq.encoders.conv import ConvEncoder
from neural_sp.models.seq2seq.encoders.encoder_base import EncoderBase
from neural_sp.models.seq2seq.encoders.subsampling import (
    AddSubsampler,
    ConcatSubsampler,
    Conv1dSubsampler,
    DropSubsampler,
    MaxpoolSubsampler
)
from neural_sp.models.seq2seq.encoders.utils import chunkwise

random.seed(1)

logger = logging.getLogger(__name__)


class RNNEncoder(EncoderBase):
    """RNN encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        enc_type (str): type of encoder (including pure CNN layers)
        n_units (int): number of units in each layer
        n_projs (int): number of units in each projection layer
        last_proj_dim (int): dimension of the last projection layer
        n_layers (int): number of layers
        n_layers_sub1 (int): number of layers in the 1st auxiliary task
        n_layers_sub2 (int): number of layers in the 2nd auxiliary task
        dropout_in (float): dropout probability for input-hidden connection
        dropout (float): dropout probability for hidden-hidden connection
        subsample (list): subsample in the corresponding RNN layers
            ex.) [1, 2, 2, 1] means that subsample is conducted in the 2nd and 3rd layers.
        subsample_type (str): drop/concat/max_pool/1dconv
        n_stacks (int): number of frames to stack
        n_splices (int): number of frames to splice
        conv_in_channel (int): number of channels of input features
        conv_channels (int): number of channels in CNN blocks
        conv_kernel_sizes (list): size of kernels in CNN blocks
        conv_strides (list): number of strides in CNN blocks
        conv_poolings (list): size of poolings in CNN blocks
        conv_batch_norm (bool): apply batch normalization only in CNN blocks
        conv_layer_norm (bool): apply layer normalization only in CNN blocks
        conv_bottleneck_dim (int): dimension of bottleneck layer between CNN and RNN layers
        bidir_sum_fwd_bwd (bool): sum up forward and backward outputs for dimension reduction
        task_specific_layer (bool): add a task specific layer for each sub task
        param_init (float): model initialization parameter
        chunk_size_current (str): current chunk size for latency-controlled bidirectional encoder
        chunk_size_right (str): right chunk size for latency-controlled bidirectional encoder
        cnn_lookahead (bool): enable lookahead for frontend CNN layers for LC-BLSTM
        rsp_prob (float): probability of Random State Passing (RSP)

    """

    def __init__(self, input_dim, enc_type, n_units, n_projs, last_proj_dim,
                 n_layers, n_layers_sub1, n_layers_sub2,
                 dropout_in, dropout,
                 subsample, subsample_type, n_stacks, n_splices,
                 conv_in_channel, conv_channels, conv_kernel_sizes, conv_strides, conv_poolings,
                 conv_batch_norm, conv_layer_norm, conv_bottleneck_dim,
                 bidir_sum_fwd_bwd, task_specific_layer, param_init,
                 chunk_size_current, chunk_size_right, cnn_lookahead,
                 rsp_prob):

        super(RNNEncoder, self).__init__()

        # parse subsample
        subsamples = [1] * n_layers
        for lth, s in enumerate(list(map(int, subsample.split('_')[:n_layers]))):
            subsamples[lth] = s

        if len(subsamples) > 0 and len(subsamples) != n_layers:
            raise ValueError('subsample must be the same size as n_layers. n_layers: %d, subsample: %s' %
                             (n_layers, subsamples))
        if n_layers_sub1 < 0 or (n_layers_sub1 > 1 and n_layers < n_layers_sub1):
            raise Warning('Set n_layers_sub1 between 1 to n_layers. n_layers: %d, n_layers_sub1: %d' %
                          (n_layers, n_layers_sub1))
        if n_layers_sub2 < 0 or (n_layers_sub2 > 1 and n_layers_sub1 < n_layers_sub2):
            raise Warning('Set n_layers_sub2 between 1 to n_layers_sub1. n_layers_sub1: %d, n_layers_sub2: %d' %
                          (n_layers_sub1, n_layers_sub2))

        self.enc_type = enc_type
        self.bidirectional = True if ('blstm' in enc_type or 'bgru' in enc_type) else False
        self.n_units = n_units
        self.n_dirs = 2 if self.bidirectional else 1
        self.n_layers = n_layers
        self.bidir_sum = bidir_sum_fwd_bwd

        # for compatiblity
        chunk_size_current = str(chunk_size_current)
        chunk_size_right = str(chunk_size_right)

        # for latency-controlled
        self.chunk_size_current = int(chunk_size_current.split('_')[0]) // n_stacks
        self.chunk_size_right = int(chunk_size_right.split('_')[0]) // n_stacks
        self.lc_bidir = self.chunk_size_current > 0 or self.chunk_size_right > 0 and self.bidirectional
        if self.lc_bidir:
            assert enc_type not in ['lstm', 'gru', 'conv_lstm', 'conv_gru']
            assert n_layers_sub2 == 0

        # for streaming
        self.rsp_prob = rsp_prob

        # for hierarchical encoder
        self.n_layers_sub1 = n_layers_sub1
        self.n_layers_sub2 = n_layers_sub2
        self.task_specific_layer = task_specific_layer

        # for bridge layers
        self.bridge = None
        self.bridge_sub1 = None
        self.bridge_sub2 = None

        # Dropout for input-hidden connection
        self.dropout_in = nn.Dropout(p=dropout_in)

        if 'conv' in enc_type:
            assert n_stacks == 1 and n_splices == 1
            self.conv = ConvEncoder(input_dim,
                                    in_channel=conv_in_channel,
                                    channels=conv_channels,
                                    kernel_sizes=conv_kernel_sizes,
                                    strides=conv_strides,
                                    poolings=conv_poolings,
                                    dropout=0.,
                                    batch_norm=conv_batch_norm,
                                    layer_norm=conv_layer_norm,
                                    residual=False,
                                    bottleneck_dim=conv_bottleneck_dim,
                                    param_init=param_init)
            self._odim = self.conv.output_dim
        else:
            self.conv = None
            self._odim = input_dim * n_splices * n_stacks
        self.cnn_lookahead = cnn_lookahead
        if not cnn_lookahead:
            assert self.chunk_size_current > 0
            assert self.lc_bidir

        if enc_type != 'conv':
            self.rnn = nn.ModuleList()
            if self.lc_bidir:
                self.rnn_bwd = nn.ModuleList()
            self.dropout = nn.Dropout(p=dropout)
            self.proj = nn.ModuleList() if n_projs > 0 else None
            self.subsample = nn.ModuleList() if np.prod(subsamples) > 1 else None
            self.padding = Padding(bidir_sum_fwd_bwd=bidir_sum_fwd_bwd if not self.lc_bidir else False)

            for lth in range(n_layers):
                if 'lstm' in enc_type:
                    rnn_i = nn.LSTM
                elif 'gru' in enc_type:
                    rnn_i = nn.GRU
                else:
                    raise ValueError('enc_type must be "(conv_)(b)lstm" or "(conv_)(b)gru".')

                if self.lc_bidir:
                    self.rnn += [rnn_i(self._odim, n_units, 1, batch_first=True)]
                    self.rnn_bwd += [rnn_i(self._odim, n_units, 1, batch_first=True)]
                else:
                    self.rnn += [rnn_i(self._odim, n_units, 1, batch_first=True,
                                       bidirectional=self.bidirectional)]
                self._odim = n_units if bidir_sum_fwd_bwd else n_units * self.n_dirs

                # Task specific layer
                if lth == n_layers_sub1 - 1 and task_specific_layer:
                    self.layer_sub1 = nn.Linear(self._odim, n_units)
                    self._odim_sub1 = n_units
                    if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                        self.bridge_sub1 = nn.Linear(n_units, last_proj_dim)
                        self._odim_sub1 = last_proj_dim
                if lth == n_layers_sub2 - 1 and task_specific_layer:
                    assert not self.lc_bidir
                    self.layer_sub2 = nn.Linear(self._odim, n_units)
                    self._odim_sub2 = n_units
                    if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                        self.bridge_sub2 = nn.Linear(n_units, last_proj_dim)
                        self._odim_sub2 = last_proj_dim

                # Projection layer
                if self.proj is not None:
                    if lth != n_layers - 1:
                        self.proj += [nn.Linear(self._odim, n_projs)]
                        self._odim = n_projs

                # subsample
                if np.prod(subsamples) > 1:
                    if subsample_type == 'max_pool':
                        self.subsample += [MaxpoolSubsampler(subsamples[lth])]
                    elif subsample_type == 'concat':
                        self.subsample += [ConcatSubsampler(subsamples[lth], self._odim)]
                    elif subsample_type == 'drop':
                        self.subsample += [DropSubsampler(subsamples[lth])]
                    elif subsample_type == '1dconv':
                        self.subsample += [Conv1dSubsampler(subsamples[lth], self._odim)]
                    elif subsample_type == 'add':
                        self.subsample += [AddSubsampler(subsamples[lth])]

            if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                self.bridge = nn.Linear(self._odim, last_proj_dim)
                self._odim = last_proj_dim

        # calculate subsampling factor
        self._factor = 1
        if self.conv is not None:
            self._factor *= self.conv.subsampling_factor
        self._factor_sub1 = self._factor
        if n_layers_sub1 > 0 and np.prod(subsamples[:n_layers_sub1 - 1]) > 1:
            self._factor_sub1 *= np.prod(subsamples[:n_layers_sub1 - 1])
        self._factor_sub2 = self._factor
        if n_layers_sub2 > 0 and np.prod(subsamples[:n_layers_sub2 - 1]) > 1:
            self._factor_sub1 *= np.prod(subsamples[:n_layers_sub2 - 1])
        if np.prod(subsamples) > 1:
            self._factor *= np.prod(subsamples)
        # NOTE: subsampling factor for frame stacking should not be included here
        if self.chunk_size_current > 0:
            assert self.chunk_size_current % self._factor == 0
        if self.chunk_size_right > 0:
            assert self.chunk_size_right % self._factor == 0

        self.reset_parameters(param_init)

        # for streaming inference
        self.reset_cache()

    @staticmethod
    def add_args(parser, args):
        group = parser.add_argument_group("RNN encoder")
        parser = ConvEncoder.add_args(parser, args)
        group.add_argument('--enc_n_units', type=int, default=512,
                           help='number of units in each encoder RNN layer')
        group.add_argument('--enc_n_projs', type=int, default=0,
                           help='number of units in the projection layer after each encoder RNN layer')
        group.add_argument('--bidirectional_sum_fwd_bwd', type=strtobool, default=False,
                           help='sum forward and backward RNN outputs for dimension reduction')
        # streaming
        group.add_argument('--lc_chunk_size_left', type=str, default="-1",
                           help='current chunk size for latency-controlled RNN encoder')
        group.add_argument('--lc_chunk_size_right', type=str, default="0",
                           help='right chunk size for latency-controlled RNN encoder')
        group.add_argument('--cnn_lookahead', type=strtobool, default=True,
                           help='disable lookahead frames in CNN layers')
        group.add_argument('--rsp_prob_enc', type=float, default=0.0,
                           help='probability for Random State Passing (RSP)')
        return parser

    @staticmethod
    def define_name(dir_name, args):
        if 'conv' in args.enc_type:
            dir_name = ConvEncoder.define_name(dir_name, args)

        dir_name += str(args.enc_n_units) + 'H'
        if args.enc_n_projs > 0:
            dir_name += str(args.enc_n_projs) + 'P'
        dir_name += str(args.enc_n_layers) + 'L'
        if args.bidirectional_sum_fwd_bwd:
            dir_name += '_sumfwdbwd'
        if int(args.lc_chunk_size_left.split('_')[0]) > 0 or int(args.lc_chunk_size_right.split('_')[0]) > 0:
            dir_name += '_chunkL' + args.lc_chunk_size_left + 'R' + args.lc_chunk_size_right
            if not args.cnn_lookahead:
                dir_name += '_blockwise'
        if args.rsp_prob_enc > 0:
            dir_name += '_RSP' + str(args.rsp_prob_enc)
        return dir_name

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if 'conv' in n:
                continue  # for CNN layers before RNN layers
            init_with_uniform(n, p, param_init)

    def reset_cache(self):
        self.hx_fwd = [None] * self.n_layers
        logger.debug('Reset cache.')

    def forward(self, xs, xlens, task, streaming=False,
                lookback=False, lookahead=False):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, input_dim]`
            xlens (list): A list of length `[B]`
            task (str): all or ys or ys_sub1 or ys_sub2
            streaming (bool): streaming encoding
            lookback (bool): truncate leftmost frames for lookback in CNN context
            lookahead (bool): truncate rightmost frames for lookahead in CNN context
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

        # Sort by lengths in the descending order for pack_padded_sequence
        perm_ids_unsort = None
        if not self.lc_bidir:
            xlens, perm_ids = torch.IntTensor(xlens).sort(0, descending=True)
            xs = xs[perm_ids]
            _, perm_ids_unsort = perm_ids.sort()

        # Dropout for inputs-hidden connection
        xs = self.dropout_in(xs)

        bs, xmax, idim = xs.size()
        N_c, N_r = self.chunk_size_current, self.chunk_size_right

        if self.lc_bidir and not self.cnn_lookahead:
            xs = chunkwise(xs, 0, N_c, 0)  # `[B * n_chunks, N_c, idim]`
            # Extract the center region
            xs = xs.contiguous().view(bs, -1, xs.size(2))
            xs = xs[:, :xlens.max()]  # `[B, emax, d_model]`

        # Path through CNN blocks before RNN layers
        if self.conv is not None:
            xs, xlens = self.conv(xs, xlens, lookback=lookback, lookahead=lookahead)
            if self.enc_type == 'conv':
                eouts['ys']['xs'] = xs
                eouts['ys']['xlens'] = xlens
                return eouts
            if self.lc_bidir:
                N_c = N_c // self.conv.subsampling_factor
                N_r = N_r // self.conv.subsampling_factor

        carry_over = self.rsp_prob > 0 and self.training and random.random() < self.rsp_prob
        carry_over = carry_over and (bs == (self.hx_fwd[0][0].size(0) if self.hx_fwd[0] is not None else 0))
        if not streaming and not carry_over:
            self.reset_cache()
            # NOTE: do not reset here for streaming inference

        if self.lc_bidir:
            # Flip the layer and time loop
            if self.chunk_size_current <= 0:
                xs, xlens, xs_sub1, xlens_sub1 = self._forward_full_context(
                    xs, xlens)
            else:
                xs, xlens, xs_sub1, xlens_sub1 = self._forward_latency_controlled(
                    xs, xlens, N_c, N_r, streaming)
            if task == 'ys_sub1':
                eouts[task]['xs'], eouts[task]['xlens'] = xs_sub1, xlens_sub1
                return eouts
        else:
            for lth in range(self.n_layers):
                self.rnn[lth].flatten_parameters()  # for multi-GPUs
                xs, state = self.padding(xs, xlens, self.rnn[lth],
                                         prev_state=self.hx_fwd[lth],
                                         streaming=streaming)
                self.hx_fwd[lth] = state
                xs = self.dropout(xs)

                # Pick up outputs in the sub task before the projection layer
                if lth == self.n_layers_sub1 - 1:
                    xs_sub1, xlens_sub1 = self.sub_module(xs, xlens, perm_ids_unsort, 'sub1')
                    if task == 'ys_sub1':
                        eouts[task]['xs'], eouts[task]['xlens'] = xs_sub1, xlens_sub1
                        return eouts
                if lth == self.n_layers_sub2 - 1:
                    xs_sub2, xlens_sub2 = self.sub_module(xs, xlens, perm_ids_unsort, 'sub2')
                    if task == 'ys_sub2':
                        eouts[task]['xs'], eouts[task]['xlens'] = xs_sub2, xlens_sub2
                        return eouts

                # Projection layer
                if self.proj is not None and lth != self.n_layers - 1:
                    xs = torch.relu(self.proj[lth](xs))
                # Subsampling layer
                if self.subsample is not None:
                    xs, xlens = self.subsample[lth](xs, xlens)

        # Bridge layer
        if self.bridge is not None:
            xs = self.bridge(xs)

        xs = xs[:, :xlens.max()]

        if task in ['all', 'ys']:
            if perm_ids_unsort is not None:
                xs = xs[perm_ids_unsort]
                xlens = xlens[perm_ids_unsort]
            eouts['ys']['xs'], eouts['ys']['xlens'] = xs, xlens
        if self.n_layers_sub1 >= 1 and task == 'all':
            eouts['ys_sub1']['xs'], eouts['ys_sub1']['xlens'] = xs_sub1, xlens_sub1
        if self.n_layers_sub2 >= 1 and task == 'all':
            eouts['ys_sub2']['xs'], eouts['ys_sub2']['xlens'] = xs_sub2, xlens_sub2
        return eouts

    def _forward_full_context(self, xs, xlens, task='all'):
        """Full context BPTT encoding.
           This is used for pre-training latency-controlled bidirectional encoder.

        Args:
            xs (FloatTensor): `[B, T, n_units]`
            xlens (IntTensor): `[B]`
            task (str):
        Returns:
            xs (FloatTensor): `[B, T, n_units]`
            xlens (IntTensor): `[B]`
            xs_sub1 (FloatTensor): `[B, T, n_units]`
            xlens_sub1 (IntTensor): `[B]`

        """
        xs_sub1, xlens_sub1 = None, None
        for lth in range(self.n_layers):
            self.rnn[lth].flatten_parameters()  # for multi-GPUs
            self.rnn_bwd[lth].flatten_parameters()  # for multi-GPUs
            xs_bwd = torch.flip(self.rnn_bwd[lth](torch.flip(xs, dims=[1]))[0], dims=[1])
            xs_fwd, self.hx_fwd[lth] = self.rnn[lth](xs, hx=self.hx_fwd[lth])
            if self.bidir_sum:
                xs = xs_fwd + xs_bwd
            else:
                xs = torch.cat([xs_fwd, xs_bwd], dim=-1)
            xs = self.dropout(xs)

            # Pick up outputs in the sub task before the projection layer
            if lth == self.n_layers_sub1 - 1:
                xs_sub1, xlens_sub1 = self.sub_module(xs, xlens, None, 'sub1')
                if task == 'ys_sub1':
                    return None, None, xs_sub1, xlens_sub1

            # Projection layer
            if self.proj is not None and lth != self.n_layers - 1:
                xs = torch.relu(self.proj[lth](xs))
            # Subsampling layer
            if self.subsample is not None:
                xs, xlens = self.subsample[lth](xs, xlens)

        return xs, xlens, xs_sub1, xlens_sub1

    def _forward_latency_controlled(self, xs, xlens, N_c, N_r, streaming,
                                    task='all'):
        """Streaming encoding for the latency-controlled bidirectional encoder.

        Args:
            xs (FloatTensor): `[B, T, n_units]`
            xlens (IntTensor): `[B]`
            N_c (int):
            N_r (int):
            streaming (bool):
            task (str):
        Returns:
            xs (FloatTensor): `[B, T, n_units]`
            xlens (IntTensor): `[B]`
            xs_sub1 (FloatTensor): `[B, T, n_units]`
            xlens (IntTensor): `[B]`

        """
        bs, xmax, _ = xs.size()
        n_chunks = math.ceil(xmax / N_c)

        if streaming:
            xlens = torch.IntTensor(bs).fill_(min(xmax, N_c))
        xlens_sub1 = xlens.clone() if self.n_layers_sub1 > 0 else None

        xs_chunks = []
        xs_chunks_sub1 = []
        for chunk_idx, t in enumerate(range(0, N_c * n_chunks, N_c)):
            xs_chunk = xs[:, t:t + (N_c + N_r)]
            _N_c = N_c

            for lth in range(self.n_layers):
                self.rnn[lth].flatten_parameters()  # for multi-GPUs
                self.rnn_bwd[lth].flatten_parameters()  # for multi-GPUs
                # bwd
                xs_chunk_bwd = torch.flip(self.rnn_bwd[lth](
                    torch.flip(xs_chunk, dims=[1]))[0], dims=[1])  # `[B, _N_c+_N_r, n_units]`
                # fwd
                if xs_chunk.size(1) <= _N_c:
                    # last chunk
                    xs_chunk_fwd, self.hx_fwd[lth] = self.rnn[lth](xs_chunk,
                                                                   hx=self.hx_fwd[lth])
                else:
                    xs_chunk_fwd1, self.hx_fwd[lth] = self.rnn[lth](xs_chunk[:, :_N_c],
                                                                    hx=self.hx_fwd[lth])
                    xs_chunk_fwd2, _ = self.rnn[lth](xs_chunk[:, _N_c:],
                                                     hx=self.hx_fwd[lth])
                    xs_chunk_fwd = torch.cat([xs_chunk_fwd1, xs_chunk_fwd2], dim=1)  # `[B, _N_c+_N_r, n_units]`
                    # NOTE: xs_chunk_fwd2 is used for xs_chunk_bwd in the next layer
                if self.bidir_sum:
                    xs_chunk = xs_chunk_fwd + xs_chunk_bwd
                else:
                    xs_chunk = torch.cat([xs_chunk_fwd, xs_chunk_bwd], dim=-1)
                xs_chunk = self.dropout(xs_chunk)

                # Pick up outputs in the sub task before the projection layer
                if lth == self.n_layers_sub1 - 1:
                    xs_chunks_sub1.append(xs_chunk.clone()[:, :_N_c])
                    if chunk_idx == 0:
                        xlens_sub1 = xlens.clone()

                # Projection layer
                if self.proj is not None and lth != self.n_layers - 1:
                    xs_chunk = torch.relu(self.proj[lth](xs_chunk))
                # Subsampling layer
                if self.subsample is not None:
                    xs_chunk, xlens_tmp = self.subsample[lth](xs_chunk, xlens)
                    if chunk_idx == 0:
                        xlens = xlens_tmp
                    _N_c = _N_c // self.subsample[lth].factor

            xs_chunks.append(xs_chunk[:, :_N_c])

            if streaming:
                break

        xs = torch.cat(xs_chunks, dim=1)
        if self.n_layers_sub1 > 0:
            xs_sub1 = torch.cat(xs_chunks_sub1, dim=1)
            xs_sub1, xlens_sub1 = self.sub_module(xs_sub1, xlens_sub1, None, 'sub1')
        else:
            xs_sub1 = None

        return xs, xlens, xs_sub1, xlens_sub1

    def sub_module(self, xs, xlens, perm_ids_unsort, module='sub1'):
        if self.task_specific_layer:
            xs_sub = self.dropout(torch.relu(getattr(self, 'layer_' + module)(xs)))
        else:
            xs_sub = xs.clone()
        if getattr(self, 'bridge_' + module) is not None:
            xs_sub = getattr(self, 'bridge_' + module)(xs_sub)
        if perm_ids_unsort is not None:
            xs_sub = xs_sub[perm_ids_unsort]
            xlens_sub = xlens[perm_ids_unsort]
        else:
            xlens_sub = xlens.clone()
        return xs_sub, xlens_sub


class Padding(nn.Module):
    """Padding variable length of sequences."""

    def __init__(self, bidir_sum_fwd_bwd):
        super(Padding, self).__init__()
        self.bidir_sum = bidir_sum_fwd_bwd

    def forward(self, xs, xlens, rnn, prev_state=None, streaming=False):
        if not streaming and xlens is not None:
            xs = pack_padded_sequence(xs, xlens.tolist(), batch_first=True)
            xs, state = rnn(xs, hx=prev_state)
            xs = pad_packed_sequence(xs, batch_first=True)[0]
        else:
            xs, state = rnn(xs, hx=prev_state)

        if self.bidir_sum:
            assert rnn.bidirectional
            half = xs.size(-1) // 2
            xs = xs[:, :, :half] + xs[:, :, half:]
        return xs, state


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
