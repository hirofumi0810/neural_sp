#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import random
import torch
import torch.nn as nn

from neural_sp.models.modules.conformer_convolution import ConformerConvBlock
from neural_sp.models.modules.positional_embedding import XLPositionalEmbedding
from neural_sp.models.modules.positionwise_feed_forward import PositionwiseFeedForward as FFN
from neural_sp.models.modules.relative_multihead_attention import RelativeMultiheadAttentionMechanism as RelMHA
from neural_sp.models.seq2seq.encoders.conv import ConvEncoder
from neural_sp.models.seq2seq.encoders.encoder_base import blockwise
from neural_sp.models.seq2seq.encoders.encoder_base import EncoderBase
from neural_sp.models.torch_utils import make_pad_mask
from neural_sp.models.torch_utils import tensor2np

random.seed(1)

logger = logging.getLogger(__name__)


class ConformerEncoder(EncoderBase):
    """Conformer encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        enc_type (str): type of encoder
        n_heads (int): number of heads for multi-head attention
        kernel_size (int): kernel size for depthwise convolution in convolution module
        n_layers (int): number of blocks
        n_layers_sub1 (int): number of layers in the 1st auxiliary task
        n_layers_sub2 (int): number of layers in the 2nd auxiliary task
        d_model (int): dimension of MultiheadAttentionMechanism
        d_ff (int): dimension of PositionwiseFeedForward
        ffn_bottleneck_dim (int): bottleneck dimension for the light-weight FFN layer
        last_proj_dim (int): dimension of the last projection layer
        pe_type (str): type of positional encoding
        layer_norm_eps (float): epsilon value for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        dropout_in (float): dropout probability for input-hidden connection
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        dropout_layer (float): LayerDrop probability for layers
        n_stacks (int): number of frames to stack
        n_splices (int): frames to splice. Default is 1 frame.
        conv_in_channel (int): number of channels of input features
        conv_channels (int): number of channles in the CNN blocks
        conv_kernel_sizes (list): size of kernels in the CNN blocks
        conv_strides (list): number of strides in the CNN blocks
        conv_poolings (list): size of poolings in the CNN blocks
        conv_batch_norm (bool): apply batch normalization only in the CNN blocks
        conv_layer_norm (bool): apply layer normalization only in the CNN blocks
        conv_bottleneck_dim (int): dimension of the bottleneck layer between CNN and self-attention layers
        conv_param_init (float): only for CNN layers before Conformer layers
        task_specific_layer (bool): add a task specific layer for each sub task
        param_init (str): parameter initialization method
        chunk_size_left (int): left chunk size for time-restricted Conformer encoder
        chunk_size_current (int): current chunk size for time-restricted Conformer encoder
        chunk_size_right (int): right chunk size for time-restricted Conformer encoder

    """

    def __init__(self, input_dim, enc_type, n_heads, kernel_size,
                 n_layers, n_layers_sub1, n_layers_sub2,
                 d_model, d_ff, ffn_bottleneck_dim, last_proj_dim,
                 pe_type, layer_norm_eps, ffn_activation,
                 dropout_in, dropout, dropout_att, dropout_layer,
                 n_stacks, n_splices,
                 conv_in_channel, conv_channels, conv_kernel_sizes, conv_strides, conv_poolings,
                 conv_batch_norm, conv_layer_norm, conv_bottleneck_dim, conv_param_init,
                 task_specific_layer, param_init,
                 chunk_size_left, chunk_size_current, chunk_size_right):

        super(ConformerEncoder, self).__init__()

        if n_layers_sub1 < 0 or (n_layers_sub1 > 1 and n_layers < n_layers_sub1):
            raise ValueError('Set n_layers_sub1 between 1 to n_layers.')
        if n_layers_sub2 < 0 or (n_layers_sub2 > 1 and n_layers_sub1 < n_layers_sub2):
            raise ValueError('Set n_layers_sub2 between 1 to n_layers_sub1.')

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pe_type = pe_type
        self.scale = math.sqrt(d_model)

        # for streaming encoder
        self.chunk_size_left = chunk_size_left
        self.chunk_size_current = chunk_size_current
        self.chunk_size_right = chunk_size_right
        self.latency_controlled = chunk_size_left > 0 or chunk_size_current > 0 or chunk_size_right > 0

        # for hierarchical encoder
        self.n_layers_sub1 = n_layers_sub1
        self.n_layers_sub2 = n_layers_sub2
        self.task_specific_layer = task_specific_layer

        # for bridge layers
        self.bridge = None
        self.bridge_sub1 = None
        self.bridge_sub2 = None

        # for attention plot
        self.aws_dict = {}
        self.data_dict = {}

        # Setting for CNNs
        if conv_channels:
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
                                    layer_norm_eps=layer_norm_eps,
                                    residual=False,
                                    bottleneck_dim=d_model,
                                    param_init=conv_param_init)
            self._odim = self.conv.output_dim
        else:
            self.conv = None
            self._odim = input_dim * n_splices * n_stacks
            self.embed = nn.Linear(self._odim, d_model)

        # calculate subsampling factor
        self._factor = 1
        if self.conv is not None:
            self._factor *= self.conv.subsampling_factor

        self.pos_emb = XLPositionalEmbedding(d_model, dropout)  # TODO: dropout_in?
        assert pe_type == 'relative'

        self.layers = nn.ModuleList([copy.deepcopy(ConformerEncoderBlock(
            d_model, d_ff, n_heads, kernel_size, dropout, dropout_att, dropout_layer,
            layer_norm_eps, ffn_activation, param_init,
            ffn_bottleneck_dim=ffn_bottleneck_dim))
            for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self._odim = d_model

        if n_layers_sub1 > 0:
            if task_specific_layer:
                self.layer_sub1 = ConformerEncoderBlock(
                    d_model, d_ff, n_heads, kernel_size, dropout, dropout_att, dropout_layer,
                    layer_norm_eps, ffn_activation, param_init,
                    ffn_bottleneck_dim=ffn_bottleneck_dim)
            self.norm_out_sub1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                self.bridge_sub1 = nn.Linear(self._odim, last_proj_dim)

        if n_layers_sub2 > 0:
            if task_specific_layer:
                self.layer_sub2 = ConformerEncoderBlock(
                    d_model, d_ff, n_heads, kernel_size, dropout, dropout_att, dropout_layer,
                    layer_norm_eps, ffn_activation, param_init,
                    ffn_bottleneck_dim=ffn_bottleneck_dim)
            self.norm_out_sub2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                self.bridge_sub2 = nn.Linear(self._odim, last_proj_dim)

        if last_proj_dim > 0 and last_proj_dim != self.output_dim:
            self.bridge = nn.Linear(self._odim, last_proj_dim)
            self._odim = last_proj_dim

        self.reset_parameters(param_init)

    @staticmethod
    def add_args(parser, args):
        """Add arguments."""
        group = parser.add_argument_group("Transformer encoder")
        if 'conv' in args.enc_type:
            parser = ConvEncoder.add_args(parser, args)
        # Transformer common
        if not hasattr(args, 'transformer_d_model'):
            group.add_argument('--transformer_d_model', type=int, default=256,
                               help='number of units in the MHA layer')
            group.add_argument('--transformer_d_ff', type=int, default=2048,
                               help='number of units in the FFN layer')
            group.add_argument('--transformer_d_ff_bottleneck_dim', type=int, default=0,
                               help='bottleneck dimension in the FFN layer')
            group.add_argument('--transformer_n_heads', type=int, default=4,
                               help='number of heads in the MHA layer')
            group.add_argument('--transformer_layer_norm_eps', type=float, default=1e-12,
                               help='epsilon value for layer normalization')
            group.add_argument('--transformer_ffn_activation', type=str, default='relu',
                               choices=['relu', 'gelu', 'gelu_accurate', 'glu', 'swish'],
                               help='nonlinear activation for the FFN layer')
            group.add_argument('--transformer_param_init', type=str, default='xavier_uniform',
                               choices=['xavier_uniform', 'pytorch'],
                               help='parameter initializatin')
        # NOTE: These checks are important to avoid conflict with args in Transformer decoder

        # Conformer encoder specific
        group.add_argument('--transformer_enc_pe_type', type=str, default='relative',
                           choices=['relative'],
                           help='type of positional encoding for the Transformer encoder')
        group.add_argument('--conformer_kernel_size', type=int, default=32,
                           help='kernel size for depthwise convolution in convolution module for Conformer encoder layers')
        group.add_argument('--dropout_enc_layer', type=float, default=0.0,
                           help='LayerDrop probability for Conformer encoder layers')
        # streaming
        group.add_argument('--lc_chunk_size_left', type=int, default=0,
                           help='left chunk size for latency-controlled Conformer encoder')
        group.add_argument('--lc_chunk_size_current', type=int, default=0,
                           help='current chunk size (and hop size) for latency-controlled Conformer encoder')
        group.add_argument('--lc_chunk_size_right', type=int, default=0,
                           help='right chunk size for latency-controlled Conformer encoder')
        return parser

    def reset_parameters(self, param_init):
        """Initialize parameters."""
        if param_init == 'xavier_uniform':
            logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
            if self.conv is None:
                nn.init.xavier_uniform_(self.embed.weight)
                nn.init.constant_(self.embed.bias, 0.)
            if self.bridge is not None:
                nn.init.xavier_uniform_(self.bridge.weight)
                nn.init.constant_(self.bridge.bias, 0.)
            if self.bridge_sub1 is not None:
                nn.init.xavier_uniform_(self.bridge_sub1.weight)
                nn.init.constant_(self.bridge_sub1.bias, 0.)
            if self.bridge_sub2 is not None:
                nn.init.xavier_uniform_(self.bridge_sub2.weight)
                nn.init.constant_(self.bridge_sub2.bias, 0.)

    def forward(self, xs, xlens, task, use_cache=False, streaming=False):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim]`
            xlens (list): `[B]`
            task (str): not supported now
            use_cache (bool):
            streaming (bool): streaming encoding
        Returns:
            eouts (dict):
                xs (FloatTensor): `[B, T, d_model]`
                xlens (list): `[B]`

        """
        eouts = {'ys': {'xs': None, 'xlens': None},
                 'ys_sub1': {'xs': None, 'xlens': None},
                 'ys_sub2': {'xs': None, 'xlens': None}}

        N_l = self.chunk_size_left
        N_c = self.chunk_size_current
        N_r = self.chunk_size_right

        bs, xmax, idim = xs.size()

        if self.latency_controlled:
            xs = blockwise(xs, N_l, N_c, N_r)

        if self.conv is None:
            xs = self.embed(xs)
        else:
            # Path through CNN blocks
            xs, xlens = self.conv(xs, xlens)

        if not self.training:
            self.data_dict['elens'] = tensor2np(xlens)

        if self.latency_controlled:
            # streaming Conformer encoder
            _N_l = max(0, N_l // self.subsampling_factor)
            _N_c = N_c // self.subsampling_factor

            n_blocks = xs.size(0) // bs
            emax = xmax // self.subsampling_factor
            if xmax % self.subsampling_factor != 0:
                emax += 1

            xs = xs * self.scale
            pos_idxs = torch.arange(xs.size(1) - 1, -1, -1.0, dtype=torch.float)
            pos_embs = self.pos_emb(pos_idxs, self.device_id)

            xx_mask = None  # NOTE: no mask
            for lth, layer in enumerate(self.layers):
                xs = layer(xs, xx_mask, pos_embs=pos_embs)
                if not self.training:
                    n_heads = layer.xx_aws.size(1)
                    xx_aws = layer.xx_aws[:, :, _N_l:_N_l + _N_c, _N_l:_N_l + _N_c]
                    xx_aws = xx_aws.view(bs, n_blocks, n_heads, _N_c, _N_c)
                    xx_aws_center = xx_aws.new_zeros(bs, n_heads, emax, emax)
                    for blc_id in range(n_blocks):
                        offset = blc_id * _N_c
                        emax_blc = xx_aws_center[:, :, offset:offset + _N_c].size(2)
                        xx_aws_chunk = xx_aws[:, blc_id, :, :emax_blc, :emax_blc]
                        xx_aws_center[:, :, offset:offset + _N_c, offset:offset + _N_c] = xx_aws_chunk
                    self.aws_dict['xx_aws_layer%d' % lth] = tensor2np(xx_aws_center)

            # Extract the center region
            xs = xs[:, _N_l:_N_l + _N_c]  # `[B * n_blocks, _N_c, d_model]`
            xs = xs.contiguous().view(bs, -1, xs.size(2))
            xs = xs[:, :emax]

        else:
            bs, xmax, idim = xs.size()
            xs = xs * self.scale

            # Create the self-attention mask
            xx_mask = make_pad_mask(xlens, self.device_id).unsqueeze(2).repeat([1, 1, xmax])

            pos_idxs = torch.arange(xmax - 1, -1, -1.0, dtype=torch.float)
            pos_embs = self.pos_emb(pos_idxs, self.device_id)

            for lth, layer in enumerate(self.layers):
                xs = layer(xs, xx_mask, pos_embs=pos_embs)
                if not self.training:
                    self.aws_dict['xx_aws_layer%d' % lth] = tensor2np(layer.xx_aws)

                # Pick up outputs in the sub task before the projection layer
                if lth == self.n_layers_sub1 - 1:
                    xs_sub1 = self.layer_sub1(
                        xs, xx_mask, pos_embs=pos_embs) if self.task_specific_layer else xs.clone()
                    xs_sub1 = self.norm_out_sub1(xs_sub1)
                    if self.bridge_sub1 is not None:
                        xs_sub1 = self.bridge_sub1(xs_sub1)
                    if task == 'ys_sub1':
                        eouts[task]['xs'], eouts[task]['xlens'] = xs_sub1, xlens
                        return eouts
                if lth == self.n_layers_sub2 - 1:
                    xs_sub2 = self.layer_sub2(
                        xs, xx_mask, pos_embs=pos_embs) if self.task_specific_layer else xs.clone()
                    xs_sub2 = self.norm_out_sub2(xs_sub2)
                    if self.bridge_sub2 is not None:
                        xs_sub2 = self.bridge_sub2(xs_sub2)
                    if task == 'ys_sub2':
                        eouts[task]['xs'], eouts[task]['xlens'] = xs_sub2, xlens
                        return eouts

        xs = self.norm_out(xs)

        # Bridge layer
        if self.bridge is not None:
            xs = self.bridge(xs)

        if task in ['all', 'ys']:
            eouts['ys']['xs'], eouts['ys']['xlens'] = xs, xlens
        if self.n_layers_sub1 >= 1 and task == 'all':
            eouts['ys_sub1']['xs'], eouts['ys_sub1']['xlens'] = xs_sub1, xlens
        if self.n_layers_sub2 >= 1 and task == 'all':
            eouts['ys_sub2']['xs'], eouts['ys_sub2']['xlens'] = xs_sub2, xlens
        return eouts


class ConformerEncoderBlock(nn.Module):
    """A single layer of the Conformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        dropout_layer (float): LayerDrop probability
        layer_norm_eps (float): epsilon parameter for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        param_init (str): parameter initialization method
        ffn_bottleneck_dim (int): bottleneck dimension for the light-weight FFN layer

    """

    def __init__(self, d_model, d_ff, n_heads, kernel_size,
                 dropout, dropout_att, dropout_layer,
                 layer_norm_eps, ffn_activation, param_init,
                 ffn_bottleneck_dim=0):
        super(ConformerEncoderBlock, self).__init__()

        self.n_heads = n_heads
        self.fc_factor = 0.5

        # first half position-wise feed-forward
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward1 = FFN(d_model, d_ff, dropout, ffn_activation, param_init,
                                 ffn_bottleneck_dim)

        # conv module
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.conv = ConformerConvBlock(d_model, kernel_size, param_init)

        # self-attention
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = RelMHA(kdim=d_model,
                                qdim=d_model,
                                adim=d_model,
                                odim=d_model,
                                n_heads=n_heads,
                                dropout=dropout_att,
                                param_init=param_init)

        # second half position-wise feed-forward
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward2 = FFN(d_model, d_ff, dropout, ffn_activation, param_init,
                                 ffn_bottleneck_dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout_layer = dropout_layer

        self.reset_visualization()

    @property
    def xx_aws(self):
        return self._xx_aws

    def reset_visualization(self):
        self._xx_aws = None

    def forward(self, xs, xx_mask=None, pos_embs=None, u=None, v=None):
        """Conformer encoder layer definition.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
            xx_mask (ByteTensor): `[B, T, T]`
            pos_embs (LongTensor): `[L, 1, d_model]`
            u (FloatTensor): global parameter for relative positinal embedding
            v (FloatTensor): global parameter for relative positinal embedding
        Returns:
            xs (FloatTensor): `[B, T, d_model]`

        """
        self.reset_visualization()

        # LayerDrop
        if self.dropout_layer > 0 and self.training and random.random() >= self.dropout_layer:
            return xs

        # first half FFN
        residual = xs
        xs = self.norm1(xs)
        xs = self.feed_forward1(xs)
        xs = self.fc_factor * self.dropout(xs) + residual  # Macaron FFN

        # conv
        residual = xs
        xs = self.norm2(xs)
        xs = self.conv(xs)
        xs = self.dropout(xs) + residual

        # self-attention
        residual = xs
        xs = self.norm3(xs)
        # relative positional encoding
        memory = None
        xs, self._xx_aws = self.self_attn(xs, xs, memory, pos_embs, xx_mask, u, v)
        xs = self.dropout(xs) + residual

        # second half FFN
        residual = xs
        xs = self.norm4(xs)
        xs = self.feed_forward2(xs)
        xs = self.fc_factor * self.dropout(xs) + residual  # Macaron FFN

        return xs
