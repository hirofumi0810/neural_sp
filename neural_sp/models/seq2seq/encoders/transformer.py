#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Self-attention encoder for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import os
import shutil
import torch
import torch.nn as nn

from neural_sp.models.modules.initialization import init_with_normal_dist
from neural_sp.models.modules.positinal_embedding import PositionalEncoding
from neural_sp.models.modules.positinal_embedding import XLPositionalEmbedding
from neural_sp.models.modules.transformer import TransformerEncoderBlock
from neural_sp.models.seq2seq.encoders.conv import ConvEncoder
from neural_sp.models.seq2seq.encoders.encoder_base import EncoderBase
from neural_sp.models.torch_utils import make_pad_mask
from neural_sp.models.torch_utils import tensor2np
from neural_sp.utils import mkdir_join

import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


class TransformerEncoder(EncoderBase):
    """Transformer encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        enc_type (str): type of encoder
        attn_type (str): type of attention
        n_heads (int): number of heads for multi-head attention
        n_layers (int): number of blocks
        n_layers_sub1 (int): number of layers in the 1st auxiliary task
        n_layers_sub2 (int): number of layers in the 2nd auxiliary task
        d_model (int): dimension of MultiheadAttentionMechanism
        d_ff (int): dimension of PositionwiseFeedForward
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
        conv_param_init (float): only for CNN layers before Transformer layers
        task_specific_layer (bool): add a task specific layer for each sub task
        param_init (str): parameter initialization method
        chunk_size_left (int): left chunk size for time-restricted Transformer encoder
        chunk_size_current (int): current chunk size for time-restricted Transformer encoder
        chunk_size_right (int): right chunk size for time-restricted Transformer encoder

    """

    def __init__(self, input_dim, enc_type, attn_type, n_heads,
                 n_layers, n_layers_sub1, n_layers_sub2,
                 d_model, d_ff, last_proj_dim,
                 pe_type, layer_norm_eps, ffn_activation,
                 dropout_in, dropout, dropout_att, dropout_layer,
                 n_stacks, n_splices,
                 conv_in_channel, conv_channels, conv_kernel_sizes, conv_strides, conv_poolings,
                 conv_batch_norm, conv_layer_norm, conv_bottleneck_dim, conv_param_init,
                 task_specific_layer, param_init,
                 chunk_size_left, chunk_size_current, chunk_size_right):

        super(TransformerEncoder, self).__init__()

        if n_layers_sub1 < 0 or (n_layers_sub1 > 1 and n_layers < n_layers_sub1):
            raise ValueError('Set n_layers_sub1 between 1 to n_layers.')
        if n_layers_sub2 < 0 or (n_layers_sub2 > 1 and n_layers_sub1 < n_layers_sub2):
            raise ValueError('Set n_layers_sub2 between 1 to n_layers_sub1.')

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pe_type = pe_type

        # for streaming TransformerXL encoder
        self.N_l = chunk_size_left
        self.N_c = chunk_size_current
        self.N_r = chunk_size_right
        self.latency_controlled = chunk_size_left > 0 or chunk_size_current > 0 or chunk_size_right > 0
        self.memory_transformer = ('transformer_xl' in enc_type)
        self.mem_len = chunk_size_left
        self.scale = math.sqrt(d_model)
        if self.memory_transformer:
            assert pe_type == 'none'
            assert chunk_size_left > 0
            assert chunk_size_current > 0

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

        if self.memory_transformer:
            self.pos_emb = XLPositionalEmbedding(d_model, dropout)
            self.u = nn.Parameter(torch.Tensor(self.n_heads, self.d_model // self.n_heads))
            self.v = nn.Parameter(torch.Tensor(self.n_heads, self.d_model // self.n_heads))
            # NOTE: u and v are global parameters
        self.pos_enc = PositionalEncoding(d_model, dropout_in, pe_type, param_init)

        self.layers = nn.ModuleList([copy.deepcopy(TransformerEncoderBlock(
            d_model, d_ff, attn_type, n_heads, dropout, dropout_att, dropout_layer,
            layer_norm_eps, ffn_activation, param_init,
            memory_transformer=self.memory_transformer))
            for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self._odim = d_model

        if n_layers_sub1 > 0:
            if task_specific_layer:
                self.layer_sub1 = TransformerEncoderBlock(
                    d_model, d_ff, attn_type, n_heads, dropout, dropout_att, dropout_layer,
                    layer_norm_eps, ffn_activation, param_init)
            self.norm_out_sub1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                self.bridge_sub1 = nn.Linear(self._odim, last_proj_dim)

        if n_layers_sub2 > 0:
            if task_specific_layer:
                self.layer_sub2 = TransformerEncoderBlock(
                    d_model, d_ff, attn_type, n_heads, dropout, dropout_att, dropout_layer,
                    layer_norm_eps, ffn_activation, param_init)
            self.norm_out_sub2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                self.bridge_sub2 = nn.Linear(self._odim, last_proj_dim)

        if last_proj_dim > 0 and last_proj_dim != self.output_dim:
            self.bridge = nn.Linear(self._odim, last_proj_dim)
            self._odim = last_proj_dim

        self.reset_parameters(param_init)

    def reset_parameters(self, param_init):
        """Initialize parameters."""
        if self.memory_transformer:
            logger.info('===== Initialize %s with normal distribution =====' % self.__class__.__name__)
            for n, p in self.named_parameters():
                if 'conv' in n:
                    continue
                init_with_normal_dist(n, p, std=0.02)

        elif param_init == 'xavier_uniform':
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

    def init_memory(self):
        """Initialize memory."""
        if self.device_id >= 0:
            return [torch.empty(0, dtype=torch.float).cuda(self.device_id)
                    for _ in range(self.n_layers)]
        else:
            return [torch.empty(0, dtype=torch.float)
                    for _ in range(self.n_layers)]

    def update_memory(self, memory_prev, hidden_states):
        """Update memory.

        Args:
            memory_prev (list): length `n_layers`, each of which contains [B, L_prev, d_model]`
            hidden_states (list): length `n_layers`, each of which contains [B, L, d_model]`
        Returns:
            new_mems (list): length `n_layers`, each of which contains `[B, mlen, d_model]`

        """
        assert len(hidden_states) == len(memory_prev)
        mlen = memory_prev[0].size(1) if memory_prev[0].dim() > 1 else 0
        qlen = hidden_states[0].size(1)

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + qlen
            start_idx = max(0, end_idx - (self.mem_len // self.subsampling_factor))
            for m, h in zip(memory_prev, hidden_states):
                cat = torch.cat([m, h], dim=1)  # `[B, mlen + qlen, d_model]`
                new_mems.append(cat[:, start_idx:end_idx].detach())  # `[B, self.mem_len, d_model]`

        return new_mems

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

        if self.latency_controlled:
            bs, xmax, idim = xs.size()
            n_blocks = xmax // self.N_c
            if xmax % self.N_c != 0:
                n_blocks += 1
            xs_tmp = xs.new_zeros(bs, n_blocks, self.N_l + self.N_c + self.N_r, idim)
            xs_pad = torch.cat([xs.new_zeros(bs, self.N_l, idim),
                                xs,
                                xs.new_zeros(bs, self.N_r, idim)], dim=1)
            for blc_id, t in enumerate(range(self.N_l, self.N_l + xmax, self.N_c)):
                xs_chunk = xs_pad[:, t - self.N_l:t + (self.N_c + self.N_r)]
                xs_tmp[:, blc_id, :xs_chunk.size(1), :] = xs_chunk
            xs = xs_tmp.view(bs * n_blocks, self.N_l + self.N_c + self.N_r, idim)

        if self.conv is None:
            xs = self.embed(xs)
        else:
            # Path through CNN blocks
            xs, xlens = self.conv(xs, xlens)
        if not self.training:
            self.data_dict['elens'] = tensor2np(xlens)

        if self.latency_controlled:
            N_l = max(0, self.N_l // self.subsampling_factor)
            N_c = self.N_c // self.subsampling_factor

            emax = xmax // self.subsampling_factor
            if xmax % self.subsampling_factor != 0:
                emax += 1

            xs = self.pos_enc(xs, scale=True)
            xx_mask = None
            for lth, layer in enumerate(self.layers):
                xs, xx_aws = layer(xs, xx_mask)
                if not self.training:
                    n_heads = xx_aws.size(1)
                    xx_aws = xx_aws[:, :, N_l:N_l + N_c, N_l:N_l + N_c]
                    xx_aws = xx_aws.view(bs, n_blocks, n_heads, N_c, N_c)
                    xx_aws_center = xx_aws.new_zeros(bs, n_heads, emax, emax)
                    for blc_id in range(n_blocks):
                        offset = blc_id * N_c
                        emax_blc = xx_aws_center[:, :, offset:offset + N_c].size(2)
                        xx_aws_chunk = xx_aws[:, blc_id, :, :emax_blc, :emax_blc]
                        xx_aws_center[:, :, offset:offset + N_c, offset:offset + N_c] = xx_aws_chunk
                    self.aws_dict['xx_aws_layer%d' % lth] = tensor2np(xx_aws_center)

            # Extract the center region
            xs = xs[:, N_l:N_l + N_c]  # `[B * n_blocks, N_c // subsampling_factor, d_model]`
            xs = xs.contiguous().view(bs, -1, xs.size(2))
            xs = xs[:, :emax]

        else:
            bs, xmax, idim = xs.size()
            xs = self.pos_enc(xs, scale=True)

            # Create the self-attention mask
            xx_mask = make_pad_mask(xlens, self.device_id).unsqueeze(2).repeat([1, 1, xmax])

            for lth, layer in enumerate(self.layers):
                xs, xx_aws = layer(xs, xx_mask)
                if not self.training:
                    self.aws_dict['xx_aws_layer%d' % lth] = tensor2np(xx_aws)

                # Pick up outputs in the sub task before the projection layer
                if lth == self.n_layers_sub1 - 1:
                    xs_sub1 = self.layer_sub1(xs, xx_mask)[0] if self.task_specific_layer else xs.clone()
                    xs_sub1 = self.norm_out_sub1(xs_sub1)
                    if self.bridge_sub1 is not None:
                        xs_sub1 = self.bridge_sub1(xs_sub1)
                    if task == 'ys_sub1':
                        eouts[task]['xs'], eouts[task]['xlens'] = xs_sub1, xlens
                        return eouts
                if lth == self.n_layers_sub2 - 1:
                    xs_sub2 = self.layer_sub2(xs, xx_mask)[0] if self.task_specific_layer else xs.clone()
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

    def _plot_attention(self, save_path, n_cols=2):
        """Plot attention for each head in all layers."""
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator

        _save_path = mkdir_join(save_path, 'enc_att_weights')

        # Clean directory
        if _save_path is not None and os.path.isdir(_save_path):
            shutil.rmtree(_save_path)
            os.mkdir(_save_path)

        for k, aw in self.aws_dict.items():
            elens = self.data_dict['elens']

            plt.clf()
            n_heads = aw.shape[1]
            n_cols_tmp = 1 if n_heads == 1 else n_cols
            fig, axes = plt.subplots(max(1, n_heads // n_cols_tmp), n_cols_tmp,
                                     figsize=(20, 8), squeeze=False)
            for h in range(n_heads):
                ax = axes[h // n_cols_tmp, h % n_cols_tmp]
                ax.imshow(aw[-1, h, :elens[-1], :elens[-1]], aspect="auto")
                ax.grid(False)
                ax.set_xlabel("Input (head%d)" % h)
                ax.set_ylabel("Output (head%d)" % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            fig.tight_layout()
            fig.savefig(os.path.join(_save_path, '%s.png' % k), dvi=500)
            plt.close()
