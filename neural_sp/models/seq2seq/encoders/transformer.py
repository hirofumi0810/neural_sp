#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Self-attention encoder for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import shutil
import torch
import torch.nn as nn

from neural_sp.models.modules.transformer import PositionalEncoding
from neural_sp.models.modules.transformer import TransformerEncoderBlock
from neural_sp.models.seq2seq.encoders.conv import ConvEncoder
from neural_sp.models.seq2seq.encoders.encoder_base import EncoderBase
from neural_sp.models.torch_utils import make_pad_mask
from neural_sp.models.torch_utils import repeat
from neural_sp.models.torch_utils import tensor2np
from neural_sp.utils import mkdir_join

import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


class TransformerEncoder(EncoderBase):
    """Transformer encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        attn_type (str): type of attention
        n_heads (int): number of heads for multi-head attention
        n_layers (int): number of blocks
        d_model (int): dimension of MultiheadAttentionMechanism
        d_ff (int): dimension of PositionwiseFeedForward
        pe_type (str): type of positional encoding
        layer_norm_eps (float): epsilon value for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        dropout_in (float): dropout probability for input-hidden connection
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        last_proj_dim (int): dimension of the last projection layer
        n_stacks (int): number of frames to stack
        n_splices (int): frames to splice. Default is 1 frame.
        conv_in_channel (int): number of channels of input features
        conv_channels (int): number of channles in the CNN blocks
        conv_kernel_sizes (list): size of kernels in the CNN blocks
        conv_strides (list): number of strides in the CNN blocks
        conv_poolings (list): size of poolings in the CNN blocks
        conv_batch_norm (bool): apply batch normalization only in the CNN blocks
        conv_bottleneck_dim (int): dimension of the bottleneck layer between CNN and self-attention layers
        conv_param_init (float): only for CNN layers before Transformer layers
        chunk_size_left (int): left chunk size for time-restricted Transformer encoder
        chunk_size_current (int): current chunk size for time-restricted Transformer encoder
        chunk_size_right (int): right chunk size for time-restricted Transformer encoder

    """

    def __init__(self,
                 input_dim,
                 attn_type,
                 n_heads,
                 n_layers,
                 d_model,
                 d_ff,
                 pe_type='add',
                 layer_norm_eps=1e-12,
                 ffn_activation='relu',
                 dropout_in=0.,
                 dropout=0.,
                 dropout_att=0.,
                 last_proj_dim=0,
                 n_stacks=1,
                 n_splices=1,
                 conv_in_channel=1,
                 conv_channels=0,
                 conv_kernel_sizes=[],
                 conv_strides=[],
                 conv_poolings=[],
                 conv_batch_norm=False,
                 conv_bottleneck_dim=0,
                 conv_param_init=0.1,
                 chunk_size_left=0,
                 chunk_size_current=0,
                 chunk_size_right=0):

        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pe_type = pe_type
        self.chunk_size_left = chunk_size_left
        self.chunk_size_current = chunk_size_current
        self.chunk_size_right = chunk_size_right

        # Setting for CNNs before RNNs
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
            self._factor *= self.conv.subsampling_factor()

        self.pos_enc = PositionalEncoding(d_model, dropout_in, pe_type)
        self.layers = repeat(TransformerEncoderBlock(
            d_model, d_ff, attn_type, n_heads,
            dropout, dropout_att,
            layer_norm_eps, ffn_activation), n_layers)
        self.norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)

        if last_proj_dim != self.output_dim:
            self.bridge = nn.Linear(self._odim, last_proj_dim)
            self._odim = last_proj_dim
        else:
            self.bridge = None
            self._odim = d_model

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        if self.conv is None:
            nn.init.xavier_uniform_(self.embed.weight)
            nn.init.constant_(self.embed.bias, 0.)
        if self.bridge is not None:
            nn.init.xavier_uniform_(self.bridge.weight)
            nn.init.constant_(self.bridge.bias, 0.)

    def forward(self, xs, xlens, task, use_cache=False, streaming=False):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim]`
            xlens (list): `[B]`
            task (str): not supported now
        Returns:
            eouts (dict):
                xs (FloatTensor): `[B, T, d_model]`
                xlens (list): `[B]`

        """
        eouts = {'ys': {'xs': None, 'xlens': None},
                 'ys_sub1': {'xs': None, 'xlens': None},
                 'ys_sub2': {'xs': None, 'xlens': None}}

        if self.conv is None:
            xs = self.embed(xs)
        else:
            # Path through CNN blocks before RNN layers
            xs, xlens = self.conv(xs, xlens)

        bs, xmax, idim = xs.size()
        xs = self.pos_enc(xs)
        if self.chunk_size_left > 0:
            # Time-restricted self-attention for streaming models
            cs_l = self.chunk_size_left
            cs_c = self.chunk_size_current
            cs_r = self.chunk_size_right
            hop_size = self.chunk_size_current
            xs_chunks = []
            xx_aws = [[] for l in range(self.n_layers)]
            xs_pad = torch.cat([xs.new_zeros(bs, cs_l, idim), xs,
                                xs.new_zeros(bs, cs_r, idim)], dim=1)
            # TODO: remove right padding
            for t in range(cs_l, xmax + cs_r, hop_size):
                xs_chunk = xs_pad[:, t - cs_l:t + cs_c + cs_r]
                for l in range(self.n_layers):
                    xs_chunk, xx_aws_chunk = self.layers[l](xs_chunk, None)  # no mask
                    xx_aws[l].append(xx_aws_chunk[:, :, cs_l:cs_l + cs_c,
                                                  cs_l:cs_l + cs_c])
                xs_chunks.append(xs_chunk[:, cs_l:cs_l + cs_c])
            xs = torch.cat(xs_chunks, dim=1)[:, :xmax]
            if not self.training:
                for l in range(self.n_layers):
                    setattr(self, 'xx_aws_layer%d' % l, tensor2np(torch.cat(xx_aws[l], dim=3)[:, :, :xmax, :xmax]))
        else:
            # Create the self-attention mask
            xx_mask = make_pad_mask(xlens, self.device_id).unsqueeze(2).repeat([1, 1, xmax])

            for l in range(self.n_layers):
                xs, xx_aws = self.layers[l](xs, xx_mask)
                if not self.training:
                    setattr(self, 'xx_aws_layer%d' % l, tensor2np(xx_aws))
        xs = self.norm_out(xs)

        # Bridge layer
        if self.bridge is not None:
            xs = self.bridge(xs)

        eouts['ys']['xs'] = xs
        eouts['ys']['xlens'] = xlens
        return eouts

    def _plot_attention(self, save_path, n_cols=2):
        """Plot attention for each head in all layers."""
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator

        save_path = mkdir_join(save_path, 'enc_xx_att_weights')

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        for l in range(self.n_layers):
            if not hasattr(self, 'xx_aws_layer%d' % l):
                continue

            xx_aws = getattr(self, 'xx_aws_layer%d' % l)

            plt.clf()
            fig, axes = plt.subplots(self.n_heads // n_cols, n_cols, figsize=(20, 8))
            for h in range(self.n_heads):
                if self.n_heads > n_cols:
                    ax = axes[h // n_cols, h % n_cols]
                else:
                    ax = axes[h]
                ax.imshow(xx_aws[-1, h, :, :], aspect="auto")
                ax.grid(False)
                ax.set_xlabel("Input (head%d)" % h)
                ax.set_ylabel("Output (head%d)" % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            fig.tight_layout()
            fig.savefig(os.path.join(save_path, 'layer%d.png' % (l)), dvi=500)
            plt.close()
