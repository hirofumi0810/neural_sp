#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utilities for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import torch
import torch.nn as nn

from neural_sp.models.modules.causal_conv import CausalConv1d
from neural_sp.models.modules.gelu import gelu, gelu_accurate
from neural_sp.models.modules.glu import LinearGLUBlock
from neural_sp.models.modules.multihead_attention import MultiheadAttentionMechanism
from neural_sp.models.modules.sync_bidir_multihead_attention import SyncBidirMultiheadAttentionMechanism

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer.

    Args:
        d_model (int): dimension of MultiheadAttentionMechanism
        dropout (float): dropout probability
        pe_type (str): type of positional encoding
        max_len (int):

    """

    def __init__(self, d_model, dropout, pe_type, max_len=5000,
                 conv_kernel_size=3, conv_nlayers=3, layer_norm_eps=1e-12):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.pe_type = pe_type
        self.scale = math.sqrt(self.d_model)

        if pe_type == '1dconv':
            causal_conv1d = CausalConv1d(in_channels=d_model,
                                         out_channels=d_model,
                                         kernel_size=conv_kernel_size,
                                         stride=1)
            layers = []
            for l in range(conv_nlayers):
                layers.append(copy.deepcopy(causal_conv1d))
                layers.append(nn.LayerNorm(d_model, eps=layer_norm_eps))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
            self.pe = nn.Sequential(*layers)
            self.dropout = nn.Dropout(p=dropout)  # for the last layer
        elif pe_type != 'none':
            # Compute the positional encodings once in log space.
            pe = torch.zeros(max_len, d_model, dtype=torch.float32)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # for batch dimension
            self.register_buffer('pe', pe)
            self.dropout = nn.Dropout(p=dropout)

        logger.info('Positional encoding: %s' % pe_type)

    def forward(self, xs):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
        Returns:
            xs (FloatTensor): `[B, T, d_model]`

        """
        xs = xs * self.scale  # after embedding

        if self.pe_type == 'none':
            return xs

        if self.pe_type == 'add':
            xs = xs + self.pe[:, :xs.size(1)]
        elif self.pe_type == 'concat':
            xs = torch.cat([xs, self.pe[:, :xs.size(1)]], dim=-1)
        elif self.pe_type == '1dconv':
            xs = self.pe(xs)
        else:
            raise NotImplementedError(self.pe_type)
        return self.dropout(xs)


class PositionwiseFeedForward(nn.Module):
    """Positionwise fully-connected feed-forward neural network.

    Args:
        d_in (int): input dimension (equal to d_model)
        d_ff (int): dimention of PositionwiseFeedForward
        d_out (int): output dimension of PositionwiseFeedForward
        dropout (float): dropout probability (equal to d_model)
        activation: non-linear function
        param_init (str):

    """

    def __init__(self, d_in, d_ff, d_out, dropout, activation, param_init):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_in, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(p=dropout)
        if activation == 'relu':
            self.activation = torch.relu
        elif activation == 'gelu':
            self.activation = lambda x: gelu(x)
        elif activation == 'gelu_accurate':
            self.activation = lambda x: gelu_accurate(x)
        elif activation == 'glu':
            self.activation = LinearGLUBlock(d_ff)
        else:
            raise NotImplementedError(activation)
        logger.info('FFN activation: %s' % activation)

        if param_init == 'xavier_uniform':
            self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        # NOTE: see https://github.com/pytorch/fairseq/blob/master/fairseq/modules/transformer_layer.py
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0.)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
            elif p.dim() == 2:
                nn.init.xavier_uniform_(p)
                logger.info('Initialize %s with %s' % (n, 'xavier_uniform'))
            else:
                raise ValueError(n)

    def forward(self, xs):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
        Returns:
            xs (FloatTensor): `[B, T, d_model]`

        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class TransformerEncoderBlock(nn.Module):
    """A single layer of the transformer encoder.

    Args:
        d_model (int): dimension of MultiheadAttentionMechanism
        d_ff (int): dimention of PositionwiseFeedForward
        atype (str): type of attention
        n_heads (int): number of heads for multi-head attention
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        layer_norm_eps (float): epsilon parameter for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        param_init (str):

    """

    def __init__(self, d_model, d_ff, atype, n_heads, dropout, dropout_att,
                 layer_norm_eps, ffn_activation, param_init):
        super(TransformerEncoderBlock, self).__init__()

        self.n_heads = n_heads

        # self-attention
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = MultiheadAttentionMechanism(kdim=d_model,
                                                     qdim=d_model,
                                                     adim=d_model,
                                                     atype=atype,
                                                     n_heads=n_heads,
                                                     dropout=dropout_att,
                                                     param_init=param_init)

        # feed-forward
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, d_model, dropout, ffn_activation, param_init)

        self.dropout = nn.Dropout(dropout)

    def forward(self, xs, xx_mask=None):
        """Transformer encoder layer definition.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
            xx_mask (ByteTensor): `[B, T, T]`
        Returns:
            xs (FloatTensor): `[B, T, d_model]`
            xx_aws (FloatTensor): `[B, T, T]`

        """
        # self-attention
        residual = xs
        xs = self.norm1(xs)
        xs, xx_aws = self.self_attn(xs, xs, xs, mask=xx_mask, cache=False)
        xs = self.dropout(xs) + residual

        # position-wise feed-forward
        residual = xs
        xs = self.norm2(xs)
        xs = self.dropout(self.feed_forward(xs)) + residual

        return xs, xx_aws


class TransformerDecoderBlock(nn.Module):
    """A single layer of the transformer decoder.

        Args:
            d_model (int): dimension of MultiheadAttentionMechanism
            d_ff (int): dimention of PositionwiseFeedForward
            atype (str): type of attention
            n_heads (int): number of heads for multi-head attention
            dropout (float): dropout probabilities for linear layers
            dropout_att (float): dropout probabilities for attention probabilities
            layer_norm_eps (float):
            ffn_activation (str): nonolinear function for PositionwiseFeedForward
            param_init (str):
            src_tgt_attention (bool): if False, ignore source-target attention

    """

    def __init__(self, d_model, d_ff, atype, n_heads, dropout, dropout_att,
                 layer_norm_eps, ffn_activation, param_init, src_tgt_attention=True):
        super(TransformerDecoderBlock, self).__init__()

        self.atype = atype
        self.n_heads = n_heads
        self.src_tgt_attention = src_tgt_attention

        # self-attention
        if atype == "average":
            raise NotImplementedError
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.self_attn = MultiheadAttentionMechanism(kdim=d_model,
                                                         qdim=d_model,
                                                         adim=d_model,
                                                         atype=atype,
                                                         n_heads=n_heads,
                                                         dropout=dropout_att,
                                                         param_init=param_init)

        # attention over encoder stacks
        if src_tgt_attention:
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.src_attn = MultiheadAttentionMechanism(kdim=d_model,
                                                        qdim=d_model,
                                                        adim=d_model,
                                                        atype=atype,
                                                        n_heads=n_heads,
                                                        dropout=dropout_att,
                                                        param_init=param_init)

        # feed-forward
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, d_model, dropout, ffn_activation, param_init)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, ys, yy_mask, xs=None, xy_mask=None, cache=None):
        """Transformer decoder layer definition.

        Args:
            ys (FloatTensor): `[B, L, d_model]`
            yy_mask (ByteTensor): `[B, L, L]`
            xs (FloatTensor): encoder outputs. `[B, T, d_model]`
            xy_mask (ByteTensor): `[B, L, T]`
            cache (FloatTensor): `[B, L-1, d_model]`
        Returns:
            out (FloatTensor): `[B, L, d_model]`
            yy_aw (FloatTensor)`[B, L, L]`
            xy_aw (FloatTensor): `[B, L, T]`

        """
        residual = ys
        ys = self.norm1(ys)

        if cache is not None:
            ys_q = ys[:, -1:]
            residual = residual[:, -1:]
            yy_mask = yy_mask[:, -1:]
        else:
            ys_q = ys

        # self-attention
        if self.atype == "average":
            raise NotImplementedError
        else:
            out, yy_aw = self.self_attn(ys, ys, ys_q, mask=yy_mask, cache=False)  # k/v/q
            out = self.dropout(out) + residual

        # attention over encoder stacks
        xy_aw = None
        if self.src_tgt_attention:
            residual = out
            out = self.norm2(out)
            out, xy_aw = self.src_attn(xs, xs, out, mask=xy_mask, cache=False)  # k/v/q
            out = self.dropout(out) + residual

        # position-wise feed-forward
        residual = out
        out = self.norm3(out)
        out = self.feed_forward(out)
        out = self.dropout(out) + residual

        if cache is not None:
            out = torch.cat([cache, out], dim=1)

        return out, yy_aw, xy_aw


class SyncBidirTransformerDecoderBlock(nn.Module):
    """A single layer of the synchronous bidirectional transformer decoder.

        Args:
            d_model (int): dimension of MultiheadAttentionMechanism
            d_ff (int): dimention of PositionwiseFeedForward
            atype (str): type of attention
            n_heads (int): number of heads for multi-head attention
            dropout (float): dropout probabilities for linear layers
            dropout_att (float): dropout probabilities for attention probabilities
            layer_norm_eps (float):
            ffn_activation (str): nonolinear function for PositionwiseFeedForward
            param_init (str):

    """

    def __init__(self, d_model, d_ff, atype, n_heads, dropout, dropout_att,
                 layer_norm_eps, ffn_activation, param_init):
        super(SyncBidirTransformerDecoderBlock, self).__init__()

        self.atype = atype
        self.n_heads = n_heads

        # synchronous bidirectional attention
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = SyncBidirMultiheadAttentionMechanism(kdim=d_model,
                                                              qdim=d_model,
                                                              adim=d_model,
                                                              atype=atype,
                                                              n_heads=n_heads,
                                                              dropout=dropout_att,
                                                              param_init=param_init)

        # attention over encoder stacks
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.src_attn = MultiheadAttentionMechanism(kdim=d_model,
                                                    qdim=d_model,
                                                    adim=d_model,
                                                    atype=atype,
                                                    n_heads=n_heads,
                                                    dropout=dropout_att,
                                                    param_init=param_init)

        # feed-forward
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, d_model, dropout, ffn_activation, param_init)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, ys_fwd, ys_bwd, yy_mask, xs, xy_mask, cache=None):
        """Transformer decoder layer definition.

        Args:
            ys_fwd (FloatTensor): `[B, L, d_model]`
            ys_bwd (FloatTensor): `[B, L, d_model]`
            yy_mask (ByteTensor): `[B, L, L]`
            xs (FloatTensor): encoder outputs. `[B, T, d_model]`
            xy_mask (ByteTensor): `[B, L, T]`
            cache (FloatTensor): `[B, L-1, d_model]`
        Returns:
            out (FloatTensor): `[B, L, d_model]`
            yy_aw (FloatTensor)`[B, L, L]`
            xy_aw (FloatTensor): `[B, L, T]`

        """
        residual_fwd = ys_fwd
        residual_bwd = ys_bwd
        ys_fwd = self.norm1(ys_fwd)
        ys_bwd = self.norm1(ys_bwd)

        if cache is not None:
            ys_fwd_q = ys_fwd[:, -1:]
            ys_bwd_q = ys_bwd[:, -1:]
            residual_fwd = residual_fwd[:, -1:]
            residual_bwd = residual_bwd[:, -1:]
            yy_mask = yy_mask[:, -1:]
        else:
            ys_fwd_q = ys_fwd
            ys_bwd_q = ys_bwd

        # synchronous bidirectional attention
        out_fwd, out_bwd, yy_aw_fwd_h, yy_aw_fwd_f, yy_aw_bwd_h, yy_aw_bwd_f = self.self_attn(
            ys_fwd, ys_fwd, ys_fwd_q,  # k/v/q
            ys_bwd, ys_bwd, ys_bwd_q,  # k/v/q
            mask=yy_mask, cache=False)
        out_fwd = self.dropout(out_fwd) + residual_fwd
        out_bwd = self.dropout(out_bwd) + residual_bwd

        # attention over encoder stacks
        # fwd
        residual_fwd = out_fwd
        out_fwd = self.norm2(out_fwd)
        out_fwd, xy_aw = self.src_attn(xs, xs, out_fwd, mask=xy_mask, cache=False)  # k/v/q
        out_fwd = self.dropout(out_fwd) + residual_fwd
        # bwd
        residual_bwd = out_bwd
        out_bwd = self.norm2(out_bwd)
        out_bwd, xy_aw = self.src_attn(xs, xs, out_bwd, mask=xy_mask, cache=False)  # k/v/q
        out_bwd = self.dropout(out_bwd) + residual_bwd

        # position-wise feed-forward
        # fwd
        residual_fwd = out_fwd
        out_fwd = self.norm3(out_fwd)
        out_fwd = self.feed_forward(out_fwd)
        out_fwd = self.dropout(out_fwd) + residual_fwd
        # bwd
        residual_bwd = out_bwd
        out_bwd = self.norm3(out_bwd)
        out_bwd = self.feed_forward(out_bwd)
        out_bwd = self.dropout(out_bwd) + residual_bwd

        if cache is not None:
            out_fwd = torch.cat([cache, out_fwd], dim=1)
            out_bwd = torch.cat([cache, out_bwd], dim=1)

        return out_fwd, out_bwd, yy_aw_fwd_h, yy_aw_fwd_f, yy_aw_bwd_h, yy_aw_bwd_f, xy_aw
