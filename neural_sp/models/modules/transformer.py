#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utilities for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import torch
import torch.nn as nn

from neural_sp.models.modules.gelu import gelu, gelu_accurate
from neural_sp.models.modules.glu import LinearGLUBlock
from neural_sp.models.modules.multihead_attention import MultiheadAttentionMechanism

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer.

    Args:
        d_model (int): dimension of MultiheadAttentionMechanism
        dropout (float): dropout probability
        pe_type (str): type of positional encoding
        max_len (int):

    """

    def __init__(self, d_model, dropout, pe_type, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.pe_type = pe_type

        if pe_type != 'none':
            # Compute the positional encodings once in log space.
            pe = torch.zeros(max_len, d_model, dtype=torch.float32)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # for batch dimension
            self.register_buffer('pe', pe)

            self.dropout = nn.Dropout(p=dropout)

    def forward(self, xs):
        xs = xs * math.sqrt(self.d_model)

        if self.pe_type == 'none':
            return xs

        if self.pe_type == 'add':
            xs = xs + self.pe[:, :xs.size(1)]
        elif self.pe_type == 'concat':
            xs = torch.cat([xs, self.pe[:, :xs.size(1)]], dim=-1)
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

    """

    def __init__(self, d_in, d_ff, d_out, dropout, activation='relu'):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_in, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
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

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
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
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class TransformerEncoderBlock(nn.Module):
    """A single layer of the transformer encoder.

    Args:
        d_model (int): dimension of MultiheadAttentionMechanism
        d_ff (int): dimention of PositionwiseFeedForward
        atype (str):
        n_heads (int): number of heads for multi-head attention
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        layer_norm_eps (float): epsilon parameter for layer normalization
        ffn_nonlinear (str): nonolinear function for PositionwiseFeedForward

    """

    def __init__(self, d_model, d_ff, atype, n_heads,
                 dropout, dropout_att, layer_norm_eps, ffn_nonlinear):
        super(TransformerEncoderBlock, self).__init__()

        self.n_heads = n_heads

        # self-attention
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = MultiheadAttentionMechanism(kdim=d_model,
                                                     qdim=d_model,
                                                     adim=d_model,
                                                     atype=atype,
                                                     n_heads=n_heads,
                                                     dropout=dropout_att)
        self.dropout1 = nn.Dropout(dropout)

        # feed-forward
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, d_model, dropout, ffn_nonlinear)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, xs, xx_mask=None):
        """Transformer encoder layer definition.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
            xx_mask (ByteTensor): `[B, T, T]`
        Returns:
            xs (FloatTensor): `[B, T, d_model]`
            xx_aws (FloatTensor): `[B, T, T]`

        """
        self.self_attn.reset()

        # self-attention
        residual = xs
        xs = self.norm1(xs)
        xs, xx_aws = self.self_attn(xs, xs, xs, mask=xx_mask)
        xs = self.dropout1(xs) + residual

        # position-wise feed-forward
        residual = xs
        xs = self.norm2(xs)
        xs = self.dropout2(self.feed_forward(xs)) + residual

        return xs, xx_aws


class TransformerDecoderBlock(nn.Module):
    """A single layer of the transformer decoder.

        Args:
            d_model (int): dimension of MultiheadAttentionMechanism
            d_ff (int): dimention of PositionwiseFeedForward
            atype (str):
            n_heads (int): number of heads for multi-head attention
            dropout (float): dropout probabilities for linear layers
            dropout_att (float): dropout probabilities for attention probabilities
            atype (str): type of self-attention, scaled_dot or average
            layer_norm_eps (float):
            src_tgt_attention (bool): if False, ignore source-target attention
            ffn_nonlinear (str): nonolinear function for PositionwiseFeedForward

    """

    def __init__(self, d_model, d_ff, atype, n_heads,
                 dropout, dropout_att, layer_norm_eps, ffn_nonlinear,
                 src_tgt_attention=True):
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
                                                         dropout=dropout_att)
            self.dropout1 = nn.Dropout(dropout)

        # attention for encoder stacks
        if src_tgt_attention:
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.src_attn = MultiheadAttentionMechanism(kdim=d_model,
                                                        qdim=d_model,
                                                        adim=d_model,
                                                        atype=atype,
                                                        n_heads=n_heads,
                                                        dropout=dropout_att)
            self.dropout2 = nn.Dropout(dropout)

        # feed-forward
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, d_model, dropout, ffn_nonlinear)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, ys, yy_mask=None, xs=None, xy_mask=None):
        """Transformer decoder layer definition.

        Args:
            ys (FloatTensor): `[B, L, d_model]`
            yy_mask (ByteTensor): `[B, L, L]`
            xs (FloatTensor): encoder outputs. `[B, T, d_model]`
            xy_mask (ByteTensor): `[B, T, L]`
        Returns:
            ys (FloatTensor): `[B, L, d_model]`
            yy_aw (FloatTensor)`[B, L, L]`
            xy_aw (FloatTensor): `[B, L, T]`

        """
        # self-attention
        if self.atype == "average":
            raise NotImplementedError
        else:
            self.self_attn.reset()
            residual = ys
            ys = self.norm1(ys)
            ys, yy_aw = self.self_attn(ys, ys, ys, mask=yy_mask)
            ys = self.dropout1(ys) + residual

        # attention for encoder stacks
        xy_aw = None
        if self.src_tgt_attention:
            self.src_attn.reset()
            residual = ys
            ys = self.norm2(ys)
            ys, xy_aw = self.src_attn(xs, xs, ys, mask=xy_mask)  # k/v/q
            ys = self.dropout2(ys) + residual

        # position-wise feed-forward
        residual = ys
        ys = self.norm3(ys)
        ys = self.feed_forward(ys)
        ys = self.dropout3(ys) + residual

        return ys, yy_aw, xy_aw
