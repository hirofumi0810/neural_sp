#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utilities for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.modules.multihead_attention import MultiheadAttentionMechanism


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer.

    Args:
        d_model (int):
        dropout (float):
        pe_type (str):
        max_len (int):

    """

    def __init__(self, d_model, dropout, pe_type, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.pe_type = pe_type

        if pe_type:
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
        if not self.pe_type:
            return xs

        if self.pe_type == 'add':
            xs = xs + self.pe[:, :xs.size(1)]
        elif self.pe_type == 'concat':
            xs = torch.cat([xs, self.pe[:, :xs.size(1)]], dim=-1)
        else:
            raise NotImplementedError(self.pe_type)
        return self.dropout(xs)


class SublayerConnection(nn.Module):
    """A residual connection with dropout and layer normalization.

        input -> layer norm -> sublayer -> dropout -> output ->
          |                                         |
          -------------------------------------------
    Args:
        d_model (int):
        dropout (float):
        layer_norm_eps (float): epsilon parameter for layer normalization
        stocastic (float):
    """

    def __init__(self, d_model, dropout, layer_norm_eps=1e-6, stocastic=0):
        super(SublayerConnection, self).__init__()

        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.stocastic = stocastic

    def forward(self, xs, sublayer):
        xs_norm = self.norm(xs)
        output = sublayer(xs_norm)

        # NOTE: output may be tuple paired with attention weights
        if isinstance(output, tuple):
            xs_norm, aws = output
        else:
            xs_norm = output
        xs_norm = self.dropout(xs_norm) + xs

        if isinstance(output, tuple):
            return xs_norm, aws
        else:
            return xs_norm


class PositionwiseFeedForward(nn.Module):
    """Positionwise fully-connected feed-forward neural network.

    Args:
        d_model (int):
        d_ff (int):
        dropout (float):

    """

    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xs):
        return self.w_2(self.dropout(F.relu(self.w_1(xs))))


class TransformerEncoderBlock(nn.Module):
    """A single layer of the transformer encoder.

    Args:
        d_model (int): dimension of keys/values/queries in
                   MultiheadAttentionMechanism, also the input size of
                   the first-layer of the PositionwiseFeedForward
        d_ff (int): second-layer of the PositionwiseFeedForward
        attn_type (str):
        attn_n_heads (int): number of heads for multi-head attention
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        layer_norm_eps (float):

    """

    def __init__(self,
                 d_model,
                 d_ff,
                 attn_type,
                 attn_n_heads,
                 dropout,
                 dropout_att,
                 layer_norm_eps):
        super(TransformerEncoderBlock, self).__init__()

        # self-attention
        self.self_attn = MultiheadAttentionMechanism(key_dim=d_model,
                                                     query_dim=d_model,
                                                     attn_type=attn_type,
                                                     attn_dim=d_model,
                                                     n_heads=attn_n_heads,
                                                     dropout=dropout_att)
        self.add_norm_self_attn = SublayerConnection(d_model, dropout, layer_norm_eps)

        # feed-forward
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm_ff = SublayerConnection(d_model, dropout, layer_norm_eps)

    def forward(self, xs, xlens, cache=False):
        """Transformer encoder layer definition.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
            xlens (IntTensor): `[B]`
            cache (bool):
        Returns:
            xs (FloatTensor): `[B, T, d_model]`
            xx_aws (FloatTensor): `[B, T, T]`

        """
        # self-attention
        if not cache:
            self.self_attn.reset()
        xs, xx_aws = self.add_norm_self_attn(xs, sublayer=lambda xs: self.self_attn(
            key=xs, klens=xlens, value=xs, query=xs))

        # position-wise feed-forward
        xs = self.add_norm_ff(xs, sublayer=lambda xs: self.ff(xs))

        return xs, xx_aws


class TransformerDecoderBlock(nn.Module):
    """A single layer of the transformer decoder.

        Args:
            d_model (int): dimension of keys/values/queries in
                           MultiheadAttentionMechanism, also the input size of
                           the first-layer of the PositionwiseFeedForward
            d_ff (int): second-layer of the PositionwiseFeedForward
            attn_type (str):
            attn_n_heads (int): number of heads for multi-head attention
            dropout (float): dropout probabilities for linear layers
            dropout_att (float): dropout probabilities for attention probabilities
            attn_type (str): type of self-attention, scaled_dot or average
            layer_norm_eps (float):
            src_attention (bool): if False, ignore source-target attention

    """

    def __init__(self,
                 d_model,
                 d_ff,
                 attn_type,
                 attn_n_heads,
                 dropout,
                 dropout_att,
                 layer_norm_eps,
                 src_attention=True):
        super(TransformerDecoderBlock, self).__init__()

        self.attn_type = attn_type
        self.src_attention = src_attention

        # self-attention
        if attn_type == "average":
            raise NotImplementedError
        else:
            self.self_attn = MultiheadAttentionMechanism(key_dim=d_model,
                                                         query_dim=d_model,
                                                         attn_type=attn_type,
                                                         attn_dim=d_model,
                                                         n_heads=attn_n_heads,
                                                         dropout=dropout_att)
        self.add_norm_self_attn = SublayerConnection(d_model, dropout, layer_norm_eps)

        # attention for encoder stacks
        if src_attention:
            self.src_attn = MultiheadAttentionMechanism(key_dim=d_model,
                                                        query_dim=d_model,
                                                        attn_type=attn_type,
                                                        attn_dim=d_model,
                                                        n_heads=attn_n_heads,
                                                        dropout=dropout_att)
            self.add_norm_src_attn = SublayerConnection(d_model, dropout, layer_norm_eps)

        # feed-forward
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm_ff = SublayerConnection(d_model, dropout, layer_norm_eps)

    def forward(self, ys, ylens, xs=None, xlens=None):
        """Transformer decoder layer definition.

        Args:
            xs (FloatTensor): encoder outputs. `[B, T, d_model]`
            xlens (IntTensor): `[B]`
            ys (FloatTensor): `[B, L, d_model]`
            ylens (IntTensor): `[B]`
        Returns:
            ys (FloatTensor): `[B, L, d_model]`
            yy_aw (FloatTensor)`[B, L, L]`
            xy_aw (FloatTensor): `[B, L, T]`

        """
        # self-attention
        if self.attn_type == "average":
            raise NotImplementedError
        else:
            self.self_attn.reset()
            ys, yy_aw = self.add_norm_self_attn(ys, lambda ys: self.self_attn(
                key=ys, klens=ylens, value=ys, query=ys, diagonal=True))

        # attention for encoder stacks
        if self.src_attention:
            self.src_attn.reset()
            ys, xy_aw = self.add_norm_src_attn(ys, lambda ys: self.src_attn(
                key=xs, klens=xlens, value=xs, query=ys, qlens=ylens))
        else:
            xy_aw = None

        # position-wise feed-forward
        ys = self.add_norm_ff(ys, lambda ys: self.ff(ys))

        return ys, yy_aw, xy_aw
