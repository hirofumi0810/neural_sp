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
import random
import torch
import torch.nn as nn

from neural_sp.models.modules.causal_conv import CausalConv1d
from neural_sp.models.modules.gelu import gelu, gelu_accurate
from neural_sp.models.modules.glu import LinearGLUBlock
from neural_sp.models.modules.mocha import MoChA
from neural_sp.models.modules.multihead_attention import MultiheadAttentionMechanism
from neural_sp.models.modules.sync_bidir_multihead_attention import SyncBidirMultiheadAttentionMechanism

random.seed(1)

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
                 conv_kernel_size=3, layer_norm_eps=1e-12):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.pe_type = pe_type
        self.scale = math.sqrt(self.d_model)

        if '1dconv' in pe_type:
            causal_conv1d = CausalConv1d(in_channels=d_model,
                                         out_channels=d_model,
                                         kernel_size=conv_kernel_size,
                                         stride=1)
            layers = []
            conv_nlayers = int(pe_type.replace('1dconv', '')[0])
            for l in range(conv_nlayers):
                layers.append(copy.deepcopy(causal_conv1d))
                layers.append(nn.LayerNorm(d_model, eps=layer_norm_eps))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
            self.pe = nn.Sequential(*layers)
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
            xs = self.dropout(xs)
        elif self.pe_type == 'concat':
            xs = torch.cat([xs, self.pe[:, :xs.size(1)]], dim=-1)
            xs = self.dropout(xs)
        elif '1dconv' in self.pe_type:
            xs = self.pe(xs)
        else:
            raise NotImplementedError(self.pe_type)
        return xs


class PositionwiseFeedForward(nn.Module):
    """Positionwise fully-connected feed-forward neural network.

    Args:
        d_in (int): input dimension (equal to d_model)
        d_ff (int): dimention of PositionwiseFeedForward
        d_out (int): output dimension of PositionwiseFeedForward
        dropout (float): dropout probability
        activation: non-linear function
        param_init (str): parameter initialization method

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
        atype (str): type of attention mechanism
        n_heads (int): number of heads for multi-head attention
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        dropout_residual (float): dropout probabilities for residual connections
        layer_norm_eps (float): epsilon parameter for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        param_init (str): parameter initialization method

    """

    def __init__(self, d_model, d_ff, atype, n_heads,
                 dropout, dropout_att, dropout_residual,
                 layer_norm_eps, ffn_activation, param_init):
        super(TransformerEncoderBlock, self).__init__()

        self.n_heads = n_heads

        # self-attention
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = MultiheadAttentionMechanism(kdim=d_model,
                                                     qdim=d_model,
                                                     adim=d_model,
                                                     atype='scaled_dot',
                                                     n_heads=n_heads,
                                                     dropout=dropout_att,
                                                     param_init=param_init)

        # feed-forward
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, d_model, dropout, ffn_activation, param_init)

        self.dropout = nn.Dropout(dropout)
        self.death_rate = dropout_residual

    def forward(self, xs, xx_mask=None):
        """Transformer encoder layer definition.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
            xx_mask (ByteTensor): `[B, T, T]`
        Returns:
            xs (FloatTensor): `[B, T, d_model]`
            xx_aws (FloatTensor): `[B, H, T, T]`

        """
        if self.death_rate > 0 and self.training and random.random() >= self.death_rate:
            return xs, None

        # self-attention
        residual = xs
        xs = self.norm1(xs)
        xs, xx_aws, _ = self.self_attn(xs, xs, xs, mask=xx_mask, cache=False)
        if self.death_rate > 0 and self.training:
            xs = xs / (1 - self.death_rate)
        xs = self.dropout(xs) + residual

        # position-wise feed-forward
        residual = xs
        xs = self.norm2(xs)
        xs = self.feed_forward(xs)
        if self.death_rate > 0 and self.training:
            xs = xs / (1 - self.death_rate)
        xs = self.dropout(xs) + residual

        return xs, xx_aws


class TransformerDecoderBlock(nn.Module):
    """A single layer of the transformer decoder.

        Args:
            d_model (int): dimension of MultiheadAttentionMechanism
            d_ff (int): dimention of PositionwiseFeedForward
            atype (str): type of attention mechanism
            n_heads (int): number of heads for multi-head attention
            dropout (float): dropout probabilities for linear layers
            dropout_att (float): dropout probabilities for attention probabilities
            dropout_residual (float): dropout probabilities for residual connections
            dropout_head (float): dropout probabilities for heads
            layer_norm_eps (float): epsilon parameter for layer normalization
            ffn_activation (str): nonolinear function for PositionwiseFeedForward
            param_init (str): parameter initialization method
            src_tgt_attention (bool): if False, ignore source-target attention
            mocha_chunk_size (int): chunk size for MoChA. -1 means infinite lookback.
            mocha_n_heads_mono (int): number of heads for monotonic attention
            mocha_n_heads_chunk (int): number of heads for chunkwise attention

    """

    def __init__(self, d_model, d_ff, atype, n_heads,
                 dropout, dropout_att, dropout_residual, dropout_head,
                 layer_norm_eps, ffn_activation, param_init,
                 src_tgt_attention=True, mocha_chunk_size=0,
                 mocha_n_heads_mono=1, mocha_n_heads_chunk=1,
                 lm_fusion=False):
        super(TransformerDecoderBlock, self).__init__()

        self.atype = atype
        self.n_heads = n_heads
        self.src_tgt_attention = src_tgt_attention

        # self-attention
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        if atype == 'average':
            raise NotImplementedError(atype)
        else:
            self.self_attn = MultiheadAttentionMechanism(kdim=d_model,
                                                         qdim=d_model,
                                                         adim=d_model,
                                                         atype='scaled_dot',
                                                         n_heads=n_heads,
                                                         dropout=dropout_att,
                                                         param_init=param_init)

        # attention over encoder stacks
        if src_tgt_attention:
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            if 'mocha' in atype:
                self.n_heads = mocha_n_heads_mono
                self.src_attn = MoChA(kdim=d_model,
                                      qdim=d_model,
                                      adim=d_model,
                                      atype='scaled_dot',
                                      chunk_size=mocha_chunk_size,
                                      n_heads_mono=mocha_n_heads_mono,
                                      n_heads_chunk=mocha_n_heads_chunk,
                                      dropout=dropout_att,
                                      dropout_head=dropout_head,
                                      param_init=param_init,
                                      simple=atype == 'mocha_simple',
                                      simple_v2=atype == 'mocha_simple_v2')
            else:
                self.src_attn = MultiheadAttentionMechanism(kdim=d_model,
                                                            qdim=d_model,
                                                            adim=d_model,
                                                            atype='scaled_dot',
                                                            n_heads=n_heads,
                                                            dropout=dropout_att,
                                                            param_init=param_init)

        # feed-forward
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, d_model, dropout, ffn_activation, param_init)

        self.dropout = nn.Dropout(p=dropout)
        self.death_rate = dropout_residual

        self.lm_fusion = lm_fusion
        if lm_fusion:
            self.norm_lm = nn.LayerNorm(d_model, eps=layer_norm_eps)
            # NOTE: LM should be projected to d_model in advance
            self.linear_lm_feat = nn.Linear(d_model, d_model)
            self.linear_lm_gate = nn.Linear(d_model * 2, d_model)
            self.linear_lm_fusion = nn.Linear(d_model * 2, d_model)
            if 'attention' in lm_fusion:
                self.lm_attn = MultiheadAttentionMechanism(kdim=d_model,
                                                           qdim=d_model,
                                                           adim=d_model,
                                                           atype='scaled_dot',
                                                           n_heads=n_heads,
                                                           dropout=dropout_att,
                                                           param_init=param_init)

    def forward(self, ys, yy_mask, xs=None, xy_mask=None, cache=None,
                xy_aws_prev=None, mode='hard', lmout=None):
        """Transformer decoder forward pass.

        Args:
            ys (FloatTensor): `[B, L, d_model]`
            yy_mask (ByteTensor): `[B, L, L]`
            xs (FloatTensor): encoder outputs. `[B, T, d_model]`
            xy_mask (ByteTensor): `[B, L, T]`
            cache (FloatTensor): `[B, L-1, d_model]`
            xy_aws_prev (FloatTensor): `[B, H, L, T]`
            mode (str):
            lmout (FloatTensor): `[B, L, d_model]`
        Returns:
            out (FloatTensor): `[B, L, d_model]`
            yy_aws (FloatTensor)`[B, H, L, L]`
            xy_aws (FloatTensor): `[B, H, L, T]`
            xy_aws_beta (FloatTensor): `[B, H, L, T]`

        """
        if self.death_rate > 0 and self.training and random.random() >= self.death_rate:
            xy_aws = None
            if self.src_tgt_attention:
                bs, qlen, klen = xy_mask.size()
                xy_aws = ys.new_zeros(bs, self.n_heads, qlen, klen)
            return ys, None, xy_aws, None

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
            out, yy_aws, _ = self.self_attn(
                ys, ys, ys_q, mask=yy_mask, cache=False)  # k/v/q
            if self.death_rate > 0 and self.training:
                out = out / (1 - self.death_rate)
            out = self.dropout(out) + residual

        # attention over encoder stacks
        xy_aws, xy_aws_beta = None, None
        if self.src_tgt_attention:
            residual = out
            out = self.norm2(out)
            out, xy_aws, xy_aws_beta = self.src_attn(
                xs, xs, out, mask=xy_mask, cache=False,  # k/v/q
                aw_prev=xy_aws_prev, mode=mode,
                n_tokens=yy_mask[:, -1, :].sum(1).float() if yy_mask is not None else None)
            if self.death_rate > 0 and self.training:
                out = out / (1 - self.death_rate)
            out = self.dropout(out) + residual

        # LM integration
        yy_aws_lm = None
        if self.lm_fusion:
            residual = out
            out = self.norm_lm(out)
            lmout = self.linear_lm_feat(lmout)

            # attention over LM outputs
            if 'attention' in self.lm_fusion:
                out, yy_aws_lm, _ = self.lm_attn(
                    lmout, lmout, out, mask=yy_mask, cache=False)  # k/v/q

            gate = torch.sigmoid(self.linear_lm_gate(torch.cat([out, lmout], dim=-1)))
            gated_lmout = gate * lmout
            out = self.linear_lm_fusion(torch.cat([out, gated_lmout], dim=-1))

            if self.death_rate > 0 and self.training:
                out = out / (1 - self.death_rate)
            out = self.dropout(out) + residual

        # position-wise feed-forward
        residual = out
        out = self.norm3(out)
        out = self.feed_forward(out)
        if self.death_rate > 0 and self.training:
            out = out / (1 - self.death_rate)
        out = self.dropout(out) + residual

        if cache is not None:
            out = torch.cat([cache, out], dim=1)

        return out, yy_aws, xy_aws, xy_aws_beta, yy_aws_lm


class SyncBidirTransformerDecoderBlock(nn.Module):
    """A single layer of the synchronous bidirectional transformer decoder.

        Args:
            d_model (int): dimension of MultiheadAttentionMechanism
            d_ff (int): dimention of PositionwiseFeedForward
            n_heads (int): number of heads for multi-head attention
            dropout (float): dropout probabilities for linear layers
            dropout_att (float): dropout probabilities for attention probabilities
            dropout_residual (float): dropout probabilities for residual connections
            layer_norm_eps (float): epsilon parameter for layer normalization
            ffn_activation (str): nonolinear function for PositionwiseFeedForward
            param_init (str): parameter initialization method

    """

    def __init__(self, d_model, d_ff, n_heads,
                 dropout, dropout_att, dropout_residual,
                 layer_norm_eps, ffn_activation, param_init):
        super(SyncBidirTransformerDecoderBlock, self).__init__()

        self.n_heads = n_heads

        # synchronous bidirectional attention
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = SyncBidirMultiheadAttentionMechanism(kdim=d_model,
                                                              qdim=d_model,
                                                              adim=d_model,
                                                              atype='scaled_dot',
                                                              n_heads=n_heads,
                                                              dropout=dropout_att,
                                                              param_init=param_init)

        # attention over encoder stacks
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.src_attn = MultiheadAttentionMechanism(kdim=d_model,
                                                    qdim=d_model,
                                                    adim=d_model,
                                                    atype='scaled_dot',
                                                    n_heads=n_heads,
                                                    dropout=dropout_att,
                                                    param_init=param_init)

        # feed-forward
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, d_model, dropout, ffn_activation, param_init)

        self.dropout = nn.Dropout(p=dropout)
        self.death_rate = dropout_residual

    def forward(self, ys, ys_bwd, yy_mask, identity_mask, xs, xy_mask,
                cache=None, cache_bwd=None):
        """Synchronous bidirectional transformer decoder forward pass.

        Args:
            ys (FloatTensor): `[B, L, d_model]`
            ys_bwd (FloatTensor): `[B, L, d_model]`
            yy_mask (ByteTensor): `[B, L, L]`
            identity_mask (ByteTensor): `[B, L, L]`
            xs (FloatTensor): encoder outputs. `[B, T, d_model]`
            xy_mask (ByteTensor): `[B, L, T]`
            cache (FloatTensor): `[B, L-1, d_model]`
            cache_bwd (FloatTensor): `[B, L-1, d_model]`
        Returns:
            out (FloatTensor): `[B, L, d_model]`
            yy_aws_h (FloatTensor)`[B, L, L]`
            yy_aws_f (FloatTensor)`[B, L, L]`
            yy_aws_bwd_h (FloatTensor)`[B, L, L]`
            yy_aws_bwd_f (FloatTensor)`[B, L, L]`
            xy_aws (FloatTensor): `[B, L, T]`
            xy_aws_bwd (FloatTensor): `[B, L, T]`

        """
        residual = ys
        residual_bwd = ys_bwd
        ys = self.norm1(ys)
        ys_bwd = self.norm1(ys_bwd)

        if cache is not None:
            assert cache_bwd is not None
            ys_q = ys[:, -1:]
            ys_bwd_q = ys_bwd[:, -1:]
            residual = residual[:, -1:]
            residual_bwd = residual_bwd[:, -1:]
            yy_mask = yy_mask[:, -1:]
        else:
            ys_q = ys
            ys_bwd_q = ys_bwd

        # synchronous bidirectional attention
        out, out_bwd, yy_aws_h, yy_aws_f, yy_aws_bwd_h, yy_aws_bwd_f = self.self_attn(
            ys, ys, ys_q,  # k/v/q
            ys_bwd, ys_bwd, ys_bwd_q,  # k/v/q
            tgt_mask=yy_mask, identity_mask=identity_mask, cache=False)
        out = self.dropout(out) + residual
        out_bwd = self.dropout(out_bwd) + residual_bwd

        # attention over encoder stacks
        # fwd
        residual = out
        out = self.norm2(out)
        out, xy_aws, _ = self.src_attn(xs, xs, out, mask=xy_mask, cache=False)  # k/v/q
        out = self.dropout(out) + residual
        # bwd
        residual_bwd = out_bwd
        out_bwd = self.norm2(out_bwd)
        out_bwd, xy_aws_bwd, _ = self.src_attn(xs, xs, out_bwd, mask=xy_mask, cache=False)  # k/v/q
        out_bwd = self.dropout(out_bwd) + residual_bwd

        # position-wise feed-forward
        # fwd
        residual = out
        out = self.norm3(out)
        out = self.feed_forward(out)
        out = self.dropout(out) + residual
        # bwd
        residual_bwd = out_bwd
        out_bwd = self.norm3(out_bwd)
        out_bwd = self.feed_forward(out_bwd)
        out_bwd = self.dropout(out_bwd) + residual_bwd

        if cache is not None:
            out = torch.cat([cache, out], dim=1)
            out_bwd = torch.cat([cache_bwd, out_bwd], dim=1)

        return out, out_bwd, yy_aws_h, yy_aws_f, yy_aws_bwd_h, yy_aws_bwd_f, xy_aws, xy_aws_bwd
