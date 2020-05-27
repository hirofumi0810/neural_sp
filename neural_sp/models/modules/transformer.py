#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer blocks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import random
import torch
import torch.nn as nn

from neural_sp.models.modules.multihead_attention import MultiheadAttentionMechanism as MHA
from neural_sp.models.modules.positionwise_feed_forward import PositionwiseFeedForward as FFN
from neural_sp.models.modules.relative_multihead_attention import RelativeMultiheadAttentionMechanism as RelMHA

random.seed(1)

logger = logging.getLogger(__name__)


class TransformerEncoderBlock(nn.Module):
    """A single layer of the Transformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        atype (str): type of attention mechanism
        n_heads (int): number of heads for multi-head attention
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        dropout_layer (float): LayerDrop probabilities
        layer_norm_eps (float): epsilon parameter for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        param_init (str): parameter initialization method
        memory_transformer (bool): streaming TransformerXL encoder

    """

    def __init__(self, d_model, d_ff, atype, n_heads,
                 dropout, dropout_att, dropout_layer,
                 layer_norm_eps, ffn_activation, param_init,
                 memory_transformer=False):
        super(TransformerEncoderBlock, self).__init__()

        self.n_heads = n_heads
        self.memory_transformer = memory_transformer

        # self-attention
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        mha = RelMHA if memory_transformer else MHA
        self.self_attn = mha(kdim=d_model,
                             qdim=d_model,
                             adim=d_model,
                             n_heads=n_heads,
                             dropout=dropout_att,
                             param_init=param_init)

        # feed-forward
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation, param_init)

        self.dropout = nn.Dropout(dropout)
        self.dropout_layer = dropout_layer

    def forward(self, xs, xx_mask=None, pos_embs=None, memory=None, u=None, v=None):
        """Transformer encoder layer definition.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
            xx_mask (ByteTensor): `[B, T, T]`
            pos_embs (LongTensor): `[L, 1, d_model]`
            memory (FloatTensor): `[B, L_prev, d_model]`
            u (FloatTensor): global parameter for TransformerXL
            v (FloatTensor): global parameter for TransformerXL
        Returns:
            xs (FloatTensor): `[B, T, d_model]`
            xx_aws (FloatTensor): `[B, H, T, T]`

        """
        if self.dropout_layer > 0 and self.training and random.random() >= self.dropout_layer:
            return xs, None

        # self-attention
        residual = xs
        xs = self.norm1(xs)
        if self.memory_transformer:
            # memory Transformer w/ relative positional encoding
            xs, xx_aws = self.self_attn(xs, xs, memory, pos_embs, xx_mask, u, v)
        elif memory is not None:
            # memory Transformer w/o relative positional encoding
            xs_memory = torch.cat([memory, xs], dim=1)
            xs, xx_aws, _ = self.self_attn(xs_memory, xs_memory, xs, mask=xx_mask)  # k/v/q
        else:
            xs, xx_aws, _ = self.self_attn(xs, xs, xs, mask=xx_mask)  # k/v/q
        xs = self.dropout(xs) + residual

        # position-wise feed-forward
        residual = xs
        xs = self.norm2(xs)
        xs = self.feed_forward(xs)
        xs = self.dropout(xs) + residual

        return xs, xx_aws


class TransformerDecoderBlock(nn.Module):
    """A single layer of the Transformer decoder.

        Args:
            d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
            d_ff (int): hidden dimension of PositionwiseFeedForward
            atype (str): type of attention mechanism
            n_heads (int): number of heads for multi-head attention
            dropout (float): dropout probabilities for linear layers
            dropout_att (float): dropout probabilities for attention probabilities
            dropout_layer (float): LayerDrop probabilities
            dropout_head (float): HeadDrop probability
            layer_norm_eps (float): epsilon parameter for layer normalization
            ffn_activation (str): nonolinear function for PositionwiseFeedForward
            param_init (str): parameter initialization method
            src_tgt_attention (bool): if False, ignore source-target attention
            memory_transformer (bool): TransformerXL decoder
            mocha_chunk_size (int): chunk size for MoChA. -1 means infinite lookback.
            mocha_n_heads_mono (int): number of heads for monotonic attention
            mocha_n_heads_chunk (int): number of heads for chunkwise attention
            mocha_init_r (int):
            mocha_eps (float):
            mocha_std (float):
            mocha_no_denominator (bool):
            mocha_1dconv (bool):
            l0_penalty (float):
            l2_penalty (float):
            lm_fusion (bool):

    """

    def __init__(self, d_model, d_ff, atype, n_heads,
                 dropout, dropout_att, dropout_layer,
                 layer_norm_eps, ffn_activation, param_init,
                 src_tgt_attention=True, memory_transformer=False,
                 mocha_chunk_size=0, mocha_n_heads_mono=1, mocha_n_heads_chunk=1,
                 mocha_init_r=2, mocha_eps=1e-6, mocha_std=1.0,
                 mocha_no_denominator=False, mocha_1dconv=False,
                 dropout_head=0, lm_fusion=False):
        super(TransformerDecoderBlock, self).__init__()

        self.atype = atype
        self.n_heads = n_heads
        self.src_tgt_attention = src_tgt_attention
        self.memory_transformer = memory_transformer

        # self-attention
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        mha = RelMHA if memory_transformer else MHA
        self.self_attn = mha(kdim=d_model,
                             qdim=d_model,
                             adim=d_model,
                             n_heads=n_heads,
                             dropout=dropout_att,
                             param_init=param_init)

        # attention over encoder stacks
        if src_tgt_attention:
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            if 'mocha' in atype:
                self.n_heads = mocha_n_heads_mono
                from neural_sp.models.modules.mocha import MoChA
                self.src_attn = MoChA(kdim=d_model,
                                      qdim=d_model,
                                      adim=d_model,
                                      atype='scaled_dot',
                                      chunk_size=mocha_chunk_size,
                                      n_heads_mono=mocha_n_heads_mono,
                                      n_heads_chunk=mocha_n_heads_chunk,
                                      init_r=mocha_init_r,
                                      eps=mocha_eps,
                                      noise_std=mocha_std,
                                      no_denominator=mocha_no_denominator,
                                      conv1d=mocha_1dconv,
                                      dropout=dropout_att,
                                      dropout_head=dropout_head,
                                      param_init=param_init)
            else:
                self.src_attn = MHA(kdim=d_model,
                                    qdim=d_model,
                                    adim=d_model,
                                    n_heads=n_heads,
                                    dropout=dropout_att,
                                    param_init=param_init)

        # feed-forward
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation, param_init)

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_layer = dropout_layer

        # LM fusion
        self.lm_fusion = lm_fusion
        if lm_fusion:
            self.norm_lm = nn.LayerNorm(d_model, eps=layer_norm_eps)
            # NOTE: LM should be projected to d_model in advance
            self.linear_lm_feat = nn.Linear(d_model, d_model)
            self.linear_lm_gate = nn.Linear(d_model * 2, d_model)
            self.linear_lm_fusion = nn.Linear(d_model * 2, d_model)
            if 'attention' in lm_fusion:
                self.lm_attn = MHA(kdim=d_model,
                                   qdim=d_model,
                                   adim=d_model,
                                   n_heads=n_heads,
                                   dropout=dropout_att,
                                   param_init=param_init)

    def forward(self, ys, yy_mask, xs=None, xy_mask=None, cache=None,
                xy_aws_prev=None, mode='hard', lmout=None,
                pos_embs=None, memory=None, u=None, v=None,
                eps_wait=-1):
        """Transformer decoder forward pass.

        Args:
            ys (FloatTensor): `[B, L, d_model]`
            yy_mask (ByteTensor): `[B, L (query), L (key)]`
            xs (FloatTensor): encoder outputs. `[B, T, d_model]`
            xy_mask (ByteTensor): `[B, L, T]`
            cache (FloatTensor): `[B, L-1, d_model]`
            xy_aws_prev (FloatTensor): `[B, H, L, T]`
            mode (str):
            lmout (FloatTensor): `[B, L, d_model]`
            pos_embs (LongTensor): `[L, 1, d_model]`
            memory (FloatTensor): `[B, L_prev, d_model]`
            u (FloatTensor): global parameter for TransformerXL
            v (FloatTensor): global parameter for TransformerXL
            eps_wait (int):
        Returns:
            out (FloatTensor): `[B, L, d_model]`
            yy_aws (FloatTensor)`[B, H, L, L]`
            xy_aws (FloatTensor): `[B, H, L, T]`
            xy_aws_beta (FloatTensor): `[B, H, L, T]`

        """
        if self.dropout_layer > 0 and self.training and random.random() >= self.dropout_layer:
            xy_aws = None
            if self.src_tgt_attention:
                bs, qlen, klen = xy_mask.size()
                xy_aws = ys.new_zeros(bs, self.n_heads, qlen, klen)
            return ys, None, xy_aws, None, None

        residual = ys
        ys = self.norm1(ys)

        if cache is not None:
            ys_q = ys[:, -1:]
            residual = residual[:, -1:]
            yy_mask = yy_mask[:, -1:]
        else:
            ys_q = ys

        # self-attention
        yy_aws = None
        if self.memory_transformer:
            if cache is not None:
                pos_embs = pos_embs[-ys_q.size(1):]
            out, yy_aws = self.self_attn(ys, ys_q, memory, pos_embs, yy_mask, u, v)
        else:
            out, yy_aws, _ = self.self_attn(ys, ys, ys_q, mask=yy_mask)  # k/v/q
        out = self.dropout(out) + residual

        # attention over encoder stacks
        xy_aws, xy_aws_beta = None, None
        if self.src_tgt_attention:
            residual = out
            out = self.norm2(out)
            out, xy_aws, xy_aws_beta = self.src_attn(xs, xs, out, mask=xy_mask,  # k/v/q
                                                     aw_prev=xy_aws_prev, mode=mode,
                                                     eps_wait=eps_wait)
            out = self.dropout(out) + residual

        # LM integration
        yy_aws_lm = None
        if self.lm_fusion:
            residual = out
            out = self.norm_lm(out)
            lmout = self.linear_lm_feat(lmout)

            # attention over LM outputs
            if 'attention' in self.lm_fusion:
                out, yy_aws_lm, _ = self.lm_attn(lmout, lmout, out, mask=yy_mask)  # k/v/q

            gate = torch.sigmoid(self.linear_lm_gate(torch.cat([out, lmout], dim=-1)))
            gated_lmout = gate * lmout
            out = self.linear_lm_fusion(torch.cat([out, gated_lmout], dim=-1))
            out = self.dropout(out) + residual

        # position-wise feed-forward
        residual = out
        out = self.norm3(out)
        out = self.feed_forward(out)
        out = self.dropout(out) + residual

        if cache is not None:
            out = torch.cat([cache, out], dim=1)

        return out, yy_aws, xy_aws, xy_aws_beta, yy_aws_lm


class SyncBidirTransformerDecoderBlock(nn.Module):
    """A single layer of the synchronous bidirectional Transformer decoder.

        Args:
            d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
            d_ff (int): hidden dimension of PositionwiseFeedForward
            n_heads (int): number of heads for multi-head attention
            dropout (float): dropout probabilities for linear layers
            dropout_att (float): dropout probabilities for attention probabilities
            dropout_layer (float): LayerDrop probabilities
            layer_norm_eps (float): epsilon parameter for layer normalization
            ffn_activation (str): nonolinear function for PositionwiseFeedForward
            param_init (str): parameter initialization method

    """

    def __init__(self, d_model, d_ff, n_heads,
                 dropout, dropout_att, dropout_layer,
                 layer_norm_eps, ffn_activation, param_init):
        super(SyncBidirTransformerDecoderBlock, self).__init__()

        self.n_heads = n_heads

        # synchronous bidirectional attention
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        from neural_sp.models.modules.sync_bidir_multihead_attention import SyncBidirMultiheadAttentionMechanism as SyncBidirMHA
        self.self_attn = SyncBidirMHA(kdim=d_model,
                                      qdim=d_model,
                                      adim=d_model,
                                      n_heads=n_heads,
                                      dropout=dropout_att,
                                      param_init=param_init)

        # attention over encoder stacks
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.src_attn = MHA(kdim=d_model,
                            qdim=d_model,
                            adim=d_model,
                            n_heads=n_heads,
                            dropout=dropout_att,
                            param_init=param_init)

        # feed-forward
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation, param_init)

        self.dropout = nn.Dropout(p=dropout)
        # self.dropout_layer = dropout_layer

    def forward(self, ys, ys_bwd, yy_mask, identity_mask, xs, xy_mask,
                cache=None, cache_bwd=None):
        """Synchronous bidirectional Transformer decoder forward pass.

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
            tgt_mask=yy_mask, identity_mask=identity_mask)
        out = self.dropout(out) + residual
        out_bwd = self.dropout(out_bwd) + residual_bwd

        # attention over encoder stacks
        # fwd
        residual = out
        out = self.norm2(out)
        out, xy_aws, _ = self.src_attn(xs, xs, out, mask=xy_mask)  # k/v/q
        out = self.dropout(out) + residual
        # bwd
        residual_bwd = out_bwd
        out_bwd = self.norm2(out_bwd)
        out_bwd, xy_aws_bwd, _ = self.src_attn(xs, xs, out_bwd, mask=xy_mask)  # k/v/q
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
