# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer blocks."""

import logging
import random
import torch
import torch.nn as nn

from neural_sp.models.modules.mocha import MoChA
from neural_sp.models.modules.multihead_attention import MultiheadAttentionMechanism as MHA
from neural_sp.models.modules.positionwise_feed_forward import PositionwiseFeedForward as FFN
from neural_sp.models.modules.relative_multihead_attention import RelativeMultiheadAttentionMechanism as RelMHA

random.seed(1)

logger = logging.getLogger(__name__)


class TransformerDecoderBlock(nn.Module):
    """A single layer of the Transformer decoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        atype (str): type of attention mechanism
        n_heads (int): number of heads for multi-head attention
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention probabilities
        dropout_layer (float): LayerDrop probability
        dropout_head (float): HeadDrop probability
        layer_norm_eps (float): epsilon parameter for layer normalization
        ffn_activation (str): nonlinear function for PositionwiseFeedForward
        param_init (str): parameter initialization method
        src_tgt_attention (bool): use source-target attention
        memory_transformer (bool): TransformerXL decoder
        mma_chunk_size (int): chunk size for chunkwise attention. -1 means infinite lookback.
        mma_n_heads_mono (int): number of MMA head
        mma_n_heads_chunk (int): number of hard chunkwise attention head
        mma_init_r (int): initial bias value for MMA
        mma_eps (float): epsilon value for MMA
        mma_std (float): standard deviation of Gaussian noise for MMA
        mma_no_denominator (bool): remove denominator in MMA
        mma_1dconv (bool): 1dconv for MMA
        share_chunkwise_attention (bool): share chunkwise attention in the same layer of MMA
        lm_fusion (str): type of LM fusion
        ffn_bottleneck_dim (int): bottleneck dimension for the light-weight FFN layer

    """

    def __init__(self, d_model, d_ff, atype, n_heads,
                 dropout, dropout_att, dropout_layer,
                 layer_norm_eps, ffn_activation, param_init,
                 src_tgt_attention=True, memory_transformer=False,
                 mma_chunk_size=0, mma_n_heads_mono=1, mma_n_heads_chunk=1,
                 mma_init_r=2, mma_eps=1e-6, mma_std=1.0,
                 mma_no_denominator=False, mma_1dconv=False,
                 dropout_head=0, share_chunkwise_attention=False,
                 lm_fusion='', ffn_bottleneck_dim=0):

        super().__init__()

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
                             odim=d_model,
                             n_heads=n_heads,
                             dropout=dropout_att,
                             dropout_head=dropout_head,
                             param_init=param_init,
                             xl_like=memory_transformer)

        # attention over encoder stacks
        if src_tgt_attention:
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            if 'mocha' in atype:
                self.n_heads = mma_n_heads_mono
                self.src_attn = MoChA(kdim=d_model,
                                      qdim=d_model,
                                      adim=d_model,
                                      odim=d_model,
                                      atype='scaled_dot',
                                      chunk_size=mma_chunk_size,
                                      n_heads_mono=mma_n_heads_mono,
                                      n_heads_chunk=mma_n_heads_chunk,
                                      init_r=mma_init_r,
                                      eps=mma_eps,
                                      noise_std=mma_std,
                                      no_denominator=mma_no_denominator,
                                      conv1d=mma_1dconv,
                                      dropout=dropout_att,
                                      dropout_head=dropout_head,
                                      param_init=param_init,
                                      share_chunkwise_attention=share_chunkwise_attention)
            else:
                self.src_attn = MHA(kdim=d_model,
                                    qdim=d_model,
                                    adim=d_model,
                                    odim=d_model,
                                    n_heads=n_heads,
                                    dropout=dropout_att,
                                    param_init=param_init)
        else:
            self.src_attn = None

        # position-wise feed-forward
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation, param_init,
                                ffn_bottleneck_dim)

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
                                   odim=d_model,
                                   n_heads=n_heads,
                                   dropout=dropout_att,
                                   param_init=param_init)

        self.reset_visualization()

    @property
    def yy_aws(self):
        return self._yy_aws

    @property
    def xy_aws(self):
        return self._xy_aws

    @property
    def xy_aws_beta(self):
        return self._xy_aws_beta

    @property
    def xy_aws_p_choose(self):
        return self._xy_aws_p_choose

    @property
    def yy_aws_lm(self):
        return self._yy_aws_lm

    def reset_visualization(self):
        self._yy_aws = None
        self._xy_aws = None
        self._xy_aws_beta = None
        self._xy_aws_p_choose = None
        self._yy_aws_lm = None

    def reset(self):
        if self.src_attn is not None:
            self.src_attn.reset()

    def forward(self, ys, yy_mask, xs=None, xy_mask=None, cache=None,
                xy_aws_prev=None,
                mode='hard', eps_wait=-1, lmout=None,
                pos_embs=None, memory=None, u_bias=None, v_bias=None):
        """Transformer decoder forward pass.

        Args:
            ys (FloatTensor): `[B, L, d_model]`
            yy_mask (ByteTensor): `[B, L (query), L (key)]`
            xs (FloatTensor): encoder outputs. `[B, T, d_model]`
            xy_mask (ByteTensor): `[B, L, T]`
            cache (FloatTensor): `[B, L-1, d_model]`
            xy_aws_prev (FloatTensor): `[B, H, L, T]`
            mode (str): decoding mode for MMA
            eps_wait (int): wait time delay for head-synchronous decoding in MMA
            lmout (FloatTensor): `[B, L, d_model]`
            pos_embs (LongTensor): `[L, 1, d_model]`
            memory (FloatTensor): `[B, L_prev, d_model]`
            u_bias (FloatTensor): global parameter for TransformerXL
            v_bias (FloatTensor): global parameter for TransformerXL
        Returns:
            out (FloatTensor): `[B, L, d_model]`

        """
        self.reset_visualization()

        # LayerDrop
        if self.dropout_layer > 0 and self.training and random.random() < self.dropout_layer:
            return ys

        residual = ys
        if self.memory_transformer:
            if cache is not None:
                pos_embs = pos_embs[-ys.size(1):]
            if memory is not None and memory.dim() > 1:
                cat = self.norm1(torch.cat([memory, ys], dim=1))
                ys = cat[:, memory.size(1):]
            else:
                ys = self.norm1(ys)
                cat = ys
        else:
            ys = self.norm1(ys)  # pre-norm

        if cache is not None:
            ys_q = ys[:, -1:]
            residual = residual[:, -1:]
            yy_mask = yy_mask[:, -1:]
        else:
            ys_q = ys

        # self-attention
        if self.memory_transformer:
            out, self._yy_aws = self.self_attn(cat, ys_q, pos_embs, yy_mask, u_bias, v_bias)  # k/q/m
        else:
            out, self._yy_aws = self.self_attn(ys, ys, ys_q, mask=yy_mask)[:2]  # k/v/q
        out = self.dropout(out) + residual

        # attention over encoder stacks
        if self.src_tgt_attention:
            residual = out
            out = self.norm2(out)
            out, self._xy_aws, attn_state = self.src_attn(
                xs, xs, out, mask=xy_mask,  # k/v/q
                aw_prev=xy_aws_prev, mode=mode, eps_wait=eps_wait)
            out = self.dropout(out) + residual

            if attn_state.get('beta', None) is not None:
                self._xy_aws_beta = attn_state['beta']
            if attn_state.get('p_choose', None) is not None:
                self._xy_aws_p_choose = attn_state['p_choose']

        # LM integration
        if self.lm_fusion:
            residual = out
            out = self.norm_lm(out)
            lmout = self.linear_lm_feat(lmout)

            # attention over LM outputs
            if 'attention' in self.lm_fusion:
                out, self._yy_aws_lm, _ = self.lm_attn(lmout, lmout, out, mask=yy_mask)  # k/v/q

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

        return out


class SyncBidirTransformerDecoderBlock(nn.Module):
    """A single layer of the synchronous bidirectional Transformer decoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention probabilities
        dropout_layer (float): LayerDrop probability
        layer_norm_eps (float): epsilon parameter for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        param_init (str): parameter initialization method

    """

    def __init__(self, d_model, d_ff, n_heads,
                 dropout, dropout_att, dropout_layer,
                 layer_norm_eps, ffn_activation, param_init):

        super().__init__()

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
                            odim=d_model,
                            n_heads=n_heads,
                            dropout=dropout_att,
                            param_init=param_init)

        # position-wise feed-forward
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation, param_init)

        self.dropout = nn.Dropout(p=dropout)
        # self.dropout_layer = dropout_layer

        self.reset_visualization()

    def reset_visualization(self):
        self._yy_aws_h, self.yy_aws_f = None, None
        self._yy_aws_bwd_h, self._yy_aws_bwd_f = None, None
        self._xy_aws, self._xy_aws_bwd = None, None

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

        """
        self.reset_visualization()

        residual = ys
        residual_bwd = ys_bwd
        ys = self.norm1(ys)  # pre-norm
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
        out, out_bwd, self._yy_aws_h, self.yy_aws_f, self._yy_aws_bwd_h, self._yy_aws_bwd_f = self.self_attn(
            ys, ys, ys_q,  # k/v/q
            ys_bwd, ys_bwd, ys_bwd_q,  # k/v/q
            tgt_mask=yy_mask, identity_mask=identity_mask)
        out = self.dropout(out) + residual
        out_bwd = self.dropout(out_bwd) + residual_bwd

        # attention over encoder stacks
        # fwd
        residual = out
        out = self.norm2(out)
        out, self._xy_aws, _ = self.src_attn(xs, xs, out, mask=xy_mask)  # k/v/q
        out = self.dropout(out) + residual
        # bwd
        residual_bwd = out_bwd
        out_bwd = self.norm2(out_bwd)
        out_bwd, self._xy_aws_bwd, _ = self.src_attn(xs, xs, out_bwd, mask=xy_mask)  # k/v/q
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

        return out, out_bwd
