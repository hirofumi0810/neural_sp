# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Relative multi-head attention layer for TransformerXL."""

import logging
import math
import numpy as np
import torch
import torch.nn as nn

from neural_sp.models.modules.mocha import headdrop


logger = logging.getLogger(__name__)


class RelativeMultiheadAttentionMechanism(nn.Module):
    """Relative multi-head attention layer for TransformerXL.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of attention space
        odim: (int) dimension of output
        n_heads (int): number of heads
        dropout (float): dropout probability for attention weights
        dropout_head (float): HeadDrop probability
        bias (bool): use bias term in linear layers
        param_init (str): parameter initialization method
        xl_like (bool): use TransformerXL like relative positional encoding.
            Otherwise, use relative positional encoding like Shaw et al. 2018
        clamp_len (int): maximum relative distance from each position

    """

    def __init__(self, kdim, qdim, adim, odim, n_heads, dropout, dropout_head=0.,
                 bias=False, param_init='', xl_like=False, clamp_len=-1):

        super().__init__()

        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(self.d_k)
        self.xl_like = xl_like
        self.clamp_len = clamp_len

        self.dropout_attn = nn.Dropout(p=dropout)
        self.dropout_head = dropout_head

        assert kdim == qdim
        # NOTE: relative attention is supprted for self-attention only
        self.w_key = nn.Linear(kdim, adim, bias=bias)
        self.w_value = nn.Linear(kdim, adim, bias=bias)
        self.w_query = nn.Linear(qdim, adim, bias=bias)
        self.w_out = nn.Linear(adim, odim, bias=bias)

        if xl_like:
            self.w_pos = nn.Linear(qdim, adim, bias=bias)  # W_{k,R}

        if param_init == 'xavier_uniform':
            self.reset_parameters(bias)
        else:
            logger.info('Parameter initialization is skipped.')

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        # NOTE: see https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_value.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.)
            nn.init.constant_(self.w_value.bias, 0.)
            nn.init.constant_(self.w_query.bias, 0.)

        nn.init.xavier_uniform_(self.w_out.weight)
        if bias:
            nn.init.constant_(self.w_out.bias, 0.)

        if self.xl_like:
            nn.init.xavier_uniform_(self.w_pos.weight)
            if bias:
                nn.init.constant_(self.w_pos.bias, 0.)

    def _rel_shift_v1(self, xs):
        """Calculate relative positional attention efficiently (old version).

        Args:
            xs (FloatTensor): `[B, qlen, klen, H]`
        Returns:
            xs_shifted (FloatTensor): `[B, qlen, klen, H]`

        """
        bs, qlen, klen, n_heads = xs.size()
        # `[qlen, klen, B, H]` -> `[B, qlen, klen, H]`
        xs = xs.permute(1, 2, 0, 3).contiguous().view(qlen, klen, bs * n_heads)

        zero_pad = xs.new_zeros((qlen, 1, bs * n_heads))
        xs_shifted = (torch.cat([zero_pad, xs], dim=1)
                      .view(klen + 1, qlen, bs * n_heads)[1:]
                      .view_as(xs))  # `[qlen, klen, B * H]`

        return xs_shifted.view(qlen, klen, bs, n_heads).permute(2, 0, 1, 3)

    def _rel_shift_v2(self, xs):
        """Calculate relative positional attention efficiently.

        Args:
            xs (FloatTensor): `[B, qlen, klen, H]`
        Returns:
            xs_shifted (FloatTensor): `[B, qlen, klen, H]`

        """
        bs, qlen, klen, n_heads = xs.size()
        xs = xs.permute(0, 3, 2, 1)  # `[B, H, klen, qlen]`

        idx = torch.arange(klen, device=xs.device)
        k_idx, q_idx = idx.unsqueeze(0), idx.unsqueeze(1)
        rel_pos_idx = torch.abs(k_idx - q_idx)
        # original postional encodings are generated with reversed order

        # for streaming inference
        if klen != qlen:
            rel_pos_idx = rel_pos_idx[:, :qlen]
            mask = xs.new_ones(qlen, klen, dtype=torch.uint8)
            mask = torch.tril(mask, diagonal=0).transpose(1, 0)
            rel_pos_idx[mask] *= -1
            rel_pos_idx = klen - qlen - rel_pos_idx
            rel_pos_idx[rel_pos_idx < 0] *= -1

        if self.clamp_len > 0:
            rel_pos_idx.clamp_(max=self.clamp_len)
        rel_pos_idx = rel_pos_idx.expand_as(xs)
        x_shift = torch.gather(xs, dim=2, index=rel_pos_idx)  # `[B, H, klen, qlen]`

        x_shift = x_shift.permute(0, 3, 2, 1)
        return x_shift

    def forward(self, key, query, pos_embs, mask, u_bias=None, v_bias=None):
        """Forward pass.

        Args:
            cat (FloatTensor): `[B, mlen+qlen, kdim]`
            mask (ByteTensor): `[B, qlen, mlen+qlen]`
            pos_embs (LongTensor): `[mlen+qlen, 1, d_model]`
            u_bias (nn.Parameter): `[H, d_k]`
            v_bias (nn.Parameter): `[H, d_k]`
        Returns:
            cv (FloatTensor): `[B, qlen, vdim]`
            aw (FloatTensor): `[B, H, qlen, mlen+qlen]`

        """
        bs, qlen = query.size()[:2]
        mlen = key.size(1) - qlen
        # NOTE: cat already includes memory, i.e., klen=mlen+qlen

        if mask is not None:
            mask = mask.unsqueeze(3).repeat([1, 1, 1, self.n_heads])
            assert mask.size() == (bs, qlen, mlen + qlen, self.n_heads), \
                (mask.size(), (bs, qlen, mlen + qlen, self.n_heads))

        k = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)  # `[B, mlen+qlen, H, d_k]`
        v = self.w_value(key).view(bs, -1, self.n_heads, self.d_k)  # `[B, mlen+qlen, H, d_k]`
        q = self.w_query(key[:, -qlen:]).view(bs, -1, self.n_heads, self.d_k)  # `[B, qlen, H, d_k]`

        if self.xl_like:
            _pos_embs = self.w_pos(pos_embs)
        else:
            _pos_embs = self.w_value(pos_embs)  # NOTE: this is not w_value
        _pos_embs = _pos_embs.view(-1, self.n_heads, self.d_k)  # `[mlen+qlen, H, d_k]`

        # content-based attention term: (a) + (c)
        if u_bias is not None:
            assert self.xl_like
            AC = torch.einsum("bihd,bjhd->bijh", (q + u_bias[None, None], k))  # `[B, qlen, mlen+qlen, H]`
        else:
            # A only accutually
            AC = torch.einsum("bihd,bjhd->bijh", (q, k))  # `[B, qlen, mlen+qlen, H]`

        # position-based attention term: (b) + (d)
        if v_bias is not None:
            assert self.xl_like
            BD = torch.einsum("bihd,jhd->bijh", (q + v_bias[None, None], _pos_embs))  # `[B, qlen, mlen+qlen, H]`
        else:
            # B only accutually
            BD = torch.einsum("bihd,jhd->bijh", (q, _pos_embs))  # `[B, qlen, mlen+qlen, H]`

        # Compute positional attention efficiently
        # BD = self._rel_shift_v1(BD)
        BD = self._rel_shift_v2(BD)

        # the attention is the sum of content-based and position-based attention
        e = (AC + BD) / self.scale  # `[B, qlen, mlen+qlen, H]`

        # Compute attention weights
        if mask is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=e.dtype).numpy().dtype).min)
            e = e.masked_fill_(mask == 0, NEG_INF)  # `[B, qlen, mlen+qlen, H]`
        aw = torch.softmax(e, dim=2)
        aw = self.dropout_attn(aw)  # `[B, qlen, mlen+qlen, H]`
        aw_masked = aw.clone()

        # mask out each head independently (HeadDrop)
        if self.dropout_head > 0 and self.training:
            aw_masked = aw_masked.permute(0, 3, 1, 2)
            aw_masked = headdrop(aw_masked, self.n_heads, self.dropout_head)  # `[B, H, qlen, klen]`
            aw_masked = aw_masked.permute(0, 2, 3, 1)

        cv = torch.einsum("bijh,bjhd->bihd", (aw, v))  # `[B, qlen, H, d_k]`
        cv = cv.contiguous().view(bs, -1, self.n_heads * self.d_k)  # `[B, qlen, H * d_k]`
        cv = self.w_out(cv)
        aw = aw.permute(0, 3, 1, 2)  # `[B, H, qlen, mlen+qlen]`

        return cv, aw
