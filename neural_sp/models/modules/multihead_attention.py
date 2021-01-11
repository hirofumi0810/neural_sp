# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-head attention (MHA) layer."""

import logging
import math
import numpy as np
import torch
import torch.nn as nn

from neural_sp.models.modules.mocha import headdrop

logger = logging.getLogger(__name__)


class MultiheadAttentionMechanism(nn.Module):
    """Multi-headed attention (MHA) layer.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of attention space
        odim: (int) dimension of output
        n_heads (int): number of heads
        dropout (float): dropout probability for attention weights
        dropout_head (float): HeadDrop probability
        atype (str): type of attention mechanism
        bias (bool): use bias term in linear layers
        param_init (str): parameter initialization method
        xl_like: dummy argument for compatibility with relative MHA
        clamp_len: dummy

    """

    def __init__(self, kdim, qdim, adim, odim, n_heads, dropout, dropout_head=0.,
                 atype='scaled_dot', bias=True, param_init='',
                 xl_like=False, clamp_len=-1):

        super().__init__()

        self.atype = atype
        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(self.d_k)
        self.reset()

        self.dropout_attn = nn.Dropout(p=dropout)
        self.dropout_head = dropout_head

        if atype == 'scaled_dot':
            # for Transformer
            self.w_key = nn.Linear(kdim, adim, bias=bias)
            self.w_value = nn.Linear(kdim, adim, bias=bias)
            self.w_query = nn.Linear(qdim, adim, bias=bias)
        elif atype == 'add':
            # for LAS
            self.w_key = nn.Linear(kdim, adim, bias=bias)
            self.w_value = nn.Linear(kdim, adim, bias=bias)
            self.w_query = nn.Linear(qdim, adim, bias=bias)
            self.v = nn.Linear(adim, n_heads, bias=bias)
        else:
            raise NotImplementedError(atype)

        self.w_out = nn.Linear(adim, odim, bias=bias)

        if param_init == 'xavier_uniform':
            self.reset_parameters(bias)

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' %
                    self.__class__.__name__)
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

    def reset(self):
        self.key = None
        self.value = None
        self.mask = None

    def forward(self, key, value, query, mask, aw_prev=None, aw_lower=None,
                cache=False, mode='', trigger_points=None, eps_wait=-1, streaming=False):
        """Forward pass.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            value (FloatTensor): `[B, klen, vdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            aw_prev: dummy interface
            cache (bool): cache key, value, and mask
            mode: dummy interface for MoChA/MMA
            trigger_points: dummy interface for MoChA/MMA
            eps_wait: dummy interface for MMA
            streaming: dummy interface for streaming attention
        Returns:
            cv (FloatTensor): `[B, qlen, vdim]`
            aw (FloatTensor): `[B, H, qlen, klen]`
            attn_state (dict): dummy interface

        """
        bs, klen = key.size()[: 2]
        qlen = query.size(1)
        attn_state = {}

        # Pre-computation of encoder-side features for computing scores
        if self.key is None or not cache:
            self.key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)  # `[B, klen, H, d_k]`
            self.value = self.w_value(value).view(bs, -1, self.n_heads, self.d_k)  # `[B, klen, H, d_k]`
            if mask is not None:
                self.mask = mask.unsqueeze(3).repeat([1, 1, 1, self.n_heads])
                mask_size = (bs, qlen, klen, self.n_heads)
                assert self.mask.size() == mask_size, (self.mask.size(), mask_size)
            else:
                self.mask = None

        key = self.key
        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)  # `[B, qlen, H, d_k]`

        if self.atype == 'scaled_dot':
            e = torch.einsum("bihd,bjhd->bijh", (query, key)) / self.scale
        elif self.atype == 'add':
            e = self.v(torch.tanh(key[:, None] + query[:, :, None]).view(bs, qlen, klen, -1))
        # e: `[B, qlen, klen, H]`

        # Compute attention weights
        if self.mask is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=e.dtype).numpy().dtype).min)
            e = e.masked_fill_(self.mask == 0, NEG_INF)  # `[B, qlen, klen, H]`
        aw = torch.softmax(e, dim=2)
        aw = self.dropout_attn(aw)
        aw_masked = aw.clone()

        # mask out each head independently (HeadDrop)
        if self.dropout_head > 0 and self.training:
            aw_masked = aw_masked.permute(0, 3, 1, 2)
            aw_masked = headdrop(aw_masked, self.n_heads, self.dropout_head)  # `[B, H, qlen, klen]`
            aw_masked = aw_masked.permute(0, 2, 3, 1)

        cv = torch.einsum("bijh,bjhd->bihd", (aw_masked, self.value))  # `[B, qlen, H, d_k]`
        cv = cv.contiguous().view(bs, -1, self.n_heads * self.d_k)  # `[B, qlen, H * d_k]`
        cv = self.w_out(cv)
        aw = aw.permute(0, 3, 1, 2)  # `[B, H, qlen, klen]`

        return cv, aw, attn_state
