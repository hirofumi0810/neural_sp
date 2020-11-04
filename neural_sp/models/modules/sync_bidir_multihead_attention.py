# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Synchronous bidirectional multi-head attention layer."""

import logging
import math
import numpy as np
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class SyncBidirMultiheadAttentionMechanism(nn.Module):
    """Multi-headed attention layer.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of attention space
        odim: (int) dimension of output
        n_heads (int): number of heads
        dropout (float): dropout probability
        atype (str): type of attention mechanisms
        bias (bool): use bias term in linear layers
        param_init (str): parameter initialization method
        future_weight (float):

    """

    def __init__(self, kdim, qdim, adim, odim, n_heads, dropout,
                 atype='scaled_dot', bias=True, param_init='',
                 future_weight=0.1):

        super().__init__()

        self.atype = atype
        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(self.d_k)
        self.future_weight = future_weight
        self.reset()

        self.dropout = nn.Dropout(p=dropout)

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

    def reset(self):
        self.key_fwd = None
        self.key_bwd = None
        self.value_fwd = None
        self.value_bwd = None
        self.tgt_mask = None
        self.identity_mask = None

    def forward(self, key_fwd, value_fwd, query_fwd,
                key_bwd, value_bwd, query_bwd,
                tgt_mask, identity_mask,
                mode='', cache=True, trigger_points=None):
        """Forward pass.

        Args:
            key_fwd (FloatTensor): `[B, klen, kdim]`
            value_fwd (FloatTensor): `[B, klen, vdim]`
            query_fwd (FloatTensor): `[B, qlen, qdim]`
            key_bwd (FloatTensor): `[B, klen, kdim]`
            value_bwd (FloatTensor): `[B, klen, vdim]`
            query_bwd (FloatTensor): `[B, qlen, qdim]`
            tgt_mask (ByteTensor): `[B, qlen, klen]`
            identity_mask (ByteTensor): `[B, qlen, klen]`
            mode: dummy interface for MoChA/MMA
            cache (bool): cache key, value, and tgt_mask
            trigger_points (IntTensor): dummy
        Returns:
            cv_fwd (FloatTensor): `[B, qlen, vdim]`
            cv_bwd (FloatTensor): `[B, qlen, vdim]`
            aw_fwd_h (FloatTensor): `[B, H, qlen, klen]`
            aw_fwd_f (FloatTensor): `[B, H, qlen, klen]`
            aw_bwd_h (FloatTensor): `[B, H, qlen, klen]`
            aw_bwd_f (FloatTensor): `[B, H, qlen, klen]`

        """
        bs, klen = key_fwd.size()[: 2]
        qlen = query_fwd.size(1)

        if self.key_fwd is None or not cache:
            key_fwd = self.w_key(key_fwd).view(bs, -1, self.n_heads, self.d_k)
            self.key_fwd = key_fwd.transpose(2, 1).contiguous()      # `[B, H, klen, d_k]`
            value_fwd = self.w_value(value_fwd).view(bs, -1, self.n_heads, self.d_k)
            self.value_fwd = value_fwd.transpose(2, 1).contiguous()  # `[B, H, klen, d_k]`
            self.tgt_mask = tgt_mask
            self.identity_mask = identity_mask
            if tgt_mask is not None:
                self.tgt_mask = tgt_mask.unsqueeze(1).repeat([1, self.n_heads, 1, 1])
                assert self.tgt_mask.size() == (bs, self.n_heads, qlen, klen)
            if identity_mask is not None:
                self.identity_mask = identity_mask.unsqueeze(1).repeat([1, self.n_heads, 1, 1])
                assert self.identity_mask.size() == (bs, self.n_heads, qlen, klen)
        if self.key_bwd is None or not cache:
            key_bwd = self.w_key(key_bwd).view(bs, -1, self.n_heads, self.d_k)
            self.key_bwd = key_bwd.transpose(2, 1).contiguous()  # `[B, H, klen, d_k]`
            value_bwd = self.w_value(value_bwd).view(bs, -1, self.n_heads, self.d_k)
            self.value_bwd = value_bwd.transpose(2, 1).contiguous()  # `[B, H, klen, d_k]`

        query_fwd = self.w_query(query_fwd).view(bs, -1, self.n_heads, self.d_k)
        query_fwd = query_fwd.transpose(2, 1).contiguous()  # `[B, H, qlen, d_k]`
        query_bwd = self.w_query(query_bwd).view(bs, -1, self.n_heads, self.d_k)
        query_bwd = query_bwd.transpose(2, 1).contiguous()  # `[B, H, qlen, d_k]`

        if self.atype == 'scaled_dot':
            e_fwd_h = torch.matmul(query_fwd, self.key_fwd.transpose(3, 2)) / self.scale
            e_fwd_f = torch.matmul(query_fwd, self.key_bwd.transpose(3, 2)) / self.scale
            e_bwd_h = torch.matmul(query_bwd, self.key_bwd.transpose(3, 2)) / self.scale
            e_bwd_f = torch.matmul(query_bwd, self.key_fwd.transpose(3, 2)) / self.scale
        elif self.atype == 'add':
            e_fwd_h = torch.tanh(self.key_fwd.unsqueeze(2) + query_fwd.unsqueeze(3))
            e_fwd_h = e_fwd_h.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)
            e_fwd_h = self.v(e_fwd_h).permute(0, 3, 1, 2)
            e_fwd_f = torch.tanh(self.key_bwd.unsqueeze(2) + query_fwd.unsqueeze(3))
            e_fwd_f = e_fwd_f.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)
            e_fwd_f = self.v(e_fwd_f).permute(0, 3, 1, 2)
            e_bwd_h = torch.tanh(self.key_bwd.unsqueeze(2) + query_bwd.unsqueeze(3))
            e_bwd_h = e_bwd_h.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)
            e_bwd_h = self.v(e_bwd_h).permute(0, 3, 1, 2)
            e_bwd_f = torch.tanh(self.key_fwd.unsqueeze(2) + query_bwd.unsqueeze(3))
            e_bwd_f = e_bwd_f.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)
            e_bwd_f = self.v(e_bwd_f).permute(0, 3, 1, 2)

        # Compute attention weights
        if self.tgt_mask is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=e_fwd_h.dtype).numpy().dtype).min)
            e_fwd_h = e_fwd_h.masked_fill_(self.tgt_mask == 0, NEG_INF)  # `[B, H, qlen, klen]`
            e_bwd_h = e_bwd_h.masked_fill_(self.tgt_mask == 0, NEG_INF)  # `[B, H, qlen, klen]`
        if self.identity_mask is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=e_fwd_f.dtype).numpy().dtype).min)
            e_fwd_f = e_fwd_f.masked_fill_(self.identity_mask == 0, NEG_INF)  # `[B, H, qlen, klen]`
            e_bwd_f = e_bwd_f.masked_fill_(self.identity_mask == 0, NEG_INF)  # `[B, H, qlen, klen]`
        aw_fwd_h = self.dropout(torch.softmax(e_fwd_h, dim=-1))
        aw_fwd_f = self.dropout(torch.softmax(e_fwd_f, dim=-1))
        aw_bwd_h = self.dropout(torch.softmax(e_bwd_h, dim=-1))
        aw_bwd_f = self.dropout(torch.softmax(e_bwd_f, dim=-1))

        cv_fwd_h = torch.matmul(aw_fwd_h, self.value_fwd)  # `[B, H, qlen, d_k]`
        cv_fwd_f = torch.matmul(aw_fwd_f, self.value_bwd)  # `[B, H, qlen, d_k]`
        cv_bwd_h = torch.matmul(aw_bwd_h, self.value_bwd)  # `[B, H, qlen, d_k]`
        cv_bwd_f = torch.matmul(aw_bwd_f, self.value_fwd)  # `[B, H, qlen, d_k]`

        cv_fwd_h = cv_fwd_h.transpose(2, 1).contiguous().view(bs, -1, self.n_heads * self.d_k)
        cv_fwd_h = self.w_out(cv_fwd_h)
        cv_fwd_f = cv_fwd_f.transpose(2, 1).contiguous().view(bs, -1, self.n_heads * self.d_k)
        cv_fwd_f = self.w_out(cv_fwd_f)
        cv_bwd_h = cv_bwd_h.transpose(2, 1).contiguous().view(bs, -1, self.n_heads * self.d_k)
        cv_bwd_h = self.w_out(cv_bwd_h)
        cv_bwd_f = cv_bwd_f.transpose(2, 1).contiguous().view(bs, -1, self.n_heads * self.d_k)
        cv_bwd_f = self.w_out(cv_bwd_f)

        # merge history and future information
        cv_fwd = cv_fwd_h + self.future_weight * torch.tanh(cv_fwd_f)
        cv_bwd = cv_bwd_h + self.future_weight * torch.tanh(cv_bwd_f)

        return cv_fwd, cv_bwd, aw_fwd_h, aw_fwd_f, aw_bwd_h, aw_bwd_f
