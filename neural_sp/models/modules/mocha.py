#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Monotonic (multihead) chunkwise atteniton."""

# [reference]
# https://github.com/j-min/MoChA-pytorch/blob/94b54a7fa13e4ac6dc255b509dd0febc8c0a0ee6/attention.py

import logging
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.modules.causal_conv import CausalConv1d
from neural_sp.models.modules.initialization import init_with_xavier_uniform

random.seed(1)

logger = logging.getLogger(__name__)


class MonotonicEnergy(nn.Module):
    def __init__(self, kdim, qdim, adim, atype, n_heads, init_r,
                 bias=True, param_init='', conv1d=False, conv_kernel_size=5):
        """Energy function for the monotonic attenion.

        Args:
            kdim (int): dimension of key
            qdim (int): dimension of quary
            adim (int): dimension of attention space
            atype (str): type of attention mechanism
            n_heads (int): number of monotonic attention heads
            init_r (int): initial value for offset r
            bias (bool): use bias term in linear layers
            param_init (str): parameter initialization method
            conv1d (bool): use 1D causal convolution for energy calculation
            conv_kernel_size (int): kernel size for 1D convolution

        """
        super().__init__()

        assert conv_kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        self.key = None
        self.mask = None

        self.atype = atype
        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(adim)

        if atype == 'add':
            self.w_key = nn.Linear(kdim, adim)
            self.v = nn.Linear(adim, n_heads, bias=False)
            self.w_query = nn.Linear(qdim, adim, bias=False)
        elif atype == 'scaled_dot':
            self.w_key = nn.Linear(kdim, adim, bias=bias)
            self.w_query = nn.Linear(qdim, adim, bias=bias)
        else:
            raise NotImplementedError(atype)

        self.r = nn.Parameter(torch.Tensor([init_r]))
        logger.info('init_r is initialized with %d' % init_r)

        self.conv1d = None
        if conv1d:
            self.conv1d = CausalConv1d(in_channels=kdim,
                                       out_channels=kdim,
                                       kernel_size=conv_kernel_size)
            # padding=(conv_kernel_size - 1) // 2

        if atype == 'add':
            self.v = nn.utils.weight_norm(self.v, name='weight', dim=0)
            # initialization
            self.v.weight_g.data = torch.Tensor([1 / adim]).sqrt()
        elif atype == 'scaled_dot':
            if param_init == 'xavier_uniform':
                self.reset_parameters(bias)

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        # NOTE: see https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.)
            nn.init.constant_(self.w_query.bias, 0.)
        if self.conv1d is not None:
            logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.conv1d.__class__.__name__)
            for n, p in self.conv1d.named_parameters():
                init_with_xavier_uniform(n, p)

    def reset(self):
        self.key = None
        self.mask = None

    def forward(self, key, query, mask, cache=False, boundary_leftmost=0):
        """Compute monotonic energy.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            cache (bool): cache key and mask
        Returns:
            e (FloatTensor): `[B, H_ma, qlen, klen]`

        """
        bs, klen, kdim = key.size()
        qlen = query.size(1)

        # Pre-computation of encoder-side features for computing scores
        if self.key is None or not cache:
            # 1d conv
            if self.conv1d is not None:
                key = torch.relu(self.conv1d(key))
            key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)
            self.key = key.transpose(2, 1).contiguous()  # `[B, H_ma, klen, d_k]`
            self.mask = mask
            if mask is not None:
                self.mask = self.mask.unsqueeze(1).repeat([1, self.n_heads, 1, 1])  # `[B, H_ma, qlen, klen]`
                assert self.mask.size() == (bs, self.n_heads, qlen, klen), \
                    (self.mask.size(), (bs, self.n_heads, qlen, klen))

        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)
        query = query.transpose(2, 1).contiguous()  # `[B, H_ma, qlen, d_k]`
        m = self.mask

        if self.atype == 'add':
            k = self.key.unsqueeze(2)  # `[B, H_ma, 1, klen, d_k]`
            # Truncate encoder memories
            if boundary_leftmost > 0:
                k = k[:, :, :, boundary_leftmost:]
                klen = k.size(3)
                if m is not None:
                    m = m[:, :, :, boundary_leftmost:]
            e = torch.relu(k + query.unsqueeze(3))  # `[B, H_ma, qlen, klen, d_k]`
            e = e.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)
            e = self.v(e).permute(0, 3, 1, 2)  # `[B, qlen, klen, H_ma]`
        elif self.atype == 'scaled_dot':
            k = self.key.transpose(3, 2)
            e = torch.matmul(query, k) / self.scale

        if self.r is not None:
            e = e + self.r
        if m is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=e.dtype).numpy().dtype).min)
            e = e.masked_fill_(m == 0, NEG_INF)
        assert e.size() == (bs, self.n_heads, qlen, klen), \
            (e.size(), (bs, self.n_heads, qlen, klen))
        return e


class ChunkEnergy(nn.Module):
    def __init__(self, kdim, qdim, adim, atype, n_heads=1,
                 bias=True, param_init=''):
        """Energy function for the chunkwise attention.

        Args:
            kdim (int): dimension of key
            qdim (int): dimension of quary
            adim (int): dimension of attention space
            atype (str): type of attention mechanism
            n_heads (int): number of chunkwise attention heads
            bias (bool): use bias term in linear layers
            param_init (str): parameter initialization method

        """
        super().__init__()

        self.key = None
        self.mask = None

        self.atype = atype
        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(adim)

        if atype == 'add':
            self.w_key = nn.Linear(kdim, adim)
            self.w_query = nn.Linear(qdim, adim, bias=False)
            self.v = nn.Linear(adim, n_heads, bias=False)
        elif atype == 'scaled_dot':
            self.w_key = nn.Linear(kdim, adim, bias=bias)
            self.w_query = nn.Linear(qdim, adim, bias=bias)
            if param_init == 'xavier_uniform':
                self.reset_parameters(bias)
        else:
            raise NotImplementedError(atype)

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        # NOTE: see https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.)
            nn.init.constant_(self.w_query.bias, 0.)

    def reset(self):
        self.key = None
        self.mask = None

    def forward(self, key, query, mask, cache=False,
                boundary_leftmost=0, boundary_rightmost=10e6):
        """Compute chunkwise energy.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            cache (bool): cache key and mask
        Returns:
            e (FloatTensor): `[B, H_ca, qlen, klen]`

        """
        bs, klen, kdim = key.size()
        qlen = query.size(1)

        # Pre-computation of encoder-side features for computing scores
        if self.key is None or not cache:
            key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)
            self.key = key.transpose(2, 1).contiguous()  # `[B, H_ca, klen, d_k]`
            self.mask = mask
            if mask is not None:
                self.mask = self.mask.unsqueeze(1).repeat([1, self.n_heads, 1, 1])  # `[B, H_ca, qlen, klen]`
                assert self.mask.size() == (bs, self.n_heads, qlen, klen), \
                    (self.mask.size(), (bs, self.n_heads, qlen, klen))

        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)
        query = query.transpose(2, 1).contiguous()  # `[B, H_ca, qlen, d_k]`
        m = self.mask

        if self.atype == 'add':
            k = self.key.unsqueeze(2)  # `[B, H_ca, 1, klen, d_k]`
            # Truncate
            k = k[:, :, :, boundary_leftmost:boundary_rightmost]
            klen = k.size(3)
            if m is not None:
                m = m[:, :, :, boundary_leftmost:boundary_rightmost]

            r = torch.relu(k + query.unsqueeze(3))  # `[B, H_ca, qlen, klen, d_k]`
            r = r.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)  # `[B, qlen, klen, H_ca * d_k]`
            r = self.v(r).permute(0, 3, 1, 2).contiguous()  # `[B, H_ca, qlen, klen]`
        elif self.atype == 'scaled_dot':
            k = self.key.transpose(3, 2)
            r = torch.matmul(query, k) / self.scale

        if m is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=r.dtype).numpy().dtype).min)
            r = r.masked_fill_(m == 0, NEG_INF)
        assert r.size() == (bs, self.n_heads, qlen, klen), \
            (r.size(), (bs, self.n_heads, qlen, klen))
        return r


class MoChA(nn.Module):
    def __init__(self, kdim, qdim, adim, odim, atype, chunk_size,
                 n_heads_mono=1, n_heads_chunk=1,
                 conv1d=False, init_r=-4, eps=1e-6, noise_std=1.0,
                 no_denominator=False, sharpening_factor=1.0,
                 dropout=0., dropout_head=0., bias=True, param_init='',
                 decot=False, lookahead=2, share_chunkwise_attention=False):
        """Monotonic (multihead) chunkwise attention.

            if chunk_size == 1, this is equivalent to Hard monotonic attention
                "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
                    https://arxiv.org/abs/1704.00784
            if chunk_size > 1, this is equivalent to monotonic chunkwise attention (MoChA)
                "Monotonic Chunkwise Attention" (ICLR 2018)
                    https://openreview.net/forum?id=Hko85plCW
            if chunk_size == -1, this is equivalent to Monotonic infinite lookback attention (Milk)
                "Monotonic Infinite Lookback Attention for Simultaneous Machine Translation" (ACL 2019)
                    https://arxiv.org/abs/1906.05218
            if chunk_size == 1 and n_heads_mono>1, this is equivalent to Monotonic Multihead Attention (MMA)-hard
                "Monotonic Multihead Attention" (ICLR 2020)
                    https://openreview.net/forum?id=Hyg96gBKPS
            if chunk_size == -1 and n_heads_mono>1, this is equivalent to Monotonic Multihead Attention (MMA)-Ilk
                "Monotonic Multihead Attention" (ICLR 2020)
                    https://openreview.net/forum?id=Hyg96gBKPS

        Args:
            kdim (int): dimension of key
            qdim (int): dimension of query
            adim: (int) dimension of the attention layer
            odim: (int) dimension of output
            atype (str): type of attention mechanism
            chunk_size (int): window size for chunkwise attention
            n_heads_mono (int): number of heads for monotonic attention
            n_heads_chunk (int): number of heads for chunkwise attention
            conv1d (bool): apply 1d convolution for energy calculation
            init_r (int): initial value for parameter 'r' used for monotonic attention
            eps (float): epsilon parameter to avoid zero division
            noise_std (float): standard deviation for input noise
            no_denominator (bool): set the denominator to 1 in the alpha recurrence
            sharpening_factor (float): sharping factor for beta calculation
            dropout (float): dropout probability for attention weights
            dropout_head (float): HeadDrop probability
            bias (bool): use bias term in linear layers
            param_init (str): parameter initialization method
            decot (bool): delay constrainted training (DeCoT)
            lookahead (int): lookahead frames for DeCoT
            share_chunkwise_attention (int): share CA heads among MA heads

        """
        super(MoChA, self).__init__()

        self.atype = atype
        assert adim % (n_heads_mono * n_heads_chunk) == 0
        self.d_k = adim // (n_heads_mono * n_heads_chunk)

        self.w = chunk_size
        self.milk = (chunk_size == -1)
        self.n_heads = n_heads_mono
        self.n_heads_mono = n_heads_mono
        self.n_heads_chunk = n_heads_chunk
        self.eps = eps
        self.noise_std = noise_std
        self.no_denom = no_denominator
        self.sharpening_factor = sharpening_factor
        self.decot = decot
        self.lookahead = lookahead
        self.share_chunkwise_attention = share_chunkwise_attention

        self.monotonic_energy = MonotonicEnergy(kdim, qdim, adim, atype,
                                                n_heads_mono, init_r,
                                                bias, param_init, conv1d=conv1d)
        self.chunk_energy = ChunkEnergy(kdim, qdim, adim, atype,
                                        n_heads_chunk if share_chunkwise_attention else n_heads_mono * n_heads_chunk,
                                        bias, param_init) if chunk_size > 1 or self.milk else None
        if n_heads_mono * n_heads_chunk > 1:
            self.w_value = nn.Linear(kdim, adim, bias=bias)
            self.w_out = nn.Linear(adim, odim, bias=bias)
            if param_init == 'xavier_uniform':
                self.reset_parameters(bias)

        # attention dropout
        self.dropout_attn = nn.Dropout(p=dropout)  # for beta
        self.dropout_head = dropout_head

        self.bd_offset = 0

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        # NOTE: see https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
        nn.init.xavier_uniform_(self.w_value.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_value.bias, 0.)

        nn.init.xavier_uniform_(self.w_out.weight)
        if bias:
            nn.init.constant_(self.w_out.bias, 0.)

    def reset(self):
        self.monotonic_energy.reset()
        if self.chunk_energy is not None:
            self.chunk_energy.reset()
        self.bd_offset = 0

    def forward(self, key, value, query, mask=None, aw_prev=None,
                mode='hard', cache=False, trigger_point=None,
                eps_wait=-1, efficient_decoding=False):
        """Forward pass.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            value (FloatTensor): `[B, klen, vdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            aw_prev (FloatTensor): `[B, H, 1, klen]`
            mode (str): recursive/parallel/hard
            cache (bool): cache key and mask
            trigger_point (IntTensor): `[B]`
            eps_wait (int): wait time delay for head-synchronous decoding in MMA
        Returns:
            cv (FloatTensor): `[B, qlen, vdim]`
            alpha (FloatTensor): `[B, H_ma, qlen, klen]`
            beta (FloatTensor): `[B, H_ma * H_ca, qlen, klen]`

        """
        bs, klen = key.size()[:2]
        qlen = query.size(1)

        if aw_prev is None:
            # aw_prev = [1, 0, 0 ... 0]
            aw_prev = key.new_zeros(bs, self.n_heads_mono, 1, klen)
            aw_prev[:, :, :, 0:1] = key.new_ones(bs, self.n_heads_mono, 1, 1)

        # Compute monotonic energy
        e_mono = self.monotonic_energy(key, query, mask, cache=cache,
                                       boundary_leftmost=self.bd_offset)  # `[B, H_ma, qlen, klen]`
        assert e_mono.size(3) + self.bd_offset == key.size(1)

        if mode == 'recursive':  # training
            p_choose = torch.sigmoid(add_gaussian_noise(e_mono, self.noise_std))  # `[B, H_ma, qlen, klen]`
            alpha = []
            for i in range(qlen):
                # Compute [1, 1 - p_choose[0], 1 - p_choose[1], ..., 1 - p_choose[-2]]
                shifted_1mp_choose = torch.cat([key.new_ones(bs, self.n_heads_mono, 1, 1),
                                                1 - p_choose[:, :, i:i + 1, :-1]], dim=-1)
                # Compute attention distribution recursively as
                # q_j = (1 - p_choose_j) * q_(j-1) + aw_prev_j
                # alpha_j = p_choose_j * q_j
                q = key.new_zeros(bs, self.n_heads_mono, 1, klen + 1)
                for j in range(klen):
                    q[:, :, i:i + 1, j + 1] = shifted_1mp_choose[:, :, i:i + 1, j].clone() * q[:, :, i:i + 1, j].clone() + \
                        aw_prev[:, :, :, j].clone()
                aw_prev = p_choose[:, :, i:i + 1] * q[:, :, i:i + 1, 1:]  # `[B, H_ma, 1, klen]`
                alpha.append(aw_prev)
            alpha = torch.cat(alpha, dim=2) if qlen > 1 else alpha[-1]  # `[B, H_ma, qlen, klen]`
            alpha_masked = alpha.clone()

        elif mode == 'parallel':  # training
            p_choose = torch.sigmoid(add_gaussian_noise(e_mono, self.noise_std))  # `[B, H_ma, qlen, klen]`
            # safe_cumprod computes cumprod in logspace with numeric checks
            cumprod_1mp_choose = safe_cumprod(1 - p_choose, eps=self.eps)  # `[B, H_ma, qlen, klen]`
            # Compute recurrence relation solution
            alpha = []
            for i in range(qlen):
                denom = 1 if self.no_denom else torch.clamp(cumprod_1mp_choose[:, :, i:i + 1], min=self.eps, max=1.0)
                aw_prev = p_choose[:, :, i:i + 1] * cumprod_1mp_choose[:, :, i:i + 1] * torch.cumsum(
                    aw_prev / denom, dim=-1)  # `[B, H_ma, 1, klen]`
                # Mask the right part from the trigger point
                if self.decot and trigger_point is not None:
                    for b in range(bs):
                        aw_prev[b, :, :, trigger_point[b] + self.lookahead + 1:] = 0
                alpha.append(aw_prev)

            alpha = torch.cat(alpha, dim=2) if qlen > 1 else alpha[-1]  # `[B, H_ma, qlen, klen]`
            alpha_masked = alpha.clone()

            # mask out each head independently (HeadDrop)
            if self.dropout_head > 0 and self.training:
                n_effective_heads = self.n_heads_mono
                head_mask = alpha.new_ones(alpha.size()).byte()
                for h in range(self.n_heads_mono):
                    if random.random() < self.dropout_head:
                        head_mask[:, h] = 0
                        n_effective_heads -= 1
                alpha_masked = alpha_masked.masked_fill_(head_mask == 0, 0)
                # Normalization
                if n_effective_heads > 0:
                    alpha_masked = alpha_masked * (self.n_heads_mono / n_effective_heads)

        elif mode == 'hard':  # inference
            assert qlen == 1
            assert not self.training
            p_choose = None
            if self.n_heads_mono == 1:
                # assert aw_prev.sum() > 0
                p_choose_i = (torch.sigmoid(e_mono) >= 0.5).float()[:, :, 0:1]
                # Attend when monotonic energy is above threshold (Sigmoid > 0.5)
                # Remove any probabilities before the index chosen at the last time step
                p_choose_i *= torch.cumsum(
                    aw_prev[:, :, 0:1, -e_mono.size(3):], dim=-1)  # `[B, H_ma, 1 (qlen), klen]`
                # Now, use exclusive cumprod to remove probabilities after the first
                # chosen index, like so:
                # p_choose_i                        = [0, 0, 0, 1, 1, 0, 1, 1]
                # 1 - p_choose_i                    = [1, 1, 1, 0, 0, 1, 0, 0]
                # exclusive_cumprod(1 - p_choose_i) = [1, 1, 1, 1, 0, 0, 0, 0]
                # alpha: product of above           = [0, 0, 0, 1, 0, 0, 0, 0]
                alpha = p_choose_i * exclusive_cumprod(1 - p_choose_i)  # `[B, H_ma, 1 (qlen), klen]`
            else:
                p_choose_i = (torch.sigmoid(e_mono) >= 0.5).float()[:, :, 0:1]
                # Attend when monotonic energy is above threshold (Sigmoid > 0.5)
                # Remove any probabilities before the index chosen at the last time step
                p_choose_i *= torch.cumsum(aw_prev[:, :, 0:1], dim=-1)  # `[B, H_ma, 1 (qlen), klen]`
                # Now, use exclusive cumprod to remove probabilities after the first
                # chosen index, like so:
                # p_choose_i                        = [0, 0, 0, 1, 1, 0, 1, 1]
                # 1 - p_choose_i                    = [1, 1, 1, 0, 0, 1, 0, 0]
                # exclusive_cumprod(1 - p_choose_i) = [1, 1, 1, 1, 0, 0, 0, 0]
                # alpha: product of above           = [0, 0, 0, 1, 0, 0, 0, 0]
                alpha = p_choose_i * exclusive_cumprod(1 - p_choose_i)  # `[B, H_ma, 1 (qlen), klen]`

            if eps_wait > 0:
                for b in range(bs):
                    # no boundary until the last frame for all heads
                    if alpha[b].sum() == 0:
                        continue

                    leftmost = alpha[b, :, 0].nonzero()[:, -1].min().item()
                    rightmost = alpha[b, :, 0].nonzero()[:, -1].max().item()
                    for h in range(self.n_heads_mono):
                        # no bondary at the h-th head
                        if alpha[b, h, 0].sum().item() == 0:
                            alpha[b, h, 0, min(rightmost, leftmost + eps_wait)] = 1
                            continue

                        # surpass acceptable latency
                        if alpha[b, h, 0].nonzero()[:, -1].min().item() >= leftmost + eps_wait:
                            alpha[b, h, 0, :] = 0  # reset
                            alpha[b, h, 0, leftmost + eps_wait] = 1

            alpha_masked = alpha.clone()

        else:
            raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")

        # Compute chunk energy
        beta = None
        if self.w > 1 or self.milk:
            bd_leftmost = 0
            bd_rightmost = klen - 1 - self.bd_offset
            if efficient_decoding and mode == 'hard' and alpha.sum() > 0:
                bd_leftmost = alpha[:, :, 0].nonzero()[:, -1].min().item()
                bd_rightmost = alpha[:, :, 0].nonzero()[:, -1].max().item()
                if bd_leftmost == bd_rightmost:
                    alpha_masked = alpha_masked[:, :, :, bd_leftmost:bd_leftmost + 1]
                else:
                    alpha_masked = alpha_masked[:, :, :, bd_leftmost:bd_rightmost]

            e_chunk = self.chunk_energy(key, query, mask, cache=cache,
                                        boundary_leftmost=max(0, self.bd_offset + bd_leftmost - self.w + 1),
                                        boundary_rightmost=self.bd_offset + bd_rightmost + 1)  # `[B, (H_ma*)H_ca, qlen, ken]`

            # padding
            additional = e_chunk.size(3) - alpha_masked.size(3)
            if efficient_decoding and mode == 'hard':
                alpha = torch.cat([alpha.new_zeros(bs, alpha.size(1), 1, klen - alpha.size(3)), alpha], dim=3)
                if additional > 0:
                    alpha_masked = torch.cat([alpha_masked.new_zeros(bs, alpha_masked.size(1), 1, additional),
                                              alpha_masked], dim=3)

            if mode == 'hard':
                beta = hard_chunkwise_attention(alpha_masked, e_chunk, mask, self.w,
                                                self.n_heads_chunk, self.sharpening_factor,
                                                self.share_chunkwise_attention)

            else:
                beta = efficient_chunkwise_attention(alpha_masked, e_chunk, mask, self.w,
                                                     self.n_heads_chunk, self.sharpening_factor,
                                                     self.share_chunkwise_attention)
            beta = self.dropout_attn(beta)  # `[B, H_ma * H_ca, qlen, klen]`

            if efficient_decoding and mode == 'hard':
                value = value[:, max(0, self.bd_offset + bd_leftmost - self.w + 1):self.bd_offset + bd_rightmost + 1]

        # Update after calculating beta
        bd_offset_old = self.bd_offset
        if efficient_decoding and mode == 'hard' and alpha.sum() > 0:
            self.bd_offset += alpha[:, :, 0, self.bd_offset:].nonzero()[:, -1].min().item()

        # Compute context vector
        if self.n_heads_mono * self.n_heads_chunk > 1:
            value = self.w_value(value).view(bs, -1, self.n_heads_mono * self.n_heads_chunk, self.d_k)
            value = value.transpose(2, 1).contiguous()  # `[B, H_ma * H_ca, klen, d_k]`
            if self.w == 1:
                cv = torch.matmul(alpha, value)  # `[B, H_ma, qlen, d_k]`
            else:
                cv = torch.matmul(beta, value)  # `[B, H_ma * H_ca, qlen, d_k]`
            cv = cv.transpose(2, 1).contiguous().view(bs, -1, self.n_heads_mono * self.n_heads_chunk * self.d_k)
            cv = self.w_out(cv)  # `[B, qlen, adim]`
        else:
            if self.w == 1:
                cv = torch.bmm(alpha.squeeze(1), value)  # `[B, 1, adim]`
            else:
                cv = torch.bmm(beta.squeeze(1), value)  # `[B, 1, adim]`

        assert alpha.size() == (bs, self.n_heads_mono, qlen, klen), \
            (alpha.size(), (bs, self.n_heads_mono, qlen, klen))
        if self.w > 1 or self.milk:
            _w = max(1, (bd_offset_old + bd_rightmost + 1) - max(0, bd_offset_old + bd_leftmost - self.w + 1))
            # assert beta.size() == (bs, self.n_heads_mono * self.n_heads_chunk, qlen, e_chunk.size(3) + additional), \
            #     (beta.size(), (bs, self.n_heads_mono * self.n_heads_chunk, qlen, e_chunk.size(3) + additional))
            assert beta.size() == (bs, self.n_heads_mono * self.n_heads_chunk, qlen, _w), \
                (beta.size(), (bs, self.n_heads_mono * self.n_heads_chunk, qlen, _w))
            # TODO: padding for beta

        return cv, alpha, beta, p_choose


def add_gaussian_noise(xs, std):
    """Additive gaussian nosie to encourage discreteness."""
    noise = xs.new_zeros(xs.size()).normal_(std=std)
    return xs + noise


def safe_cumprod(x, eps):
    """Numerically stable cumulative product by cumulative sum in log-space.
        Args:
            x (FloatTensor): `[B, H, qlen, klen]`
        Returns:
            x (FloatTensor): `[B, H, qlen, klen]`

    """
    return torch.exp(exclusive_cumsum(torch.log(torch.clamp(x, min=eps, max=1.0))))


def exclusive_cumsum(x):
    """Exclusive cumulative summation [a, b, c] => [0, a, a + b].

        Args:
            x (FloatTensor): `[B, H, qlen, klen]`
        Returns:
            x (FloatTensor): `[B, H, qlen, klen]`

    """
    return torch.cumsum(torch.cat([x.new_zeros(x.size(0), x.size(1), x.size(2), 1),
                                   x[:, :, :, :-1]], dim=-1), dim=-1)


def exclusive_cumprod(x):
    """Exclusive cumulative product [a, b, c] => [1, a, a * b].

        Args:
            x (FloatTensor): `[B, H, qlen, klen]`
        Returns:
            x (FloatTensor): `[B, H, qlen, klen]`

    """
    return torch.cumprod(torch.cat([x.new_ones(x.size(0), x.size(1), x.size(2), 1),
                                    x[:, :, :, :-1]], dim=-1), dim=-1)


def moving_sum(x, back, forward):
    """Compute the moving sum of x over a chunk_size with the provided bounds.

    Args:
        x (FloatTensor): `[B, H_ma, H_ca, qlen, klen]`
        back (int):
        forward (int):

    Returns:
        x_sum (FloatTensor): `[B, H_ma, H_ca, qlen, klen]`

    """
    bs, n_heads_mono, n_heads_chunk, qlen, klen = x.size()
    x = x.view(-1, klen)
    # Moving sum is computed as a carefully-padded 1D convolution with ones
    x_padded = F.pad(x, pad=[back, forward])  # `[B * H_ma * H_ca * qlen, back + klen + forward]`
    # Add a "channel" dimension
    x_padded = x_padded.unsqueeze(1)
    # Construct filters
    filters = x.new_ones(1, 1, back + forward + 1)
    x_sum = F.conv1d(x_padded, filters)
    x_sum = x_sum.squeeze(1).view(bs, n_heads_mono, n_heads_chunk, qlen, -1)
    return x_sum


def efficient_chunkwise_attention(alpha, u, mask, chunk_size, n_heads_chunk,
                                  sharpening_factor, share_chunkwise_attention):
    """Compute chunkwise attention efficiently by clipping logits at training time.

    Args:
        alpha (FloatTensor): `[B, H_ma, qlen, klen]`
        u (FloatTensor): `[B, (H_ma*)H_ca, qlen, klen]`
        mask (ByteTensor): `[B, qlen, klen]`
        chunk_size (int): window size for chunkwise attention
        n_heads_chunk (int): number of chunkwise attention heads
        sharpening_factor (float): sharping factor for beta calculation
        share_chunkwise_attention (int): share CA heads among MA heads
    Returns:
        beta (FloatTensor): `[B, H_ma * H_ca, qlen, klen]`

    """
    bs, n_heads_mono, qlen, klen = alpha.size()
    alpha = alpha.unsqueeze(2)  # `[B, H_ma, 1, qlen, klen]`
    u = u.unsqueeze(1)  # `[B, 1, (H_ma*)H_ca, qlen, klen]`
    if n_heads_chunk > 1:
        alpha = alpha.repeat([1, 1, n_heads_chunk, 1, 1])
    if n_heads_mono > 1 and not share_chunkwise_attention:
        u = u.view(bs, n_heads_mono, n_heads_chunk, qlen, klen)
    # Shift logits to avoid overflow
    u -= torch.max(u, dim=-1, keepdim=True)[0]
    # Limit the range for numerical stability
    softmax_exp = torch.clamp(torch.exp(u), min=1e-5)
    # Compute chunkwise softmax denominators
    if chunk_size == -1:
        # infinite lookback attention
        # inner_items = alpha * sharpening_factor / torch.cumsum(softmax_exp, dim=-1)
        # beta = softmax_exp * torch.cumsum(inner_items.flip(dims=[-1]), dim=-1).flip(dims=[-1])
        # beta = beta.masked_fill(mask.unsqueeze(1), 0)
        # beta = beta / beta.sum(dim=-1, keepdim=True)

        softmax_denominators = torch.cumsum(softmax_exp, dim=-1)
        # Compute \beta_{i, :}. emit_probs are \alpha_{i, :}.
        beta = softmax_exp * moving_sum(alpha * sharpening_factor / softmax_denominators,
                                        back=0, forward=klen - 1)
    else:
        softmax_denominators = moving_sum(softmax_exp,
                                          back=chunk_size - 1, forward=0)
        # Compute \beta_{i, :}. emit_probs are \alpha_{i, :}.
        beta = softmax_exp * moving_sum(alpha * sharpening_factor / softmax_denominators,
                                        back=0, forward=chunk_size - 1)
    return beta.view(bs, -1, qlen, klen)


def hard_chunkwise_attention(alpha, u, mask, chunk_size, n_heads_chunk,
                             sharpening_factor, share_chunkwise_attention):
    """Compute chunkwise attention over hard attention at test time.

    Args:
        alpha (FloatTensor): `[B, H_ma, qlen, klen]`
        u (FloatTensor): `[B, (H_ma*)H_ca, qlen, klen]`
        mask (ByteTensor): `[B, qlen, klen]`
        chunk_size (int): window size for chunkwise attention
        n_heads_chunk (int): number of chunkwise attention heads
        sharpening_factor (float): sharping factor for beta calculation
        share_chunkwise_attention (int): share CA heads among MA heads
    Returns:
        beta (FloatTensor): `[B, H_ma * H_ca, qlen, klen]`

    """
    bs, n_heads_mono, qlen, klen = alpha.size()
    alpha = alpha.unsqueeze(2)   # `[B, H_ma, 1, qlen, klen]`
    u = u.unsqueeze(1)  # `[B, 1, (H_ma*)H_ca, qlen, klen]`
    if n_heads_chunk > 1:
        alpha = alpha.repeat([1, 1, n_heads_chunk, 1, 1])
    if n_heads_mono > 1:
        if share_chunkwise_attention:
            u = u.repeat([1, n_heads_mono, 1, 1, 1])
        else:
            u = u.view(bs, n_heads_mono, n_heads_chunk, qlen, klen)

    mask = alpha.clone().byte()  # `[B, H_ma, H_ca, qlen, klen]`
    for b in range(bs):
        for h in range(n_heads_mono):
            if alpha[b, h, 0, 0].sum() > 0:
                boundary = alpha[b, h, 0, 0].nonzero()[:, -1].min().item()
                if chunk_size == -1:
                    # infinite lookback attention
                    mask[b, h, :, 0, 0:boundary + 1] = 1
                else:
                    mask[b, h, :, 0, max(0, boundary - chunk_size + 1):boundary + 1] = 1

    NEG_INF = float(np.finfo(torch.tensor(0, dtype=u.dtype).numpy().dtype).min)
    u = u.masked_fill(mask == 0, NEG_INF)
    beta = torch.softmax(u, dim=-1)
    return beta.view(bs, -1, qlen, klen)
