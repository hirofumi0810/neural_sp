#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""(Multi-head) Monotonic chunkwise atteniton (MoChA)."""

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

random.seed(1)

NEG_INF = float(np.finfo(np.float32).min)

logger = logging.getLogger(__name__)


class MonotonicEnergy(nn.Module):
    def __init__(self, kdim, qdim, adim, atype, n_heads, init_r,
                 conv1d=False, conv_kernel_size=5, param_init=''):
        """Energy function for the monotonic attenion.

        Args:
            kdim (int): dimension of key
            qdim (int): dimension of quary
            adim (int): dimension of attention space
            atype (str): type of attention mechanism
            n_heads (int): number of heads
            init_r (int): initial value for offset r
            conv1d (bool): use 1D causal convolution for energy calculation
            conv_kernel_size (int): kernel size for 1D convolution
            param_init (str): parameter initialization method

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

        self.w_key = nn.Linear(kdim, adim)
        if atype == 'add':
            self.v = nn.Linear(adim, n_heads, bias=False)
            self.w_query = nn.Linear(qdim, adim, bias=False)
        elif atype == 'scaled_dot':
            self.w_query = nn.Linear(qdim, adim)
        else:
            raise NotImplementedError(atype)

        self.r = nn.Parameter(torch.Tensor([init_r]))
        if atype == 'add':
            self.v = nn.utils.weight_norm(self.v, name='weight', dim=0)
            # initialization
            self.v.weight_g.data = torch.Tensor([1 / adim]).sqrt()
        elif atype == 'scaled_dot':
            # self.w_query = nn.utils.weight_norm(self.w_query, name='weight', dim=0)
            # initialization
            # self.w_query.weight_g.data = torch.Tensor([1 / adim]).sqrt()
            if param_init == 'xavier_uniform':
                self.reset_parameters(True)
            # TODO: debug weight normalization

        self.conv1d = None
        if conv1d:
            self.conv1d = CausalConv1d(in_channels=kdim,
                                       out_channels=kdim,
                                       kernel_size=conv_kernel_size,
                                       stride=1)
            # padding=(conv_kernel_size - 1) // 2

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        # NOTE: see https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        # nn.init.xavier_uniform_(self.w_query.weight_v, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            # newly introduced
            nn.init.constant_(self.w_key.bias, 0.)
            nn.init.constant_(self.w_query.bias, 0.)

    def reset(self):
        self.key = None
        self.mask = None

    def forward(self, key, query, mask, cache=True):
        """Compute monotonic energy.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            cache (bool): cache key and mask
        Return:
            e (FloatTensor): `[B, H, qlen, klen]`

        """
        bs, klen, kdim = key.size()
        qlen = query.size(1)

        # Pre-computation of encoder-side features for computing scores
        if self.key is None or not cache:
            # 1d conv
            if self.conv1d is not None:
                key = torch.relu(self.conv1d(key))
            key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)
            self.key = key.transpose(2, 1).contiguous()  # `[B, H, klen, d_k]`
            self.mask = mask
            if mask is not None:
                self.mask = self.mask.unsqueeze(1).repeat([1, self.n_heads, 1, 1])  # `[B, H, qlen, klen]`
                assert self.mask.size() == (bs, self.n_heads, qlen, klen)

        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)
        query = query.transpose(2, 1).contiguous()  # `[B, H, qlen, d_k]`

        if self.atype == 'add':
            key = self.key.unsqueeze(2)  # `[B, H, 1, klen, d_k]`
            query = query.unsqueeze(3)  # `[B, H, qlen, 1, d_k]`
            e = torch.relu(key + query)  # `[B, H, qlen, klen, d_k]`
            e = e.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)
            e = self.v(e).permute(0, 3, 1, 2)  # `[B, qlen, klen, H]`
        elif self.atype == 'scaled_dot':
            e = torch.matmul(query, self.key.transpose(3, 2)) / self.scale

        if self.r is not None:
            e = e + self.r
        if self.mask is not None:
            e = e.masked_fill_(self.mask == 0, NEG_INF)
        assert e.size() == (bs, self.n_heads, qlen, klen)
        return e


class ChunkEnergy(nn.Module):
    def __init__(self, kdim, qdim, adim, atype, n_heads=1, param_init=''):
        """Energy function for the chunkwise attention.

        Args:
            kdim (int): dimension of key
            qdim (int): dimension of quary
            adim (int): dimension of attention space
            atype (str): type of attention mechanism
            n_heads (int): number of heads
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

        self.w_key = nn.Linear(kdim, adim)
        if atype == 'add':
            assert self.n_heads == 1
            self.w_query = nn.Linear(qdim, adim, bias=False)
            self.v = nn.Linear(adim, 1, bias=False)
        elif atype == 'scaled_dot':
            self.w_query = nn.Linear(qdim, adim)
            if param_init == 'xavier_uniform':
                self.reset_parameters(True)
        else:
            raise NotImplementedError(atype)

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        # NOTE: see https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            # newly introduced
            nn.init.constant_(self.w_key.bias, 0.)
            nn.init.constant_(self.w_query.bias, 0.)

    def reset(self):
        self.key = None
        self.mask = None

    def forward(self, key, query, mask, cache=True):
        """Compute chunkwise energy.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            cache (bool): cache key and mask
        Return:
            energy (FloatTensor): `[B, H, qlen, klen]`

        """
        bs, klen, kdim = key.size()
        qlen = query.size(1)

        # Pre-computation of encoder-side features for computing scores
        if self.key is None or not cache:
            key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)
            self.key = key.transpose(2, 1).contiguous()  # `[B, H, klen, d_k]`
            self.mask = mask
            if mask is not None:
                self.mask = self.mask.unsqueeze(1).repeat([1, self.n_heads, 1, 1])  # `[B, H, qlen, klen]`
                assert self.mask.size() == (bs, self.n_heads, qlen, klen)

        if self.atype == 'add':
            key = self.key.unsqueeze(2)  # `[B, 1, 1, klen, d_k]`
            query = self.w_query(query).unsqueeze(1).unsqueeze(3)  # `[B, 1, qlen, 1, d_k]`
            energy = torch.relu(key + query)  # `[B, 1, klen, qlen, d_k]`
            energy = self.v(energy).squeeze(4)  # `[B, 1, qlen, klen]`
        elif self.atype == 'scaled_dot':
            query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)
            query = query.transpose(2, 1).contiguous()  # `[B, H, qlen, d_k]`
            energy = torch.matmul(query, self.key.transpose(3, 2)) / self.scale

        if self.mask is not None:
            energy = energy.masked_fill_(self.mask == 0, NEG_INF)
        assert energy.size() == (bs, self.n_heads, qlen, klen)
        return energy


class MoChA(nn.Module):
    def __init__(self, kdim, qdim, adim, atype, chunk_size,
                 n_heads_mono=1, n_heads_chunk=1,
                 conv1d=False, init_r=-4, noise_std=1.0, eps=1e-6,
                 sharpening_factor=1.0, dropout=0., dropout_head=0., param_init='',
                 decot=False, lookahead=2, simple=False):
        """Monotonic chunk-wise attention.

            "Monotonic Chunkwise Attention" (ICLR 2018)
            https://openreview.net/forum?id=Hko85plCW
            "Monotonic Multihead Attention" (ICLR 2020)
            https://openreview.net/forum?id=Hyg96gBKPS

            if chunk_size == 1, this is equivalent to Hard monotonic attention
                "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
                 http://arxiv.org/abs/1704.00784

        Args:
            kdim (int): dimension of key
            qdim (int): dimension of query
            adim: (int) dimension of the attention layer
            atype (str): type of attention mechanism
            chunk_size (int): window size for chunkwise attention
            n_heads_mono (int): number of heads for monotonic attention
            n_heads_chunk (int): number of heads for chunkwise attention
            conv1d (bool): apply 1d convolution for energy calculation
            init_r (int): initial value for parameter 'r' used for monotonic attention
            noise_std (float): standard deviation for input noise
            eps (float): epsilon parameter to avoid zero division
            sharpening_factor (float): sharping factor for beta calculation
            dropout (float): dropout probability for attention weights
            dropout_head (float): dropout probability for heads
            param_init (str): parameter initialization method
            decot (bool): delay constrainted training (DeCoT)
            lookahead (int): lookahead frames for DeCoT
            simple (bool):

        """
        super(MoChA, self).__init__()

        self.atype = atype
        assert adim % (n_heads_mono * n_heads_chunk) == 0
        self.d_k = adim // (n_heads_mono * n_heads_chunk)

        self.chunk_size = chunk_size
        self.milk = (chunk_size == -1)
        self.n_heads = n_heads_mono
        self.n_heads_mono = n_heads_mono
        self.n_heads_chunk = n_heads_chunk
        self.noise_std = noise_std
        self.eps = eps
        self.sharpening_factor = sharpening_factor

        self.decot = decot
        self.lookahead = lookahead
        self.simple = simple

        self.monotonic_energy = MonotonicEnergy(kdim, qdim, adim, atype,
                                                n_heads_mono, init_r, conv1d,
                                                param_init=param_init)
        self.chunk_energy = ChunkEnergy(kdim, qdim, adim, atype,
                                        n_heads_chunk, param_init) if chunk_size > 1 or self.milk else None
        if n_heads_mono * n_heads_chunk > 1:
            self.w_value = nn.Linear(kdim, adim)
            self.w_out = nn.Linear(adim, kdim)
            if param_init == 'xavier_uniform':
                self.reset_parameters(True)

        # attention dropout
        self.dropout = nn.Dropout(p=dropout)
        # head dropout
        self.dropout_head = dropout_head

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        # NOTE: see https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
        nn.init.xavier_uniform_(self.w_value.weight, gain=1 / math.sqrt(2))
        if bias:
            # newly introduced
            nn.init.constant_(self.w_value.bias, 0.)

        nn.init.xavier_uniform_(self.w_out.weight)
        if bias:
            nn.init.constant_(self.w_out.bias, 0.)

    def reset(self):
        self.monotonic_energy.reset()
        if self.chunk_energy is not None:
            self.chunk_energy.reset()

    def forward(self, key, value, query, mask=None, aw_prev=None,
                mode='hard', cache=True, trigger_point=None):
        """Soft monotonic attention during training.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            value (FloatTensor): `[B, klen, vdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            aw_prev (FloatTensor): `[B, H, 1, klen]`
            mode (str): recursive/parallel/hard
            cache (bool): cache key and mask
            trigger_point (IntTensor): `[B]`
        Return:
            cv (FloatTensor): `[B, qlen, vdim]`
            alpha (FloatTensor): `[B, H_mono, qlen, klen]`
            beta (FloatTensor): `[B, H_chunk, qlen, klen]`

        """
        bs, klen = key.size()[:2]
        qlen = query.size(1)

        aw_prev_bwd = None
        if aw_prev is None:
            # aw_prev = [1, 0, 0 ... 0]
            aw_prev = key.new_zeros(bs, self.n_heads_mono, 1, klen)
            aw_prev[:, :, :, 0:1] = key.new_ones(bs, self.n_heads_mono, 1, 1)
        if self.fwd_bwd and aw_prev_bwd is None:
            aw_prev_bwd = aw_prev.clone()

        # Compute monotonic energy
        e_mono = self.monotonic_energy(key, query, mask, cache=cache)  # `[B, H_mono, qlen, klen]`

        if mode == 'recursive':  # training
            p_choose = torch.sigmoid(add_gaussian_noise(e_mono, self.noise_std))  # `[B, H_mono, qlen, klen]`
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
                aw_prev = p_choose[:, :, i:i + 1] * q[:, :, i:i + 1, 1:]  # `[B, H_mono, 1, klen]`
                alpha.append(aw_prev)
            alpha = torch.cat(alpha, dim=2) if qlen > 1 else alpha[-1]  # `[B, H_mono, qlen, klen]`
            alpha = self.dropout(alpha)

        elif mode == 'parallel':  # training
            p_choose = torch.sigmoid(add_gaussian_noise(e_mono, self.noise_std))  # `[B, H_mono, qlen, klen]`
            # safe_cumprod computes cumprod in logspace with numeric checks
            cumprod_1mp_choose = safe_cumprod(1 - p_choose, eps=self.eps)  # `[B, H_mono, qlen, klen]`
            # Compute recurrence relation solution
            if self.simple:
                alpha = p_choose * cumprod_1mp_choose   # `[B, H_mono, qlen, klen]`
            else:
                alpha = []
                for i in range(qlen):
                    aw_prev = p_choose[:, :, i:i + 1] * cumprod_1mp_choose[:, :, i:i + 1] * torch.cumsum(
                        aw_prev / torch.clamp(cumprod_1mp_choose[:, :, i:i + 1], min=self.eps, max=1.0), dim=-1)  # `[B, H_mono, 1, klen]`

                    # Mask the right part from the trigger point
                    if self.decot and trigger_point is not None:
                        for b in range(bs):
                            aw_prev[b, :, :, trigger_point[b] + self.lookahead + 1:] = 0

                    alpha.append(aw_prev)
                alpha = torch.cat(alpha, dim=2) if qlen > 1 else alpha[-1]  # `[B, H_mono, qlen, klen]`
            alpha = self.dropout(alpha)

        elif mode == 'hard':  # inference
            # Attend when monotonic energy is above threshold (Sigmoid > 0.5)
            emit_probs = torch.sigmoid(e_mono)  # `[B, H_mono, qlen, klen]`
            p_choose = (emit_probs >= 0.5).float()
            # Remove any probabilities before the index chosen at the last time step
            p_choose *= torch.cumsum(aw_prev, dim=-1)  # `[B, H_mono, qlen, klen]`
            # Now, use exclusive cumprod to remove probabilities after the first
            # chosen index, like so:
            # p_choose                        = [0, 0, 0, 1, 1, 0, 1, 1]
            # 1 - p_choose                    = [1, 1, 1, 0, 0, 1, 0, 0]
            # exclusive_cumprod(1 - p_choose) = [1, 1, 1, 1, 0, 0, 0, 0]
            # alpha: product of above         = [0, 0, 0, 1, 0, 0, 0, 0]
            alpha = p_choose * exclusive_cumprod(1 - p_choose)  # `[B, H_mono, qlen, klen]`
        else:
            raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")

        if self.dropout_head > 0 and self.training and random.random() < self.dropout_head:
            head_idx = random.randint(0, self.n_heads_mono - 1)
            head_mask = alpha.new_ones(alpha.size()).byte()
            head_mask[:, head_idx] = 0
            alpha = alpha.masked_fill_(head_mask == 0, 0)

        # Compute chunk energy
        beta = None
        if self.chunk_size > 1 or self.milk:
            e_chunk = self.chunk_energy(key, query, mask, cache=cache)  # `[B, H_chunk, qlen, ken]`
            beta = efficient_chunkwise_attention(
                alpha, e_chunk, self.chunk_size,
                self.n_heads_chunk, self.sharpening_factor)  # `[B, H_mono * H_chunk, qlen, klen]`

        # Compute context vector
        if self.n_heads_mono * self.n_heads_chunk > 1:
            value = self.w_value(value).view(bs, -1, self.n_heads_mono * self.n_heads_chunk, self.d_k)
            value = value.transpose(2, 1).contiguous()  # `[B, H_mono * H_chunk, klen, d_k]`
            if self.chunk_size == 1:
                cv = torch.matmul(alpha, value)  # `[B, H_mono, qlen, d_k]`
            else:
                cv = torch.matmul(beta, value)  # `[B, H_mono * H_chunk, qlen, d_k]`
            cv = cv.transpose(2, 1).contiguous().view(bs, -1, self.n_heads_mono * self.n_heads_chunk * self.d_k)
            cv = self.w_out(cv)  # `[B, qlen, adim]`
        else:
            if self.chunk_size == 1:
                cv = torch.bmm(alpha.squeeze(1), value)  # `[B, 1, adim]`
            else:
                cv = torch.bmm(beta.squeeze(1), value)  # `[B, 1, adim]`

        assert alpha.size() == (bs, self.n_heads_mono, qlen, klen)
        if self.chunk_size > 1 or self.milk:
            assert beta.size() == (bs, self.n_heads_mono * self.n_heads_chunk, qlen, klen)
        return cv, alpha, beta


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
        x (FloatTensor): `[B, H_mono, H_chunk, qlen, klen]`
        back (int):
        forward (int):

    Returns:
        x_sum (FloatTensor): `[B, H_mono, H_chunk, qlen, klen]`

    """
    bs, n_heads_mono, n_heads_chunk, qlen, klen = x.size()
    x = x.view(-1, klen)
    # Moving sum is computed as a carefully-padded 1D convolution with ones
    x_padded = F.pad(x, pad=[back, forward])  # `[B * H_mono * H_chunk * qlen, back + klen + forward]`
    # Add a "channel" dimension
    x_padded = x_padded.unsqueeze(1)
    # Construct filters
    filters = x.new_ones(1, 1, back + forward + 1)
    x_sum = F.conv1d(x_padded, filters)
    x_sum = x_sum.squeeze(1).view(bs, n_heads_mono, n_heads_chunk, qlen, -1)
    return x_sum


def efficient_chunkwise_attention(alpha, e, chunk_size, n_heads, sharpening_factor):
    """Compute chunkwise attention distribution efficiently by clipping logits.

    Args:
        alpha (FloatTensor): `[B, H_mono, qlen, klen]`
        e (FloatTensor): `[B, H_chunk, qlen, klen]`
        chunk_size (int): window size for chunkwise attention
        n_heads (int): number of heads for chunkwise attention
        sharpening_factor (float):
    Return
        beta (FloatTensor): `[B, H_mono * H_chunk, qlen, klen]`

    """
    bs, _, qlen, klen = alpha.size()
    alpha = alpha.unsqueeze(2)
    e = e.unsqueeze(1)
    if n_heads > 1:
        alpha = alpha.repeat([1, 1, n_heads, 1, 1])
    # Shift logits to avoid overflow
    e -= torch.max(e, dim=-1, keepdim=True)[0]  # `[B, H_mono, H_chunk, qlen, klen]`
    # Limit the range for numerical stability
    softmax_exp = torch.clamp(torch.exp(e), min=1e-5)
    # Compute chunkwise softmax denominators
    if chunk_size == -1:
        # infinite lookback
        inner_items = alpha / (torch.cumsum(softmax_exp, dim=2))
        beta = softmax_exp * torch.cumsum(inner_items.flip(dims=[2]), dim=2).flip(dims=[2])
    else:
        softmax_denominators = moving_sum(softmax_exp,
                                          back=chunk_size - 1, forward=0)
        # Compute \beta_{i, :}. emit_probs are \alpha_{i, :}.
        beta = softmax_exp * moving_sum(alpha * sharpening_factor / softmax_denominators,
                                        back=0, forward=chunk_size - 1)
    return beta.view(bs, -1, qlen, klen)
