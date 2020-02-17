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
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.modules.causal_conv import CausalConv1d

NEG_INF = float(np.finfo(np.float32).min)

logger = logging.getLogger(__name__)


class MonotonicEnergy(nn.Module):
    def __init__(self, kdim, qdim, adim, atype,
                 init_r, conv1d=False, conv_kernel_size=5, param_init=''):
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
            param_init (str):

        """
        super().__init__()

        assert conv_kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        self.key = None
        self.mask = None

        self.atype = atype
        self.scale = math.sqrt(adim)

        self.w_key = nn.Linear(kdim, adim)
        if atype == 'add':
            self.v = nn.Linear(adim, 1, bias=False)
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
            energy (FloatTensor): `[B * qlen, klen]`

        """
        bs, klen, kdim = key.size()

        # Pre-computation of encoder-side features for computing scores
        if self.key is None or not cache:
            # 1d conv
            if self.conv1d is not None:
                key = torch.relu(self.conv1d(key))
            self.key = self.w_key(key)  # `[B, klen, adim]`
            self.mask = mask.view(-1, klen) if mask is not None else None  # `[B * qlen, klen]`

        if self.atype == 'add':
            energy = torch.relu(self.key + self.w_query(query))
            energy = self.v(energy).squeeze(2)  # `[B * 1, klen]`
        elif self.atype == 'scaled_dot':
            energy = torch.matmul(self.w_query(query.clone()),
                                  self.key.transpose(2, 1)) / self.scale
            energy = energy.view(-1, klen)  # `[B * qlen, klen]`

        if self.r is not None:
            energy = energy + self.r
        if self.mask is not None:
            energy = energy.masked_fill_(self.mask == 0, NEG_INF)
        return energy


class ChunkEnergy(nn.Module):
    def __init__(self, kdim, qdim, adim, atype, n_heads=1, param_init=''):
        """Energy function for the chunkwise attention.

        Args:
            kdim (int): dimension of key
            qdim (int): dimension of quary
            adim (int): dimension of attention space
            atype (str): type of attention mechanism
            n_heads (int): number of heads
            param_init (str):

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
            energy (FloatTensor): `[B * qlen * n_heads, klen]`

        """
        bs, klen, kdim = key.size()

        # Pre-computation of encoder-side features for computing scores
        if self.key is None or not cache:
            self.key = self.w_key(key)  # `[B, klen, adim]`
            self.mask = mask.view(-1, klen) if mask is not None else None  # `[B * qlen * n_heads, klen]`
            if self.mask is not None:
                if self.n_heads > 1:
                    self.mask = self.mask.repeat([self.n_heads, 1])
                assert self.mask.size() == (bs * query.size(1) * self.n_heads, klen)

        if self.atype == 'add':
            energy = torch.relu(self.key + self.w_query(query))  # `[B, klen, adim]`
            energy = self.v(energy).squeeze(2)  # `[B * 1 * 1, klen]`
        elif self.atype == 'scaled_dot':
            key = self.key.view(bs, -1, self.n_heads, self.d_k)
            key = key.transpose(2, 1).transpose(3, 2).contiguous()  # `[B, n_heads, d_k, klen]`
            query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)
            query = query.transpose(2, 1)  # `[B, n_heads, qlen, d_k]`
            energy = torch.matmul(query, key) / self.scale  # `[B, n_heads, qlen, klen]`
            energy = energy.transpose(2, 1).contiguous().view(-1, klen)  # `[B * qlen * n_heads, klen]`

        if self.mask is not None:
            energy = energy.masked_fill_(self.mask == 0, NEG_INF)
        return energy


class MoChA(nn.Module):
    def __init__(self, kdim, qdim, adim, atype, chunk_size, n_heads=1,
                 conv1d=False, init_r=-4, noise_std=1.0, eps=1e-6,
                 sharpening_factor=1.0, param_init=''):
        """Monotonic chunk-wise attention.

            "Monotonic Chunkwise Attention" (ICLR 2018)
            https://openreview.net/forum?id=Hko85plCW

            if chunk_size == 1, this is equivalent to Hard monotonic attention
                "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
                 http://arxiv.org/abs/1704.00784

        Args:
            kdim (int): dimension of key
            qdim (int): dimension of query
            adim: (int) dimension of the attention layer
            atype (str): type of attention mechanism
            chunk_size (int): size of chunk
            n_heads (int): number of heads
            conv1d (bool): apply 1d convolution for energy calculation
            init_r (int): initial value for parameter 'r' used for monotonic attention
            noise_std (float): standard deviation for input noise
            eps (float):
            sharpening_factor (float): sharping factor for beta calculation
            param_init (str):

        """
        super(MoChA, self).__init__()

        self.atype = atype
        assert adim % n_heads == 0
        self.d_k = adim // n_heads

        self.chunk_size = chunk_size
        self.n_heads = n_heads
        self.noise_std = noise_std
        self.eps = eps
        self.sharpening_factor = sharpening_factor

        self.monotonic_energy = MonotonicEnergy(kdim, qdim, adim, atype, init_r, conv1d,
                                                param_init=param_init)
        self.chunk_energy = ChunkEnergy(kdim, qdim, adim, atype,
                                        n_heads, param_init) if chunk_size > 1 else None
        if n_heads > 1:
            self.w_value = nn.Linear(kdim, adim)
            self.w_out = nn.Linear(adim, kdim)
            if param_init == 'xavier_uniform':
                self.reset_parameters(True)

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
        if self.chunk_size > 1:
            self.chunk_energy.reset()

    def forward(self, key, value, query, mask=None, aw_prev=None,
                mode='hard', cache=True, trigger_point=None):
        """Soft monotonic attention during training.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            value (FloatTensor): `[B, klen, vdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            aw_prev (FloatTensor): `[B, klen, 1]`
            mode (str): recursive/parallel/hard
            cache (bool): cache key and mask
            trigger_point (IntTensor): `[B]`
        Return:
            cv (FloatTensor): `[B, qlen, vdim]`
            alpha (FloatTensor): `[B, klen, qlen]`
            beta (FloatTensor): `[B, n_heads, klen, qlen]`

        """
        bs, klen = key.size()[:2]
        qlen = query.size(1)

        if aw_prev is None:
            # aw_prev = [1, 0, 0 ... 0]
            aw_prev = key.new_zeros(bs, klen)
            aw_prev[:, 0:1] = key.new_ones(bs, 1)
        else:
            aw_prev = aw_prev.squeeze(2)

        # Compute monotonic energy
        e_mono = self.monotonic_energy(key, query, mask, cache=cache)  # `[B * qlen, klen]`

        if mode == 'recursive':  # training
            assert qlen == 1
            p_choose = torch.sigmoid(add_gaussian_noise(e_mono, self.noise_std))  # `[B * qlen, klen]`
            # Compute [1, 1 - p_choose[0], 1 - p_choose[1], ..., 1 - p_choose[-2]]
            shifted_1mp_choose = torch.cat([key.new_ones(bs, 1), 1 - p_choose[:, :-1]], dim=1)
            # Compute attention distribution recursively as
            # q_j = (1 - p_choose_j) * q_(j-1) + aw_prev_j
            # alpha_j = p_choose_j * q_j
            q = key.new_zeros(bs, klen + 1)
            for j in range(klen):
                q[:, j + 1] = shifted_1mp_choose[:, j].clone() * q[:, j].clone() + aw_prev[:, j].clone()
            alpha = p_choose * q[:, 1:]  # `[B, klen]`
            alpha = alpha.unsqueeze(1)  # `[B, 1, klen]`

        elif mode == 'parallel':  # training
            p_choose = torch.sigmoid(add_gaussian_noise(e_mono, self.noise_std))  # `[B * qlen, klen]`
            # safe_cumprod computes cumprod in logspace with numeric checks
            cumprod_1mp_choose = safe_cumprod(1 - p_choose, eps=self.eps)  # `[B * qlen, klen]`
            # Compute recurrence relation solution
            if self.atype == 'add':
                assert qlen == 1
                alpha = p_choose * cumprod_1mp_choose * torch.cumsum(
                    aw_prev / torch.clamp(cumprod_1mp_choose, min=self.eps, max=1.0), dim=1)  # `[B, klen]`
                alpha = alpha.unsqueeze(1)  # `[B, 1, klen]`

                # Mask the right part from the trigger point
                if trigger_point is not None:
                    for b in range(bs):
                        alpha[b, :, trigger_point[b] + 1:] = 0
                        # TODO(hirofumi): add tolerance parameter
            elif self.atype == 'scaled_dot':
                p_choose = p_choose.view(bs, qlen, klen)
                cumprod_1mp_choose = cumprod_1mp_choose.view(bs, qlen, klen)

                # parallel version
                # alpha = p_choose * cumprod_1mp_choose   # `[B, qlen, klen]`

                alpha = []
                for i in range(query.size(1)):
                    p_choose_i = p_choose[:, i]
                    cumprod_1mp_choose_i = cumprod_1mp_choose[:, i]
                    aw_prev = p_choose_i * cumprod_1mp_choose_i * torch.cumsum(
                        aw_prev / torch.clamp(cumprod_1mp_choose_i, min=self.eps, max=1.0), dim=1)  # `[B, klen]`
                    alpha.append(aw_prev.unsqueeze(1))
                alpha = torch.cat(alpha, dim=1)  # `[B, qlen, klen]`

        elif mode == 'hard':  # inference
            # Attend when monotonic energy is above threshold (Sigmoid > 0.5)
            emit_probs = torch.sigmoid(e_mono)  # `[B, klen]`
            p_choose = (emit_probs >= 0.5).float()
            # Remove any probabilities before the index chosen at the last time step
            p_choose *= torch.cumsum(aw_prev, dim=1)  # `[B, klen]`
            # Now, use exclusive cumprod to remove probabilities after the first
            # chosen index, like so:
            # p_choose                        = [0, 0, 0, 1, 1, 0, 1, 1]
            # 1 - p_choose                    = [1, 1, 1, 0, 0, 1, 0, 0]
            # exclusive_cumprod(1 - p_choose) = [1, 1, 1, 1, 0, 0, 0, 0]
            # alpha: product of above         = [0, 0, 0, 1, 0, 0, 0, 0]
            alpha = p_choose * exclusive_cumprod(1 - p_choose)  # `[B, klen]`
            alpha = alpha.unsqueeze(1)  # `[B, 1, klen]`
        else:
            raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")

        # Compute chunk energy
        beta = None
        if self.chunk_size > 1:
            e_chunk = self.chunk_energy(key, query, mask, cache=cache)  # `[B * qlen * n_heads, klen]`
            beta = efficient_chunkwise_attention(alpha.view(-1, klen), e_chunk, self.chunk_size,
                                                 self.n_heads, self.sharpening_factor)  # [B * qlen * n_heads, klen]`

        # Compute context vector
        if self.chunk_size == 1:
            cv = torch.bmm(alpha, value)
        else:
            if self.n_heads > 1:
                value = self.w_value(value).view(bs, -1, self.n_heads, self.d_k)
                value = value.transpose(2, 1).contiguous()  # `[B, n_heads, klen, d_k]`
                beta = beta.view(bs, qlen, self.n_heads, klen)
                beta = beta.transpose(2, 1).contiguous()  # `[B, n_heads, qlen, klen]`
                cv = torch.matmul(beta, value)  # `[B, n_heads, qlen, d_k]`
                cv = cv.transpose(2, 1).contiguous().view(bs, -1, self.n_heads * self.d_k)
                cv = self.w_out(cv)
            else:
                beta = beta.unsqueeze(1)  # `[B, 1, klen]`
                cv = torch.bmm(beta, value)  # `[B, 1, d_k]`
                beta = beta.unsqueeze(3)  # `[B, 1, klen, 1]`

        return cv, alpha.transpose(2, 1), beta


def add_gaussian_noise(xs, std):
    """Additive gaussian nosie to encourage discreteness."""
    noise = xs.new_zeros(xs.size()).normal_(std=std)
    return xs + noise


def safe_cumprod(x, eps):
    """Numerically stable cumulative product by cumulative sum in log-space.
        Args:
            x (FloatTensor): `[B, klen]`
        Returns:
            x (FloatTensor): `[B, klen]`

    """
    return torch.exp(exclusive_cumsum(torch.log(torch.clamp(x, min=eps, max=1.0))))


def exclusive_cumsum(x):
    """Exclusive cumulative summation [a, b, c] => [0, a, a + b].

        Args:
            x (FloatTensor): `[B, klen]`
        Returns:
            x (FloatTensor): `[B, klen]`

    """
    return torch.cumsum(torch.cat([x.new_zeros(x.size(0), 1), x[:, :-1]], dim=1), dim=1)


def exclusive_cumprod(x):
    """Exclusive cumulative product [a, b, c] => [1, a, a * b].

        Args:
            x (FloatTensor): `[B, klen]`
        Returns:
            x (FloatTensor): `[B, klen]`

    """
    return torch.cumprod(torch.cat([x.new_ones(x.size(0), 1), x[:, :-1]], dim=1), dim=1)


def moving_sum(x, back, forward):
    """Compute the moving sum of x over a chunk_size with the provided bounds.

    Args:
        x (FloatTensor): `[B, klen]`
        back (int):
        forward (int):

    Returns:
        x_sum (FloatTensor): `[B, klen]`

    """
    # Moving sum is computed as a carefully-padded 1D convolution with ones
    x_padded = F.pad(x, pad=[back, forward]).unsqueeze(1)  # `[B, 1, back + T + forward]`
    # Construct filters
    filters = x.new_ones(1, 1, back + forward + 1)
    x_sum = F.conv1d(x_padded, filters)
    return x_sum.squeeze(1)


def efficient_chunkwise_attention(alpha, e, chunk_size, n_heads, sharpening_factor):
    """Compute chunkwise attention distribution efficiently by clipping logits.

    Args:
        alpha (FloatTensor): `[B * qlen, klen]`
        e (FloatTensor): `[B * qlen * n_heads, klen]`
        chunk_size (int): size of chunk
    Return
        beta (FloatTensor): `[B * qlen * n_heads, klen]`

    """
    if n_heads > 1:
        alpha = alpha.repeat([n_heads, 1])
    # Shift logits to avoid overflow
    e -= torch.max(e, dim=1, keepdim=True)[0]
    # Limit the range for numerical stability
    softmax_exp = torch.clamp(torch.exp(e), min=1e-5)
    # Compute chunkwise softmax denominators
    softmax_denominators = moving_sum(softmax_exp,
                                      back=chunk_size - 1, forward=0)
    # Compute \beta_{i, :}. emit_probs are \alpha_{i, :}.
    beta = softmax_exp * moving_sum(alpha * sharpening_factor / softmax_denominators,
                                    back=0, forward=chunk_size - 1)
    return beta
