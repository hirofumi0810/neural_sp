#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# [reference]
# https://github.com/j-min/MoChA-pytorch/blob/94b54a7fa13e4ac6dc255b509dd0febc8c0a0ee6/attention.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.modules.linear import Linear

NEG_INF = float(np.finfo(np.float32).min)


class Energy(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_type, attn_dim, init_r,
                 conv_out_channels=10, conv_kernel_size=100):
        """Energy function."""
        super().__init__()

        self.attn_type = attn_type
        self.key = None
        self.mask = None

        self.w_key = Linear(enc_dim, attn_dim)
        self.w_query = Linear(dec_dim, attn_dim, bias=False)
        if attn_type == 'location':
            self.w_conv = Linear(conv_out_channels, attn_dim, bias=False)
            self.conv = nn.Conv1d(in_channels=1,
                                  out_channels=conv_out_channels,
                                  kernel_size=conv_kernel_size * 2 + 1,
                                  stride=1,
                                  padding=conv_kernel_size,
                                  bias=False)
        else:
            assert attn_type == 'add'

        self.v = nn.utils.weight_norm(nn.Linear(attn_dim, 1))
        self.r = nn.Parameter(torch.Tensor([init_r]))

        # initialization
        self.v.weight_g.data = torch.Tensor([1 / attn_dim]).sqrt()

    def reset(self):
        self.key = None
        self.mask = None

    def forward(self, key, query, mask, aw_prev=None):
        """Compute energy.

        Args:
            key (FloatTensor): `[B, kmax, key_dim]`
            query (FloatTensor): `[B, 1, query_dim]`
            mask (ByteTensor): `[B, qmax, kmax]`
            aw_prev (FloatTensor): `[B, kmax, 1 (n_heads)]`
        Return:
            energy (FloatTensor): `[B, value_dim]`

        """
        bs, kmax, key_dim = key.size()

        # Pre-computation of encoder-side features for computing scores
        if self.key is None:
            self.key = self.w_key(key)
            self.mask = mask

        key = key.view(-1, key_dim)
        tmp = self.key + self.w_query(query)
        if self.attn_type == 'location':
            conv_feat = self.conv(aw_prev.transpose(2, 1))  # `[B, ch, kmax]`
            conv_feat = conv_feat.transpose(2, 1).contiguous()  # `[B, kmax, ch]`
            tmp = tmp + self.w_conv(conv_feat)
        energy = torch.tanh(tmp)
        energy = self.v(energy).squeeze(-1) + self.r  # `[B, kmax]`
        if self.mask is not None:
            energy = energy.masked_fill_(self.mask == 0, NEG_INF)
        return energy


class MoChA(nn.Module):
    def __init__(self, key_dim, query_dim, attn_dim, chunk_size, init_r=-4,
                 adaptive=False):
        """Monotonic chunk-wise attention.

            "Monotonic Chunkwise Attention" (ICLR 2018)
            https://openreview.net/forum?id=Hko85plCW

            if chunk_size == 1, this is equivalent to Hard monotonic attention
                "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
                 http://arxiv.org/abs/1704.00784

        Args:
            key_dim (int): dimension of key
            query_dim (int): dimension of query
            attn_dim: (int) dimension of the attention layer
            chunk_size (int): size of chunk
            init_r (int): initial value for parameter 'r' used in monotonic/chunk attention
            adaptive (bool): turn on adaptive MoChA

        """
        super(MoChA, self).__init__()

        self.chunk_size = chunk_size
        self.adaptive = adaptive
        self.constrained = True
        if chunk_size >= 100:
            self.constrained = False
        self.n_heads = 1

        # Monotonic energy
        self.monotonic_energy = Energy(key_dim, query_dim, 'add', attn_dim, init_r)

        # Chunk energy
        self.chunk_energy = None
        if chunk_size > 1:
            self.chunk_energy = Energy(key_dim, query_dim, 'add', attn_dim, init_r)

        self.chunk_len_energy = None
        if adaptive:
            assert chunk_size > 1
            self.chunk_len_energy = Energy(key_dim, query_dim, 'add', attn_dim, init_r)

        self.tail_prediction = False

    def reset(self):
        self.tail_prediction = False
        self.monotonic_energy.reset()
        if self.chunk_size > 1:
            self.chunk_energy.reset()
            if self.adaptive:
                self.chunk_len_energy.reset()

    def forward(self, key, value, query, mask=None, aw_prev=None, mode='parallel',
                eps=1e-8):
        """Soft monotonic attention during training.

        Args:
            key (FloatTensor): `[B, kmax, key_dim]`
            value (FloatTensor): `[B, kmax, value_dim]`
            query (FloatTensor): `[B, 1, query_dim]`
            mask (ByteTensor): `[B, qmax, kmax]`
            aw_prev (FloatTensor): `[B, kmax, 1 (n_heads)]`
            mode (str): recursive/parallel/hard
        Return:
            cv (FloatTensor): `[B, 1, value_dim]`
            aw_prev (FloatTensor): `[B, kmax, 1 (n_heads)]`

        """
        bs, kmax = key.size()[:2]

        if aw_prev is None:
            # aw_prev = [1, 0, 0 ... 0]
            aw_prev = key.new_zeros(bs, kmax, 1)
            aw_prev[:, 0:1] = key.new_ones(bs, 1, 1)

        # Compute monotonic energy
        e_mono = self.monotonic_energy(key, query, mask)

        if mode == 'recursive':  # training
            p_choose_i = torch.sigmoid(add_gaussian_noise(e_mono))  # `[B, kmax]`
            # Compute [1, 1 - p_choose_i[0], 1 - p_choose_i[1], ..., 1 - p_choose_i[-2]]
            shifted_1mp_choose_i = torch.cat([key.new_ones(bs, 1),
                                              1 - p_choose_i[:, :-1]], dim=1)
            # Compute attention distribution recursively as
            # q[j] = (1 - p_choose_i[j]) * q[j - 1] + aw_prev[j]
            # alpha[j] = p_choose_i[j] * q[j]
            q = key.new_zeros(bs, kmax + 1)
            for j in range(kmax):
                q[:, j + 1] = shifted_1mp_choose_i[:, j].clone() * q[:, j].clone() + aw_prev[:, j, 0].clone()
            alpha = p_choose_i * q[:, 1:]

        elif mode == 'parallel':  # training
            p_choose_i = torch.sigmoid(add_gaussian_noise(e_mono))  # `[B, kmax]`
            # safe_cumprod computes cumprod in logspace with numeric checks
            cumprod_1mp_choose_i = safe_cumprod(1 - p_choose_i, eps=eps)
            # Compute recurrence relation solution
            alpha = p_choose_i * cumprod_1mp_choose_i * torch.cumsum(
                aw_prev.squeeze(2) / torch.clamp(cumprod_1mp_choose_i, min=eps, max=1.0), dim=1)
            # alpha = p_choose_i * cumprod_1mp_choose_i * torch.cumsum(aw_prev.squeeze(2), dim=1)

        elif mode == 'hard':  # test
            # Attend when monotonic energy is above threshold (Sigmoid > 0.5)
            emit_probs = torch.sigmoid(e_mono)
            p_choose_i = (emit_probs > 0.5).float()
            # Remove any probabilities before the index chosen at the last time step
            p_choose_i *= torch.cumsum(aw_prev.squeeze(2), dim=1)  # `[B, kmax]`
            # Now, use exclusive cumprod to remove probabilities after the first
            # chosen index, like so:
            # p_choose_i                        = [0, 0, 0, 1, 1, 0, 1, 1]
            # 1 - p_choose_i                    = [1, 1, 1, 0, 0, 1, 0, 0]
            # exclusive_cumprod(1 - p_choose_i) = [1, 1, 1, 1, 0, 0, 0, 0]
            # alpha: product of above           = [0, 0, 0, 1, 0, 0, 0, 0]
            alpha = p_choose_i * exclusive_cumprod(1 - p_choose_i)

            if not self.tail_prediction and alpha.sum() == 0:
                emit_probs *= torch.cumsum(aw_prev.squeeze(2), dim=1)  # `[B, kmax]`
                triggered_point = torch.argmax(emit_probs, dim=1)
                alpha = e_mono.new_zeros(e_mono.size())
                for b in range(bs):
                    alpha[b, triggered_point[b]] = 1
                self.tail_prediction = True
        else:
            raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")

        # Compute chunk energy
        if self.chunk_size > 1:
            e_chunk = self.chunk_energy(key, query, mask)
            if self.adaptive:
                e_chunk_len = self.chunk_len_energy(key, query, mask)
                if self.constrained:
                    chunk_len_dist = self.chunk_size * torch.sigmoid(e_chunk_len)
                else:
                    chunk_len_dist = torch.exp(e_chunk_len)
                # avoid zero length
                chunk_len_dist = chunk_len_dist.int() + 1
                beta = efficient_adaptive_chunkwise_attention(alpha, e_chunk, chunk_len_dist)
            else:
                beta = efficient_chunkwise_attention(alpha, e_chunk, self.chunk_size)
            # alpha_norm = alpha / torch.sum(alpha, dim=1, keepdim=True)
            # beta = efficient_chunkwise_attention(alpha_norm, e_chunk, self.chunk_size)

        # Compute context vector
        if self.chunk_size > 1:
            cv = torch.bmm(beta.unsqueeze(1), value)
        else:
            cv = torch.bmm(alpha.unsqueeze(1), value)

        return cv, alpha.unsqueeze(2)


def add_gaussian_noise(xs, std=1.0):
    """Additive gaussian nosie to encourage discreteness."""
    noise = xs.new_zeros(xs.size()).normal_(std=std)
    return xs + noise


def safe_cumprod(x, eps=1e-8):
    """Numerically stable cumulative product by cumulative sum in log-space."""
    return torch.exp(exclusive_cumsum(torch.log(torch.clamp(x, min=eps, max=1.0))))


def exclusive_cumsum(x):
    """Exclusive cumulative summation [a, b, c] => [0, a, a + b]"""
    return torch.cumsum(torch.cat([x.new_zeros(x.size(0), 1), x[:, :-1]], dim=1), dim=1)


def exclusive_cumprod(x):
    """Exclusive cumulative product [a, b, c] => [1, a, a * b].

    * TensorFlow: https://www.tensorflow.org/api_docs/python/tf/cumprod
    * PyTorch: https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614

    """
    return torch.cumprod(torch.cat([x.new_ones(x.size(0), 1), x[:, :-1]], dim=1), dim=1)


def moving_sum(x, back, forward):
    """Compute the moving sum of x over a chunk_size with the provided bounds.

    Args:
        x (FloatTensor): `[B, T]`
        back (int):
        forward (int):

    Returns:
        x_sum (FloatTensor): `[B, T]`
    """
    # Moving sum is computed as a carefully-padded 1D convolution with ones
    x_padded = F.pad(x, pad=[back, forward])
    # Add a "channel" dimension
    x_padded = x_padded.unsqueeze(1)  # `[B, 1, back + T + forward]`
    # Construct filters
    filters = x.new_ones(1, 1, back + forward + 1)
    x_sum = F.conv1d(x_padded, filters)
    # Remove channel dimension
    return x_sum.squeeze(1)


def efficient_chunkwise_attention(alpha, u, chunk_size):
    """Compute chunkwise attention distribution efficiently by clipping logits.

    Args:
        alpha (FloatTensor): emission probability in monotonic attention, `[B, T]`
        u (FloatTensor): chunk energy, `[B, T]`
        chunk_size (int): size of chunk
    Return
        beta (FloatTensor): MoChA weights, `[B, T]`

    """
    # Shift logits to avoid overflow
    u -= torch.max(u, dim=1, keepdim=True)[0]
    # Limit the range for numerical stability
    softmax_exp = torch.clamp(torch.exp(u), min=1e-5)
    # Compute chunkwise softmax denominators
    softmax_denominators = moving_sum(softmax_exp,
                                      back=chunk_size - 1, forward=0)
    # Compute \beta_{i, :}. emit_probs are \alpha_{i, :}.
    beta = softmax_exp * moving_sum(alpha / softmax_denominators,
                                    back=0, forward=chunk_size - 1)
    return beta


def efficient_adaptive_chunkwise_attention(alpha, u, chunk_len_dist):
    """Compute adaptive chunkwise attention distribution efficiently by clipping logits.

    Args:
        alpha (FloatTensor): emission probability in monotonic attention, `[B, T]`
        u (FloatTensor): chunk energy, `[B, T]`
        chunk_len_dist (IntTensor): `[B, T]`
    Return
        beta (FloatTensor): MoChA weights, `[B, T]`

    """
    # Shift logits to avoid overflow
    u -= torch.max(u, dim=1, keepdim=True)[0]
    # Limit the range for numerical stability
    softmax_exp = torch.clamp(torch.exp(u), min=1e-5)
    # Compute chunkwise softmax denominators
    triggered_points = torch.argmax(alpha, dim=1)
    bs = alpha.size(0)
    softmax_denominators = [moving_sum(
        softmax_exp[b:b + 1],
        back=chunk_len_dist[b, triggered_points[b]] - 1, forward=0)
        for b in range(bs)]
    # Compute \beta_{i, :}. emit_probs are \alpha_{i, :}.
    beta = [softmax_exp[b:b + 1] * moving_sum(
        alpha[b:b + 1] / softmax_denominators[b],
        back=0, forward=chunk_len_dist[b, triggered_points[b]] - 1)
        for b in range(bs)]
    beta = torch.cat(beta, dim=0)
    return beta
