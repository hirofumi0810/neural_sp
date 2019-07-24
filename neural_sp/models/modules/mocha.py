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


class MoChA(nn.Module):
    def __init__(self, key_dim, query_dim, attn_dim, window, init_r=-4):
        """Monotonic chunk-wise attention.

            "Monotonic Chunkwise Attention" (ICLR 2018)
            https://openreview.net/forum?id=Hko85plCW

            if window == 1, this is equivalent to Hard monotonic attention
                "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
                 http://arxiv.org/abs/1704.00784

        Args:
            key_dim (int): dimensions of key
            query_dim (int): dimensions of query
            attn_dim: (int) dimension of the attention layer
            window (int): chunk size
            init_r (int): initial value for parameter 'r' used in monotonic/chunk attention

        """
        super(MoChA, self).__init__()

        self.window = window
        self.n_heads = 1

        # Monotonic energy
        self.w_key_mono = Linear(key_dim, attn_dim, bias=True)
        self.w_query_mono = Linear(query_dim, attn_dim, bias=False)
        self.v_mono = Linear(attn_dim, 1, bias=False, weight_norm=True)
        self.r_mono = nn.Parameter(torch.Tensor([init_r]))

        # Chunk energy
        if window > 1:
            self.w_key_chunk = Linear(key_dim, attn_dim, bias=True)
            self.w_query_chunk = Linear(query_dim, attn_dim, bias=False)
            self.v_chunk = Linear(attn_dim, 1, bias=False, weight_norm=True)
            self.r_chunk = nn.Parameter(torch.Tensor([init_r]))

        # initialization
        self.v_mono.fc.weight_g.data = torch.Tensor([1 / attn_dim]).sqrt()
        if window > 1:
            self.v_mono.fc.weight_g.data = torch.Tensor([1 / attn_dim]).sqrt()

    def reset(self):
        self.key = None
        self.key_chunk = None
        self.mask = None

    def forward(self, key, value, query, mask=None, aw_prev=None, mode='parallel'):
        """Soft monotonic attention during training.

        Args:
            key (FloatTensor): `[B, kmax, key_dim]`
            klens (IntTensor): `[B]`
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

        # Pre-computation of encoder-side features for computing scores
        if self.key is None:
            self.key = self.w_key_mono(key)
            self.mask = mask
            if self.window > 1:
                self.key_chunk = self.w_key_chunk(key)

        # Compute monotonic energy
        query = query.repeat([1, kmax, 1])
        e_mono = self.v_mono(torch.tanh(self.key + self.w_query_mono(query)))
        e_mono = e_mono + self.r_mono  # `[B, kmax, 1 (n_heads)]`
        e_mono = e_mono.squeeze(2)  # `[B, kmax]`
        if self.mask is not None:
            e_mono = e_mono.masked_fill_(self.mask == 0, NEG_INF)

        if mode == 'recursive':  # training time
            raise NotImplementedError

        elif mode == 'parallel':  # training time
            p_choose = torch.sigmoid(add_gaussian_noise(e_mono))  # `[B, kmax]`
            # safe_cumprod computes cumprod in logspace with numeric checks
            cumprod_1_minus_p_choose = safe_cumprod(1 - p_choose, eps=1e-10)
            # Compute recurrence relation solution
            aw = p_choose * cumprod_1_minus_p_choose * torch.cumsum(
                aw_prev.squeeze(2) / torch.clamp(cumprod_1_minus_p_choose, min=1e-10, max=1.0), dim=1)

            # Compute chunk energy
            if self.window > 1:
                e_chunk = self.v_chunk(torch.tanh(self.key_chunk + self.w_query_chunk(query)))
                e_chunk = e_chunk + self.r_chunk  # `[B, kmax, 1 (n_heads)]`
                e_chunk = e_chunk.squeeze(2)  # `[B, kmax]`
                if self.mask is not None:
                    e_chunk = e_chunk.masked_fill_(self.mask == 0, NEG_INF)
                beta = efficient_chunkwise_attention(aw, e_chunk, self.window)

        elif mode == 'hard':  # test time
            # Attend when monotonic energy is above threshold (Sigmoid > 0.5)
            p_choose = (e_mono > 0).float()
            # Remove any probabilities before the index chosen last time step
            p_choose *= torch.cumsum(aw_prev.squeeze(2), dim=1)  # `[B, kmax]`

            # Now, use exclusive cumprod to remove probabilities after the first
            # chosen index, like so:
            # p_choose                        = [0, 0, 0, 1, 1, 0, 1, 1]
            # 1 - p_choose                    = [1, 1, 1, 0, 0, 1, 0, 0]
            # exclusive_cumprod(1 - p_choose) = [1, 1, 1, 1, 0, 0, 0, 0]
            # aw: product of above            = [0, 0, 0, 1, 0, 0, 0, 0]
            aw = p_choose * exclusive_cumprod(1 - p_choose)
            # Not attended => attend at last encoder output
            # NOTE: Assume that encoder outputs are not padded
            attended = aw.sum(dim=1)
            for i_b in range(bs):
                if attended[i_b] == 0:
                    aw[i_b, -1] = 1
            # Original paperによるとzero vector
        else:
            raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")

        # Compute context vector
        if self.window > 1:
            cv = torch.bmm(beta.unsqueeze(1), value)
        else:
            cv = torch.bmm(aw.unsqueeze(1), value)

        return cv, aw.unsqueeze(2)


def add_gaussian_noise(xs, std=1.0):
    """Additive gaussian nosie to encourage discreteness."""
    noise = xs.new_zeros(xs.size()).normal_(std=std)
    return xs + noise


def safe_cumprod(xs, eps=1e-10):
    """Numerically stable cumulative product by cumulative sum in log-space."""
    return torch.exp(exclusive_cumsum(torch.log(torch.clamp(xs, min=eps, max=1))))


def exclusive_cumsum(xs):
    """Exclusive cumulative summation [a, b, c] => [1, a, a + b]"""
    assert len(xs.size()) == 2
    return torch.cumsum(torch.cat([xs.new_ones(xs.size(0), 1), xs], dim=1)[:, :-1], dim=1)


def exclusive_cumprod(xs):
    """Exclusive cumulative product [a, b, c] => [1, a, a * b].

    * TensorFlow: https://www.tensorflow.org/api_docs/python/tf/cumprod
    * PyTorch: https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614

    """
    assert len(xs.size()) == 2
    return torch.cumprod(torch.cat([xs.new_ones(xs.size(0), 1), xs], dim=1)[:, :-1], dim=1)


def moving_sum(x, back, forward):
    """Compute the moving sum of x over a window with the provided bounds.

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
    """Compute chunkwise attention distribution efficiently by clipping logits (training).

    Args:
        alpha (FloatTensor): emission probability in monotonic attention, `[B, T]`
        u (FloatTensor): chunk energy, `[B, T]`
        chunk_size (int): window size of chunk
    Return
        beta (FloatTensor): MoChA weights, `[B, T]`

    """
    # Shift logits to avoid overflow
    u -= torch.max(u, dim=1, keepdim=True)[0]
    # Limit the range for numerical stability
    softmax_exp = torch.clamp(torch.exp(u), min=1e-5)
    # Compute chunkwise softmax denominators
    softmax_denominators = moving_sum(softmax_exp, back=chunk_size - 1, forward=0)
    # Compute \beta_{i, :}. emit_probs are \alpha_{i, :}.
    beta = softmax_exp * moving_sum(alpha / softmax_denominators, back=0, forward=chunk_size - 1)
    return beta
