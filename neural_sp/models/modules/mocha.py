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
    def __init__(self, key_dim, query_dim, attn_dim, window=1, init_r=-4):
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
            mode (str): parallel/hard
        Return:
            cv (FloatTensor): `[B, 1, value_dim]`
            aw_prev (FloatTensor): `[bs, kmax, 1 (n_heads)]`

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
                raise NotImplementedError

        # Compute monotonic energy
        query = query.repeat([1, kmax, 1])
        e_mono = self.v_mono(torch.tanh(self.key + self.w_query_mono(query)))
        e_mono = e_mono + self.r_mono  # `[B, kmax, 1 (n_heads)]`
        e_mono = e_mono.squeeze(2)  # `[B, kmax]`
        if self.mask is not None:
            e_mono = e_mono.masked_fill_(self.mask == 0, NEG_INF)

        if mode == 'recursive':  # training time
            p_choose = torch.sigmoid(add_gaussian_noise(e_mono))  # `[B, kmax]`
            raise NotImplementedError

            # Compute [1, 1 - p_choose[0], 1 - p_choose[1], ..., 1 - p_choose[-2]]
            # shifted_1_minus_p_choose = torch.cat([key.new_ones(bs, 1), 1 - p_choose[:, :-1]], dim=1)
            # # Compute attention distribution recursively as
            # # q[i] = (1 - p_choose[i])*q[i - 1] + aw_prev[i]
            # # aw[i] = p_choose[i]*q[i]
            # aw = p_choose * tf.transpose(tf.scan(
            #     # Need to use reshape to remind TF of the shape between loop iterations
            #     lambda x, yz: tf.reshape(yz[0] * x + yz[1], (bs,)),
            #     # Loop variables yz[0] and yz[1]
            #     [shifted_1_minus_p_choose.transpose(1, 0), aw_prev.squeeze(2).transpose(1, 0)],
            #     # Initial value of x is just zeros
            #     key.new_zeros(bs,)))

        elif mode == 'parallel':  # training time
            p_choose = torch.sigmoid(add_gaussian_noise(e_mono))  # `[B, kmax]`
            # safe_cumprod computes cumprod in logspace with numeric checks
            cumprod_1_minus_p_choose = safe_cumprod(1 - p_choose, eps=1e-10)
            # Compute recurrence relation solution
            aw = p_choose * cumprod_1_minus_p_choose * torch.cumsum(
                aw_prev.squeeze(2) / torch.clamp(cumprod_1_minus_p_choose, min=1e-10, max=1.0), dim=1)

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
        else:
            raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")

        # Compute chunk energy
        if self.window > 1:
            raise NotImplementedError

        # Compute context vector
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
    """Exclusive cumulative product [a, b, c] => [1, a, a * b]

    * TensorFlow: https://www.tensorflow.org/api_docs/python/tf/cumprod
    * PyTorch: https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614

    """
    assert len(xs.size()) == 2
    return torch.cumprod(torch.cat([xs.new_ones(xs.size(0), 1), xs], dim=1)[:, :-1], dim=1)
