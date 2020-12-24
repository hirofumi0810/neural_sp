# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Monotonic (multihead) chunkwise attention."""

# [reference]
# https://github.com/j-min/MoChA-pytorch/blob/94b54a7fa13e4ac6dc255b509dd0febc8c0a0ee6/attention.py

import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.modules.causal_conv import CausalConv1d
from neural_sp.models.modules.headdrop import headdrop


logger = logging.getLogger(__name__)


class MonotonicEnergy(nn.Module):
    """Energy function for the monotonic attention.

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

    def __init__(self, kdim, qdim, adim, atype, n_heads, init_r,
                 bias=True, param_init='', conv1d=False, conv_kernel_size=5):

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
                                       kernel_size=conv_kernel_size,
                                       param_init=param_init)
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

    def reset(self):
        self.key = None
        self.mask = None

    def forward(self, key, query, mask, cache=False,
                boundary_leftmost=0):
        """Compute monotonic energy.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            cache (bool): cache key and mask
            boundary_leftmost (int): leftmost boundary position
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
            self.key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)  # `[B, klen, H_ma, d_k]`
            self.mask = mask
            if mask is not None:
                self.mask = self.mask.unsqueeze(3).repeat([1, 1, 1, self.n_heads])  # `[B, qlen, klen, H_ca]`
                mask_size = (bs, qlen, klen, self.n_heads)
                assert self.mask.size() == mask_size, (self.mask.size(), mask_size)

        key = self.key
        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)
        m = self.mask

        # Truncate encoder memories for efficient decoding
        if boundary_leftmost > 0:
            key = key[:, boundary_leftmost:]
            klen = key.size(1)
            if m is not None:
                m = m[:, :, boundary_leftmost:]

        if self.atype == 'scaled_dot':
            e = torch.einsum("bihd,bjhd->bijh", (query, key)) / self.scale
        elif self.atype == 'add':
            e = self.v(torch.relu(key[:, None] + query[:, :, None]).view(bs, qlen, klen, -1))
        # e: `[B, qlen, klen, H_ma]`

        if self.r is not None:
            e = e + self.r
        if m is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=e.dtype).numpy().dtype).min)
            e = e.masked_fill_(m == 0, NEG_INF)
        e = e.permute(0, 3, 1, 2)  # `[B, H_ma, qlen, klen]`

        return e


class ChunkEnergy(nn.Module):
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

    def __init__(self, kdim, qdim, adim, atype, n_heads=1,
                 bias=True, param_init=''):

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
                boundary_leftmost=0, boundary_rightmost=100000):
        """Compute chunkwise energy.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            cache (bool): cache key and mask
            boundary_leftmost (int): leftmost boundary position
            boundary_rightmost (int): rightmost boundary position
        Returns:
            e (FloatTensor): `[B, H_ca, qlen, klen]`

        """
        bs, klen, kdim = key.size()
        qlen = query.size(1)

        # Pre-computation of encoder-side features for computing scores
        if self.key is None or not cache:
            self.key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)  # `[B, klen, H_ca, d_k]`
            self.mask = mask
            if mask is not None:
                self.mask = self.mask.unsqueeze(3).repeat([1, 1, 1, self.n_heads])  # `[B, qlen, klen, H_ca]`
                mask_size = (bs, qlen, klen, self.n_heads)
                assert self.mask.size() == mask_size, (self.mask.size(), mask_size)

        key = self.key
        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)  # `[B, qlen, H_ca, d_k]`
        m = self.mask

        # Truncate encoder memories for efficient decoding
        if boundary_leftmost > 0 or boundary_rightmost < klen:
            key = key[:, boundary_leftmost:boundary_rightmost]
            klen = key.size(1)
            if m is not None:
                m = m[:, :, boundary_leftmost:boundary_rightmost]

        if self.atype == 'scaled_dot':
            e = torch.einsum("bihd,bjhd->bijh", (query, key)) / self.scale
        elif self.atype == 'add':
            e = self.v(torch.relu(key[:, None] + query[:, :, None]).view(bs, qlen, klen, -1))
        # e: `[B, qlen, klen, H_ca]`

        if m is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=e.dtype).numpy().dtype).min)
            e = e.masked_fill_(m == 0, NEG_INF)
        e = e.permute(0, 3, 1, 2)  # `[B, H_ca, qlen, klen]`

        return e


class MoChA(nn.Module):
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

    def __init__(self, kdim, qdim, adim, odim, atype, chunk_size,
                 n_heads_mono=1, n_heads_chunk=1,
                 conv1d=False, init_r=-4, eps=1e-6, noise_std=1.0,
                 no_denominator=False, sharpening_factor=1.0,
                 dropout=0., dropout_head=0., bias=True, param_init='',
                 decot=False, lookahead=2, share_chunkwise_attention=False):

        super().__init__()

        self.atype = atype
        assert adim % (n_heads_mono * n_heads_chunk) == 0
        self.d_k = adim // (n_heads_mono * n_heads_chunk)

        self.w = chunk_size
        self.milk = (chunk_size == -1)
        self.n_heads = n_heads_mono
        self.n_heads_ma = n_heads_mono
        self.n_heads_ca = n_heads_chunk
        self.eps = eps
        self.noise_std = noise_std
        self.no_denom = no_denominator
        self.sharpening_factor = sharpening_factor
        self.decot = decot
        self.lookahead = lookahead
        self.share_ca = share_chunkwise_attention

        if n_heads_mono >= 1:
            self.monotonic_energy = MonotonicEnergy(
                kdim, qdim, adim, atype,
                n_heads_mono, init_r,
                bias, param_init, conv1d=conv1d)
        else:
            self.monotonic_energy = None
            logger.info('Only chunkwise attention is enabled.')

        if chunk_size > 1 or self.milk:
            self.chunk_energy = ChunkEnergy(
                kdim, qdim, adim, atype,
                n_heads_chunk if self.share_ca else n_heads_mono * n_heads_chunk,
                bias, param_init)
        else:
            self.chunk_energy = None

        if n_heads_mono * n_heads_chunk > 1:
            self.w_value = nn.Linear(kdim, adim, bias=bias)
            self.w_out = nn.Linear(adim, odim, bias=bias)
            if param_init == 'xavier_uniform':
                self.reset_parameters(bias)

        # attention dropout
        self.dropout_attn = nn.Dropout(p=dropout)  # for beta
        self.dropout_head = dropout_head

        self.bd_offset = 0
        self.key_prev_tail = None

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
        if self.monotonic_energy is not None:
            self.monotonic_energy.reset()
        if self.chunk_energy is not None:
            self.chunk_energy.reset()
        self.bd_offset = 0
        self.key_prev_tail = None

    def register_key_prev_tail(self, key):
        # for chunkwise attention during streaming decoding
        self.key_prev_tail = key[:, -(self.w - 1):]

    def recursive(self, e_ma, aw_prev):
        bs, n_heads_ma, qlen, klen = e_ma.size()
        p_choose = torch.sigmoid(add_gaussian_noise(e_ma, self.noise_std))  # `[B, H_ma, qlen, klen]`
        alpha = []
        for i in range(qlen):
            # Compute [1, 1 - p_choose[0], 1 - p_choose[1], ..., 1 - p_choose[-2]]
            shifted_1mp_choose = torch.cat([e_ma.new_ones(bs, self.n_heads_ma, 1, 1),
                                            1 - p_choose[:, :, i:i + 1, :-1]], dim=-1)
            # Compute attention distribution recursively as
            # q_j = (1 - p_choose_j) * q_(j-1) + aw_prev_j
            # alpha_j = p_choose_j * q_j
            q = e_ma.new_zeros(bs, self.n_heads_ma, 1, klen + 1)
            for j in range(klen):
                q[:, :, i:i + 1, j + 1] = shifted_1mp_choose[:, :, i:i + 1, j].clone() * q[:, :, i:i + 1, j].clone() + \
                    aw_prev[:, :, :, j].clone()
            aw_prev = p_choose[:, :, i:i + 1] * q[:, :, i:i + 1, 1:]  # `[B, H_ma, 1, klen]`
            alpha.append(aw_prev)
        alpha = torch.cat(alpha, dim=2) if qlen > 1 else alpha[-1]  # `[B, H_ma, qlen, klen]`
        return alpha, p_choose

    def parallel(self, e_ma, aw_prev, trigger_points):
        bs, n_heads_ma, qlen, klen = e_ma.size()
        p_choose = torch.sigmoid(add_gaussian_noise(e_ma, self.noise_std))  # `[B, H_ma, qlen, klen]`
        alpha = []
        # safe_cumprod computes cumprod in logspace with numeric checks
        cumprod_1mp_choose = safe_cumprod(1 - p_choose, eps=self.eps)  # `[B, H_ma, qlen, klen]`
        # Compute recurrence relation solution
        for i in range(qlen):
            denom = 1 if self.no_denom else torch.clamp(
                cumprod_1mp_choose[:, :, i:i + 1], min=self.eps, max=1.0)
            aw_prev = p_choose[:, :, i:i + 1] * cumprod_1mp_choose[:, :, i:i + 1] * torch.cumsum(
                aw_prev / denom, dim=-1)  # `[B, H_ma, 1, klen]`
            # Mask the right part from the trigger point
            if self.decot:
                assert trigger_points is not None
                for b in range(bs):
                    aw_prev[b, :, :, trigger_points[b, i:i + 1] + self.lookahead + 1:] = 0
            alpha.append(aw_prev)

        alpha = torch.cat(alpha, dim=2) if qlen > 1 else alpha[-1]  # `[B, H_ma, qlen, klen]`
        return alpha, p_choose

    def hard(self, e_ma, aw_prev, eps_wait):
        bs, n_heads_ma, qlen, klen = e_ma.size()
        assert qlen == 1
        assert not self.training

        aw_prev = aw_prev[:, :, :, -klen:]

        if self.n_heads_ma == 1:
            # assert aw_prev.sum() > 0
            p_choose_i = (torch.sigmoid(e_ma) >= 0.5).float()[:, :, 0:1]
            # Attend when monotonic energy is above threshold (Sigmoid > 0.5)
            # Remove any probabilities before the index chosen at the last time step
            p_choose_i *= torch.cumsum(
                aw_prev[:, :, 0:1, -e_ma.size(3):], dim=-1)  # `[B, H_ma, 1 (qlen), klen]`
            # Now, use exclusive cumprod to remove probabilities after the first
            # chosen index, like so:
            # p_choose_i                        = [0, 0, 0, 1, 1, 0, 1, 1]
            # 1 - p_choose_i                    = [1, 1, 1, 0, 0, 1, 0, 0]
            # exclusive_cumprod(1 - p_choose_i) = [1, 1, 1, 1, 0, 0, 0, 0]
            # alpha: product of above           = [0, 0, 0, 1, 0, 0, 0, 0]
            alpha = p_choose_i * exclusive_cumprod(1 - p_choose_i)  # `[B, H_ma, 1 (qlen), klen]`
        else:
            p_choose_i = (torch.sigmoid(e_ma) >= 0.5).float()[:, :, 0:1]
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

                leftmost = alpha[b, :, -1].nonzero()[:, -1].min().item()
                rightmost = alpha[b, :, -1].nonzero()[:, -1].max().item()
                for h in range(self.n_heads_ma):
                    # no bondary at the h-th head
                    if alpha[b, h, -1].sum().item() == 0:
                        alpha[b, h, -1, min(rightmost, leftmost + eps_wait)] = 1
                        continue

                    # surpass acceptable latency
                    if alpha[b, h, -1].nonzero()[:, -1].min().item() >= leftmost + eps_wait:
                        alpha[b, h, -1, :] = 0  # reset
                        alpha[b, h, -1, leftmost + eps_wait] = 1

        return alpha, None

    def forward(self, key, value, query, mask=None, aw_prev=None,
                cache=False, mode='hard', trigger_points=None, eps_wait=-1,
                efficient_decoding=False):
        """Forward pass.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            value (FloatTensor): `[B, klen, vdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            aw_prev (FloatTensor): `[B, H_ma, 1, klen]`
            cache (bool): cache key and mask
            mode (str): recursive/parallel/hard
            trigger_points (IntTensor): `[B, qlen]`
            eps_wait (int): wait time delay for head-synchronous decoding in MMA
        Returns:
            cv (FloatTensor): `[B, qlen, vdim]`
            alpha (FloatTensor): `[B, H_ma, qlen, klen]`
            beta (FloatTensor): `[B, H_ma * H_ca, qlen, klen]`
            p_choose (FloatTensor): `[B, H_ma, qlen, klen]`

        """
        bs, klen = key.size()[:2]
        qlen = query.size(1)
        tail_len = self.key_prev_tail.size(1) if self.key_prev_tail is not None else 0

        if aw_prev is None:
            # aw_prev = [1, 0, 0 ... 0]
            aw_prev = key.new_zeros(bs, self.n_heads_ma, 1, klen)
            aw_prev[:, :, :, 0:1] = key.new_ones(bs, self.n_heads_ma, 1, 1)

        # Compute monotonic energy
        e_ma = self.monotonic_energy(key, query, mask, cache=cache,
                                     boundary_leftmost=self.bd_offset)  # `[B, H_ma, qlen, klen]`
        assert e_ma.size(3) + self.bd_offset == key.size(1)

        if mode == 'recursive':  # training
            alpha, p_choose = self.recursive(e_ma, aw_prev)
            alpha_masked = alpha.clone()

        elif mode == 'parallel':  # training (efficient)
            alpha, p_choose = self.parallel(e_ma, aw_prev, trigger_points)

            # mask out each head independently (HeadDrop)
            if self.dropout_head > 0 and self.training:
                alpha_masked = headdrop(alpha.clone(), self.n_heads_ma, self.dropout_head)
            else:
                alpha_masked = alpha.clone()

        elif mode == 'hard':  # inference
            alpha, p_choose = self.hard(e_ma, aw_prev, eps_wait)
            alpha_masked = alpha.clone()

        else:
            raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")

        # Compute chunk energy
        beta = None
        if self.chunk_energy is not None:
            bd_leftmost = 0
            bd_rightmost = klen - 1 - self.bd_offset
            if efficient_decoding and mode == 'hard' and alpha.sum() > 0:
                bd_leftmost = alpha[:, :, 0].nonzero()[:, -1].min().item()
                bd_rightmost = alpha[:, :, 0].nonzero()[:, -1].max().item()
                if bd_leftmost == bd_rightmost:
                    alpha_masked = alpha_masked[:, :, :, bd_leftmost:bd_leftmost + 1]
                else:
                    alpha_masked = alpha_masked[:, :, :, bd_leftmost:bd_rightmost]

            if mode == 'hard':
                if self.key_prev_tail is not None:
                    key_ = torch.cat([self.key_prev_tail[0:1].repeat([bs, 1, 1]), key], dim=1)
                else:
                    key_ = key
                e_ca = self.chunk_energy(key_, query, mask, cache=cache,
                                         boundary_leftmost=0 if self.milk else max(
                                             0, self.bd_offset + bd_leftmost - self.w + 1),
                                         boundary_rightmost=self.bd_offset + bd_rightmost + 1 + tail_len)  # `[B, (H_ma*)H_ca, qlen, ken]`
            else:
                e_ca = self.chunk_energy(key, query, mask, cache=cache)  # `[B, (H_ma*)H_ca, qlen, ken]`

            # padding for chunkwise attention over adjacent input segments
            additional = e_ca.size(3) - alpha_masked.size(3)
            if efficient_decoding and mode == 'hard':
                alpha = torch.cat([alpha.new_zeros(bs, alpha.size(1), 1, klen - alpha.size(3)), alpha], dim=3)
                if additional > 0:
                    alpha_masked = torch.cat([alpha_masked.new_zeros(bs, alpha_masked.size(1), 1, additional),
                                              alpha_masked], dim=3)

            if mode == 'hard':
                if self.key_prev_tail is not None:
                    alpha_masked = torch.cat([alpha_masked.new_zeros(bs, self.n_heads_ma, qlen, tail_len),
                                              alpha_masked], dim=3)
                beta = hard_chunkwise_attention(alpha_masked, e_ca, mask, self.w,
                                                self.n_heads_ca, self.sharpening_factor,
                                                self.share_ca)
            else:
                beta = efficient_chunkwise_attention(alpha_masked, e_ca, mask, self.w,
                                                     self.n_heads_ca, self.sharpening_factor,
                                                     self.share_ca)
            beta = self.dropout_attn(beta)  # `[B, H_ma * H_ca, qlen, klen]`

            if efficient_decoding and mode == 'hard':
                value = value[:, max(0, self.bd_offset + bd_leftmost - self.w + 1):self.bd_offset + bd_rightmost + 1]

        # Update after calculating beta
        bd_offset_prev = self.bd_offset
        if efficient_decoding and mode == 'hard' and alpha.sum() > 0:
            self.bd_offset += alpha[:, :, 0, self.bd_offset:].nonzero()[:, -1].min().item()

        # Compute context vector
        if self.n_heads_ma * self.n_heads_ca > 1:
            value = self.w_value(value).view(bs, -1, self.n_heads_ma * self.n_heads_ca, self.d_k)
            value = value.transpose(2, 1).contiguous()  # `[B, H_ma * H_ca, klen, d_k]`
            cv = torch.matmul(alpha if self.w == 1 else beta, value)  # `[B, H_ma * H_ca, qlen, d_k]`
            cv = cv.transpose(2, 1).contiguous().view(bs, -1, self.n_heads_ma * self.n_heads_ca * self.d_k)
            cv = self.w_out(cv)  # `[B, qlen, adim]`
        else:
            if self.w == 1:
                cv = torch.bmm(alpha.squeeze(1), value)  # `[B, 1, adim]`
            else:
                if self.key_prev_tail is not None:
                    value_ = torch.cat([self.key_prev_tail[0:1].repeat([bs, 1, 1]), value], dim=1)
                    cv = torch.bmm(beta.squeeze(1), value_)  # `[B, 1, adim]`
                else:
                    cv = torch.bmm(beta.squeeze(1), value)  # `[B, 1, adim]`

        assert alpha.size() == (bs, self.n_heads_ma, qlen, klen), \
            (alpha.size(), (bs, self.n_heads_ma, qlen, klen))
        if self.w > 1:
            _w = max(1, (bd_offset_prev + bd_rightmost + 1) - max(0, bd_offset_prev + bd_leftmost - self.w + 1))
            assert beta.size() == (bs, self.n_heads_ma * self.n_heads_ca, qlen, _w + tail_len), \
                (beta.size(), (bs, self.n_heads_ma * self.n_heads_ca, qlen, _w + tail_len))
        elif self.milk:
            assert beta.size() == (bs, self.n_heads_ma * self.n_heads_ca, qlen, klen), \
                (beta.size(), (bs, self.n_heads_ma * self.n_heads_ca, qlen, klen))

        return cv, alpha, beta, p_choose


def add_gaussian_noise(xs, std):
    """Add Gaussian noise to encourage discreteness."""
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
        back (int): number of lookback frames
        forward (int): number of lookahead frames

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
