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

from neural_sp.models.modules.headdrop import headdrop
from neural_sp.models.modules.initialization import init_with_lecun_normal
from neural_sp.models.modules.initialization import init_with_xavier_uniform

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
                 bias=True, param_init='',
                 conv1d=False, conv_kernel_size=5):

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
            self.conv1d = nn.Conv1d(kdim, kdim, conv_kernel_size,
                                    padding=(conv_kernel_size - 1) // 2)
            # NOTE: lookahead is introduced
            for n, p in self.conv1d.named_parameters():
                init_with_lecun_normal(n, p, 0.1)

        if atype == 'add':
            self.v = nn.utils.weight_norm(self.v, name='weight', dim=0)
            # initialization
            self.v.weight_g.data = torch.Tensor([1 / adim]).sqrt()
        elif atype == 'scaled_dot':
            if param_init == 'xavier_uniform':
                self.reset_parameters_xavier_uniform(bias)

    def reset_parameters_xavier_uniform(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' %
                    self.__class__.__name__)
        # NOTE: see https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.)
            nn.init.constant_(self.w_query.bias, 0.)
        if self.conv1d is not None:
            for n, p in self.conv1d.named_parameters():
                init_with_xavier_uniform(n, p)

    def reset(self):
        self.key = None
        self.mask = None

    def forward(self, key, query, mask, cache=False,
                boundary_leftmost=0, boundary_rightmost=100000):
        """Compute monotonic energy.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            cache (bool): cache key and mask
            boundary_leftmost (int): leftmost boundary offset
            boundary_rightmost (int): rightmost boundary offset
        Returns:
            e (FloatTensor): `[B, H_ma, qlen, klen]`

        """
        klen, kdim = key.size()[1:]
        bs, qlen = query.size()[:2]

        # Pre-computation of encoder-side features for computing scores
        if self.key is None or not cache:
            # 1d conv
            if self.conv1d is not None:
                key = torch.relu(self.conv1d(key.transpose(2, 1))).transpose(2, 1)
            self.key = self.w_key(key)  # `[B, klen, adim]`
            self.key = self.key.view(-1, klen, self.n_heads, self.d_k)  # `[B, klen, H_ma, d_k]`
            if mask is not None:
                self.mask = mask.unsqueeze(3).repeat([1, 1, 1, self.n_heads])  # `[B, qlen, klen, H_ca]`
                mask_size = (bs, qlen, klen, self.n_heads)
                assert self.mask.size() == mask_size, (self.mask.size(), mask_size)
            else:
                self.mask = None

        k = self.key
        if k.size(0) != bs:  # for infernece
            k = k[0:1].repeat([bs, 1, 1, 1])
        klen = k.size(1)
        q = self.w_query(query).view(-1, qlen, self.n_heads, self.d_k)
        m = self.mask

        # Truncate encoder memories for efficient DECODING
        if boundary_leftmost > 0:
            k = k[:, boundary_leftmost:]
            klen = k.size(1)
            if m is not None:
                m = m[:, :, boundary_leftmost:]

        if self.atype == 'scaled_dot':
            e = torch.einsum("bihd,bjhd->bijh", (q, k)) / self.scale
        elif self.atype == 'add':
            e = self.v(torch.relu(k[:, None] + q[:, :, None]).view(bs, qlen, klen, -1))
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
                self.reset_parameters_xavier_uniform(bias)
        else:
            raise NotImplementedError(atype)

    def reset_parameters_xavier_uniform(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' %
                    self.__class__.__name__)
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
            boundary_leftmost (int): leftmost boundary offset
            boundary_rightmost (int): rightmost boundary offset
        Returns:
            e (FloatTensor): `[B, H_ca, qlen, klen]`

        """
        klen, kdim = key.size()[1:]
        bs, qlen = query.size()[:2]

        # Pre-computation of encoder-side features for computing scores
        if self.key is None or not cache:
            self.key = self.w_key(key).view(-1, klen, self.n_heads, self.d_k)  # `[B, klen, H_ca, d_k]`
            if mask is not None:
                self.mask = mask.unsqueeze(3).repeat([1, 1, 1, self.n_heads])  # `[B, qlen, klen, H_ca]`
                mask_size = (bs, qlen, klen, self.n_heads)
                assert self.mask.size() == mask_size, (self.mask.size(), mask_size)
            else:
                self.mask = None

        k = self.key
        if k.size(0) != bs:  # for infernece
            k = k[0:1].repeat([bs, 1, 1, 1])
        klen = k.size(1)
        q = self.w_query(query).view(-1, qlen, self.n_heads, self.d_k)  # `[B, qlen, H_ca, d_k]`
        m = self.mask

        # Truncate encoder memories for efficient DECODING
        if boundary_leftmost > 0 or (0 < boundary_rightmost < klen):
            k = k[:, boundary_leftmost:boundary_rightmost + 1]
            klen = k.size(1)
            if m is not None:
                m = m[:, :, boundary_leftmost:boundary_rightmost + 1]

        if self.atype == 'scaled_dot':
            e = torch.einsum("bihd,bjhd->bijh", (q, k)) / self.scale
        elif self.atype == 'add':
            e = self.v(torch.relu(k[:, None] + q[:, :, None]).view(bs, qlen, klen, -1))
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
        assert adim % (max(1, n_heads_mono) * n_heads_chunk) == 0
        self.d_k = adim // (max(1, n_heads_mono) * n_heads_chunk)

        self.w = chunk_size
        self.milk = (chunk_size == -1)
        self.n_heads = n_heads_mono
        self.H_ma = max(1, n_heads_mono)
        self.H_ca = n_heads_chunk
        self.H_total = self.H_ma * self.H_ca
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
                n_heads_chunk if self.share_ca else self.H_ma * n_heads_chunk,
                bias, param_init)
        else:
            self.chunk_energy = None

        if self.H_ma * n_heads_chunk > 1:
            self.w_value = nn.Linear(kdim, adim, bias=bias)
            self.w_out = nn.Linear(adim, odim, bias=bias)
            if param_init == 'xavier_uniform':
                self.reset_parameters_xavier_uniform(bias)

        # attention dropout
        self.dropout_attn = nn.Dropout(p=dropout)  # for beta
        self.dropout_head = dropout_head

        self.reset()

    def reset_parameters_xavier_uniform(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' %
                    self.__class__.__name__)
        # NOTE: see https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
        nn.init.xavier_uniform_(self.w_value.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_value.bias, 0.)

        nn.init.xavier_uniform_(self.w_out.weight)
        if bias:
            nn.init.constant_(self.w_out.bias, 0.)

    def reset(self):
        """Reset when a speaker changes."""
        if self.monotonic_energy is not None:
            self.monotonic_energy.reset()
        if self.chunk_energy is not None:
            self.chunk_energy.reset()
        self.bd_L_prev = 0
        self.key_prev_tail = None
        self.key_cur_tail = None

    def reset_block(self):
        """Reset when moving to the next block. This is used for streaming inference."""
        if self.monotonic_energy is not None:
            self.monotonic_energy.reset()
        if self.chunk_energy is not None:
            self.chunk_energy.reset()
        self.bd_L_prev = 0
        self.key_prev_tail = self.key_cur_tail
        self.key_cur_tail = None
        # NOTE: cache encoder outputs at the previous block

    def recursive(self, e_ma, aw_prev):
        bs, n_heads_ma, qlen, klen = e_ma.size()
        p_choose = torch.sigmoid(add_gaussian_noise(e_ma, self.noise_std))  # `[B, H_ma, qlen, klen]`
        alpha = []
        for i in range(qlen):
            # Compute [1, 1 - p_choose[0], 1 - p_choose[1], ..., 1 - p_choose[-2]]
            shifted_1mp_choose = torch.cat([e_ma.new_ones(bs, self.H_ma, 1, 1),
                                            1 - p_choose[:, :, i:i + 1, :-1]], dim=-1)
            # Compute attention distribution recursively as
            # q_j = (1 - p_choose_j) * q_(j-1) + aw_prev_j
            # alpha_j = p_choose_j * q_j
            q = e_ma.new_zeros(bs, self.H_ma, 1, klen + 1)
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
        assert e_ma.size(-1) == aw_prev.size(-1)
        assert not self.training

        aw_prev = aw_prev[:, :, :, -klen:]

        if self.H_ma == 1:
            # assert aw_prev.sum() > 0
            p_choose = (torch.sigmoid(e_ma) >= 0.5).float()[:, :, 0:1]
            # Attend when monotonic energy is above threshold (Sigmoid > 0.5)
            # Remove any probabilities before the index chosen at the last time step
            p_choose *= torch.cumsum(
                aw_prev[:, :, 0:1, -e_ma.size(3):], dim=-1)  # `[B, H_ma, 1 (qlen), klen]`
            # Now, use exclusive cumprod to remove probabilities after the first
            # chosen index, like so:
            # p_choose                        = [0, 0, 0, 1, 1, 0, 1, 1]
            # 1 - p_choose                    = [1, 1, 1, 0, 0, 1, 0, 0]
            # exclusive_cumprod(1 - p_choose) = [1, 1, 1, 1, 0, 0, 0, 0]
            # alpha: product of above         = [0, 0, 0, 1, 0, 0, 0, 0]
            alpha = p_choose * exclusive_cumprod(1 - p_choose)  # `[B, H_ma, 1 (qlen), klen]`
        else:
            p_choose = (torch.sigmoid(e_ma) >= 0.5).float()[:, :, 0:1]
            # Attend when monotonic energy is above threshold (Sigmoid > 0.5)
            # Remove any probabilities before the index chosen at the last time step
            p_choose *= torch.cumsum(aw_prev[:, :, 0:1], dim=-1)  # `[B, H_ma, 1 (qlen), klen]`
            # Now, use exclusive cumprod to remove probabilities after the first
            # chosen index, like so:
            # p_choose                        = [0, 0, 0, 1, 1, 0, 1, 1]
            # 1 - p_choose                    = [1, 1, 1, 0, 0, 1, 0, 0]
            # exclusive_cumprod(1 - p_choose) = [1, 1, 1, 1, 0, 0, 0, 0]
            # alpha: product of above         = [0, 0, 0, 1, 0, 0, 0, 0]
            alpha = p_choose * exclusive_cumprod(1 - p_choose)  # `[B, H_ma, 1 (qlen), klen]`

        if eps_wait > 0:
            for b in range(bs):
                # no boundary until the last frame for all heads
                if alpha[b].sum() == 0:
                    continue

                leftmost = alpha[b, :, -1].nonzero()[:, -1].min().item()
                rightmost = alpha[b, :, -1].nonzero()[:, -1].max().item()
                for h in range(self.H_ma):
                    # no bondary at the h-th head
                    if alpha[b, h, -1].sum().item() == 0:
                        alpha[b, h, -1, min(rightmost, leftmost + eps_wait)] = 1
                        continue

                    # surpass acceptable latency
                    if alpha[b, h, -1].nonzero()[:, -1].min().item() >= leftmost + eps_wait:
                        alpha[b, h, -1, :] = 0  # reset
                        alpha[b, h, -1, leftmost + eps_wait] = 1

        return alpha, p_choose

    def forward(self, key, value, query, mask, aw_prev=None,
                cache=False, mode='hard', trigger_points=None, eps_wait=-1,
                linear_decoding=False, streaming=False):
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
            linear_decoding (bool): linear-time decoding mode
            streaming (bool): streaming mode (use self.key_prev_tail)
        Returns:
            cv (FloatTensor): `[B, qlen, vdim]`
            alpha (FloatTensor): `[B, H_ma, qlen, klen]`
            attn_state (dict):
                beta (FloatTensor): `[B, H_ma * H_ca, qlen, klen]`
                p_choose (FloatTensor): `[B, H_ma, qlen, klen]`

        """
        klen = key.size(1)
        bs, qlen = query.size()[:2]
        tail_len = self.key_prev_tail.size(1) if self.key_prev_tail is not None else 0
        bd_L = self.bd_L_prev
        bd_R = klen - 1
        attn_state = {}

        if aw_prev is None:
            aw_prev = key.new_zeros(bs, self.H_ma, 1, klen)
            aw_prev[:, :, :, 0:1] = key.new_ones(bs, self.H_ma, 1, 1)  # aw_prev = [1, 0, 0 ... 0]

        # Compute monotonic energy
        e_ma = self.monotonic_energy(key, query, mask, cache, bd_L, bd_R)  # `[B, H_ma, qlen, klen]`
        assert e_ma.size(3) + bd_L == klen, (e_ma.size(), self.bd_L_prev, key.size())

        if mode == 'recursive':  # training (incremental)
            alpha, p_choose = self.recursive(e_ma, aw_prev)
            alpha_masked = alpha.clone()
        elif mode == 'parallel':  # training (efficient)
            alpha, p_choose = self.parallel(e_ma, aw_prev, trigger_points)
            # mask out each head independently (HeadDrop)
            if self.dropout_head > 0 and self.training:
                alpha_masked = headdrop(alpha.clone(), self.H_ma, self.dropout_head)
            else:
                alpha_masked = alpha.clone()
        elif mode == 'hard':  # inference
            aw_prev = aw_prev[:, :, :, -e_ma.size(3):]
            alpha, p_choose = self.hard(e_ma, aw_prev, eps_wait)
            alpha_masked = alpha.clone()
        else:
            raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")

        is_boundary = (alpha.sum().item() > 0)

        # to the right of the leftmost boundary offset at the current step
        if linear_decoding and mode == 'hard' and is_boundary:
            bd_L = self.bd_L_prev + alpha[:, :, -1].nonzero()[:, -1].min().item()
            bd_R = self.bd_L_prev + alpha[:, :, -1].nonzero()[:, -1].max().item()
        bd_L_ca = max(0, bd_L + 1 - self.w) if not self.milk else 0
        use_tail = streaming and is_boundary and (bd_L + 1 < self.w) and tail_len > 0

        # Compute chunk energy
        beta = None
        if self.chunk_energy is not None:
            if mode == 'hard':
                if not is_boundary:
                    # No boundary detected
                    beta = alpha.new_zeros(bs, self.H_total, qlen, value.size(1))
                else:
                    if use_tail:
                        key = torch.cat([self.key_prev_tail[0:1], key[0:1]], dim=1)
                        bd_L += tail_len
                        bd_R += tail_len
                    bd_L_ca = max(0, bd_L + 1 - self.w) if not self.milk else 0

                    e_ca = self.chunk_energy(key, query, mask, cache, bd_L_ca, bd_R)  # `[B, (H_ma*)H_ca, qlen, ken]`
                    assert e_ca.size(3) == bd_R - bd_L_ca + 1, (e_ca.size(), bd_L_ca, bd_R, key.size())

                    if alpha_masked.size(3) < klen:
                        # back to the original shape
                        alpha_masked = torch.cat([alpha.new_zeros(bs, self.H_ma, qlen, klen - alpha_masked.size(3)),
                                                  alpha_masked], dim=3)
                    if use_tail:
                        alpha_masked = torch.cat([alpha.new_zeros(bs, self.H_ma, qlen, tail_len),
                                                  alpha_masked], dim=3)
                        value = torch.cat([self.key_prev_tail[0:1], value[0:1]], dim=1)

                    alpha_masked = alpha_masked[:, :, :, bd_L_ca:bd_R + 1]
                    value = value[:, bd_L_ca:bd_R + 1]
                    # NOTE: alpha_masked must have the same shape as beta

                    beta = hard_chunkwise_attention(alpha_masked, e_ca, mask, self.w,
                                                    self.H_ca, self.sharpening_factor,
                                                    self.share_ca)  # `[B, H_ma * H_ca, qlen, klen]`
                    beta = self.dropout_attn(beta)

                    assert beta.size() == (bs, self.H_total, qlen, bd_R - bd_L_ca + 1), \
                        (beta.size(), (bs, self.H_total, qlen, bd_L_ca, bd_R))
            else:
                e_ca = self.chunk_energy(key, query, mask, cache, 0, bd_R)  # `[B, (H_ma*)H_ca, qlen, ken]`

                beta = soft_chunkwise_attention(alpha_masked, e_ca, mask, self.w,
                                                self.H_ca, self.sharpening_factor,
                                                self.share_ca)  # `[B, H_ma * H_ca, qlen, klen]`
                beta = self.dropout_attn(beta)

                assert beta.size() == (bs, self.H_total, qlen, klen), \
                    (beta.size(), (bs, self.H_total, qlen, klen))

        if value.size(0) != bs:  # for infernece
            value = value[0:1].repeat([bs, 1, 1])

        # Compute context vector
        if self.H_total > 1:
            v = self.w_value(value).view(bs, -1, self.H_total, self.d_k)
            # TODO: cache at test time
            v = v.transpose(2, 1).contiguous()  # `[B, H_ma * H_ca, klen, d_k]`
            cv = torch.matmul(alpha_masked if self.w == 1 else beta, v)  # `[B, H_ma * H_ca, qlen, d_k]`
            cv = cv.transpose(2, 1).contiguous().view(bs, -1, self.H_total * self.d_k)
            cv = self.w_out(cv)  # `[B, qlen, adim]`
        else:
            cv = torch.bmm(alpha_masked.squeeze(1) if self.w == 1 else beta.squeeze(1), value)  # `[B, 1, adim]`

        if mode == 'hard' and use_tail:
            bd_L -= tail_len
            bd_R -= tail_len
            alpha_masked = alpha_masked[:, :, :, -klen:]
        self.bd_L_prev = bd_L

        # padding for the next step
        if mode == 'hard':
            alpha = alpha.new_zeros(bs, alpha.size(1), qlen, klen)
            if is_boundary:
                alpha[:, :, :, bd_L:bd_R + 1] = alpha_masked[:, :, :, -(bd_R - bd_L + 1):]

        assert alpha.size() == (bs, self.H_ma, qlen, klen), \
            (alpha.size(), (bs, self.H_ma, qlen, klen, bd_L, bd_R))

        # cache encoder outputs when moving to the next block
        if mode == 'hard' and streaming and self.key_cur_tail is None:
            if not is_boundary:
                self.key_cur_tail = key.detach()[:, -(self.w - 1):]
            elif bd_L + 1 < self.w:
                n_rest = self.w - (bd_L + 1)
                if n_rest < klen:
                    self.key_cur_tail = key.detach()[:, -n_rest:]
                elif self.key_prev_tail is not None:
                    # concatetane multiple blocks (>=3)
                    self.key_cur_tail = torch.cat([self.key_prev_tail[:, -(klen - n_rest):],
                                                   key.detach()], dim=1)[:, -n_rest:]

        attn_state['beta'] = beta
        attn_state['p_choose'] = p_choose

        return cv, alpha, attn_state


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
    x = x.reshape(-1, klen)
    # Moving sum is computed as a carefully-padded 1D convolution with ones
    x_padded = F.pad(x, pad=[back, forward])  # `[B * H_ma * H_ca * qlen, back + klen + forward]`
    # Add a "channel" dimension
    x_padded = x_padded.unsqueeze(1)
    # Construct filters
    filters = x.new_ones(1, 1, back + forward + 1)
    x_sum = F.conv1d(x_padded, filters)
    x_sum = x_sum.squeeze(1).view(bs, n_heads_mono, n_heads_chunk, qlen, -1)
    return x_sum


def soft_chunkwise_attention(alpha, u, mask, chunk_size, H_ca,
                             sharpening_factor, share_chunkwise_attention):
    """Compute chunkwise attention efficiently by clipping logits at training time.

    Args:
        alpha (FloatTensor): `[B, H_ma, qlen, klen]`
        u (FloatTensor): `[B, (H_ma*)H_ca, qlen, klen]`
        mask (ByteTensor): `[B, qlen, klen]`
        chunk_size (int): window size for chunkwise attention
        H_ca (int): number of chunkwise attention heads
        sharpening_factor (float): sharping factor for beta calculation
        share_chunkwise_attention (int): share CA heads among MA heads
    Returns:
        beta (FloatTensor): `[B, H_ma * H_ca, qlen, klen]`

    """
    bs, H_ma, qlen, klen = alpha.size()
    alpha = alpha.unsqueeze(2)  # `[B, H_ma, 1, qlen, klen]`
    u = u.unsqueeze(1)  # `[B, 1, (H_ma*)H_ca, qlen, klen]`
    if H_ca > 1:
        alpha = alpha.repeat([1, 1, H_ca, 1, 1])
    if H_ma > 1 and not share_chunkwise_attention:
        u = u.view(bs, H_ma, H_ca, qlen, klen)
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


def hard_chunkwise_attention(alpha, u, mask, chunk_size, H_ca,
                             sharpening_factor, share_chunkwise_attention):
    """Compute chunkwise attention over hard attention at test time.

    Args:
        alpha (FloatTensor): `[B, H_ma, qlen, klen]`
        u (FloatTensor): `[B, (H_ma*)H_ca, qlen, klen]`
        mask (ByteTensor): `[B, qlen, klen]`
        chunk_size (int): window size for chunkwise attention
        H_ca (int): number of chunkwise attention heads
        sharpening_factor (float): sharping factor for beta calculation
        share_chunkwise_attention (int): share CA heads among MA heads
    Returns:
        beta (FloatTensor): `[B, H_ma * H_ca, qlen, klen]`

    """
    bs, H_ma, qlen, klen = alpha.size()
    assert (u.size(2) == qlen) and (u.size(3) == klen), (u.size(), alpha.size())
    alpha = alpha.unsqueeze(2)   # `[B, H_ma, 1, qlen, klen]`
    u = u.unsqueeze(1)  # `[B, 1, (H_ma*)H_ca, qlen, klen]`
    if H_ca > 1:
        alpha = alpha.repeat([1, 1, H_ca, 1, 1])
    if H_ma > 1:
        if share_chunkwise_attention:
            u = u.repeat([1, H_ma, 1, 1, 1])
        else:
            u = u.view(bs, H_ma, H_ca, qlen, klen)

    mask = alpha.clone().byte()  # `[B, H_ma, H_ca, qlen, klen]`
    for b in range(bs):
        for h in range(H_ma):
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
