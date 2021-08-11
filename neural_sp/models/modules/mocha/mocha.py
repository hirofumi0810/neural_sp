# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Monotonic (multihead) chunkwise attention."""

# [reference]
# https://github.com/j-min/MoChA-pytorch/blob/94b54a7fa13e4ac6dc255b509dd0febc8c0a0ee6/attention.py

import logging
import math
import torch
import torch.nn as nn

from neural_sp.models.modules.headdrop import headdrop
from neural_sp.models.modules.mocha.hma_train import parallel_monotonic_attention
from neural_sp.models.modules.mocha.hma_test import hard_monotonic_attention
from neural_sp.models.modules.mocha.mocha_train import soft_chunkwise_attention
from neural_sp.models.modules.mocha.mocha_test import hard_chunkwise_attention
from neural_sp.models.modules.mocha.chunk_energy import ChunkEnergy
from neural_sp.models.modules.mocha.monotonic_energy import MonotonicEnergy

logger = logging.getLogger(__name__)


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
        noise_std (float): standard deviation for Gaussian noise
        no_denominator (bool): set the denominator to 1 in the alpha recurrence
        sharpening_factor (float): sharping factor for beta calculation
        dropout (float): dropout probability for attention weights
        dropout_head (float): HeadDrop probability
        bias (bool): use bias term in linear layers
        param_init (str): parameter initialization method
        decot (bool): delay constrainted training (DeCoT)
        decot_delta (int): tolerance frames for DeCoT
        share_chunkwise_attention (int): share CA heads among MA heads
        stableemit_weight (float): StableEmit weight for selection probability

    """

    def __init__(self, kdim, qdim, adim, odim, atype, chunk_size,
                 n_heads_mono=1, n_heads_chunk=1,
                 conv1d=False, init_r=-4, eps=1e-6, noise_std=1.0,
                 no_denominator=False, sharpening_factor=1.0,
                 dropout=0., dropout_head=0., bias=True, param_init='',
                 decot=False, decot_delta=2, share_chunkwise_attention=False,
                 stableemit_weight=0.0):

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
        self.decot_delta = decot_delta
        self.share_ca = share_chunkwise_attention
        self.stableemit_weight = stableemit_weight
        assert stableemit_weight >= 0
        self._stableemit_weight = 0  # for curriculum
        self.p_threshold = 0.5
        logger.info('stableemit_weight: %.3f' % stableemit_weight)

        if n_heads_mono >= 1:
            self.monotonic_energy = MonotonicEnergy(
                kdim, qdim, adim, atype,
                n_heads_mono, init_r,
                bias, param_init, conv1d=conv1d)
        else:
            self.monotonic_energy = None
            logger.info('monotonic attention is disabled.')

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
        self.key_tail = None

    def register_tail(self, key_tail):
        self.key_tail = key_tail

    def trigger_stableemit(self):
        logger.info('Activate StableEmit')
        self._stableemit_weight = self.stableemit_weight

    def set_p_choose_threshold(self, p):
        self.p_threshold = p

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
            mode (str): parallel/hard
            trigger_points (IntTensor): `[B, qlen]`
            eps_wait (int): wait time delay for head-synchronous decoding in MMA
            linear_decoding (bool): linear-time decoding mode
            streaming (bool): streaming mode (use self.key_tail)
        Returns:
            cv (FloatTensor): `[B, qlen, vdim]`
            alpha (FloatTensor): `[B, H_ma, qlen, klen]`
            attn_state (dict):
                beta (FloatTensor): `[B, H_ma * H_ca, qlen, klen]`
                p_choose (FloatTensor): `[B, H_ma, qlen, klen]`

        """
        klen = key.size(1)
        bs, qlen = query.size()[:2]
        tail_len = self.key_tail.size(1) if self.key_tail is not None else 0
        bd_L = self.bd_L_prev
        bd_R = klen - 1
        assert bd_L <= bd_R
        attn_state = {}

        if aw_prev is None:
            aw_prev = key.new_zeros(bs, self.H_ma, 1, klen)
            aw_prev[:, :, :, 0:1] = key.new_ones(bs, self.H_ma, 1, 1)  # [1, 0, 0 ... 0]

        # Compute monotonic energy
        e_ma = self.monotonic_energy(key, query, mask, cache, bd_L, bd_R)  # `[B, H_ma, qlen, klen]`
        assert e_ma.size(3) + bd_L == klen, (e_ma.size(), self.bd_L_prev, key.size())

        if mode == 'parallel':  # training
            alpha, p_choose = parallel_monotonic_attention(
                e_ma, aw_prev, trigger_points, self.eps, self.noise_std, self.no_denom,
                self.decot, self.decot_delta, self._stableemit_weight)
            # NOTE: do not mask alpha for numerial issue

            # mask out each head independently (HeadDrop)
            if self.dropout_head > 0 and self.training:
                alpha_masked = headdrop(alpha.clone(), self.H_ma, self.dropout_head)
            else:
                alpha_masked = alpha.clone()
        elif mode == 'hard':  # inference
            aw_prev = aw_prev[:, :, :, -e_ma.size(3):]
            alpha, p_choose = hard_monotonic_attention(e_ma, aw_prev, eps_wait, self.p_threshold)
            alpha_masked = alpha.clone()
        else:
            raise ValueError("mode must be 'parallel' or 'hard'.")

        is_boundary = (alpha.sum().item() > 0)

        # to the right of the leftmost boundary offset at the current step
        if linear_decoding and mode == 'hard' and is_boundary:
            bd_L = self.bd_L_prev + alpha[:, :, -1].nonzero()[:, -1].min().item()
            bd_R = self.bd_L_prev + alpha[:, :, -1].nonzero()[:, -1].max().item()
        bd_L_ca = max(0, bd_L + 1 - self.w) if not self.milk else 0
        use_tail = streaming and is_boundary and tail_len > 0

        # Compute chunk energy
        beta = None
        if self.chunk_energy is not None:
            if mode == 'hard':
                if not is_boundary:
                    # No boundary detected
                    beta = alpha.new_zeros(bs, self.H_total, qlen, value.size(1))
                else:
                    if use_tail:
                        key = torch.cat([self.key_tail, key], dim=1)
                        bd_L += tail_len
                        bd_R += tail_len
                    bd_L_ca = max(0, bd_L + 1 - self.w) if not self.milk else 0

                    e_ca = self.chunk_energy(key, query, mask, cache, bd_L_ca, bd_R)  # `[B, (H_ma*)H_ca, qlen, klen]`
                    assert e_ca.size(3) == bd_R - bd_L_ca + 1, (e_ca.size(), bd_L_ca, bd_R, key.size())

                    if alpha_masked.size(3) < klen:
                        # back to the original shape
                        alpha_masked = torch.cat([alpha.new_zeros(bs, self.H_ma, qlen, klen - alpha_masked.size(3)),
                                                  alpha_masked], dim=3)
                    if use_tail:
                        alpha_masked = torch.cat([alpha.new_zeros(bs, self.H_ma, qlen, tail_len),
                                                  alpha_masked], dim=3)
                        value = torch.cat([self.key_tail[0:1], value[0:1]], dim=1)

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
                e_ca = self.chunk_energy(key, query, mask, cache, 0, bd_R)  # `[B, (H_ma*)H_ca, qlen, klen]`

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
            cv = torch.bmm(alpha_masked.squeeze(1) if self.w == 1 else beta.squeeze(1), value)  # `[B, qlen, adim]`

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

        attn_state['beta'] = beta
        attn_state['p_choose'] = p_choose

        return cv, alpha, attn_state
