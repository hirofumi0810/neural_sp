# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Monotonic energy activation."""

import logging
import math
import numpy as np
import torch
import torch.nn as nn

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
