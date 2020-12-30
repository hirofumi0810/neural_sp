# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GMM attention."""

import logging
import math
import numpy as np
import torch
import torch.nn as nn

from neural_sp.models.modules.initialization import init_with_xavier_uniform
from neural_sp.models.modules.softplus import softplus

logger = logging.getLogger(__name__)


class GMMAttention(nn.Module):
    """GMM attention.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of attention space
        n_mixtures (int): number of mixtures
        dropout (float): dropout probability for attention weights
        param_init (str): parameter initialization method
        vfloor (float): parameter for numerical stability
        nonlinear (torch.function): exp or softplus

    """

    def __init__(self, kdim, qdim, adim, n_mixtures, dropout=0.,
                 param_init='', vfloor=1e-8, nonlinear='exp'):

        super().__init__()

        self.n_mix = n_mixtures
        self.n_heads = 1  # dummy for attention plot
        self.vfloor = vfloor
        self.reset()

        # attention dropout applied to output of GMM attention
        self.dropout = nn.Dropout(p=dropout)

        self.w_mixture = nn.Linear(qdim, n_mixtures)
        self.w_var = nn.Linear(qdim, n_mixtures)
        self.w_myu = nn.Linear(qdim, n_mixtures)
        if nonlinear == 'exp':
            self.nonlinear = torch.exp
        elif nonlinear == 'softplus':
            self.nonlinear = softplus
        else:
            raise NotImplementedError

        if param_init == 'xavier_uniform':
            self.reset_parameters_xavier_uniform()
        else:
            logger.info('Parameter initialization is skipped.')

    def reset_parameters_xavier_uniform(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' %
                    self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_xavier_uniform(n, p)

    def reset(self):
        pass

    def forward(self, key, value, query, mask=None, aw_prev=None,
                cache=False, mode='', trigger_points=None, streaming=False):
        """Forward pass.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            value (FloatTensor): `[B, klen, vdim]`
            query (FloatTensor): `[B, 1, qdim]`
            mask (ByteTensor): `[B, qmax, klen]`
            aw_prev (FloatTensor): `[B, klen, 1]`
            cache (bool): cache key and mask
            mode: dummy interface for MoChA/MMA
            trigger_points: dummy interface for MoChA/MMA
            streaming: dummy interface for streaming attention
        Returns:
            cv (FloatTensor): `[B, 1, vdim]`
            aw (FloatTensor): `[B, 1 (H), 1 (qlen), klen]`
            attn_state (dict):
                myu (FloatTensor): `[B, 1 (qlen), n_mix]`

        """
        bs, klen = key.size()[:2]
        attn_state = {}

        if aw_prev is None:
            myu_prev = query.new_zeros(bs, 1, self.n_mix)
        else:
            myu_prev = aw_prev

        if mask is not None:
            assert mask.size() == (bs, 1, klen), (mask.size(), (bs, 1, klen))

        w_mix = torch.softmax(self.w_mixture(query), dim=-1)  # `[B, 1, n_mix]`
        var = self.nonlinear(self.w_var(query))  # `[B, 1, n_mix]`
        myu = self.nonlinear(self.w_myu(query))
        myu = myu + myu_prev + myu_prev  # `[B, 1, n_mix]`
        attn_state['myu'] = myu

        # Compute attention weights
        js = torch.arange(klen, dtype=torch.float, device=query.device)
        js = js.unsqueeze(0).unsqueeze(2).repeat([bs, 1, self.n_mix])  # `[B, klen, n_mix]`
        numerator = torch.exp(-torch.pow(js - myu, 2) / (2 * var + self.vfloor))
        denominator = torch.pow(2 * math.pi * var + self.vfloor, 0.5)
        aw = w_mix * numerator / denominator  # `[B, klen, n_mix]`
        aw = aw.sum(2).unsqueeze(1)  # `[B, 1 (qlen), klen]`

        # Compute context vector
        if mask is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=myu.dtype).numpy().dtype).min)
            aw = aw.masked_fill_(mask == 0, NEG_INF)
        aw = self.dropout(aw)
        cv = torch.bmm(aw, value)

        return cv, aw.unsqueeze(1), attn_state
