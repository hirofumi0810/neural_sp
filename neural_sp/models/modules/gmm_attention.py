#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GMM attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn

NEG_INF = float(np.finfo(np.float32).min)


class GMMAttention(nn.Module):
    def __init__(self, kdim, qdim, adim, n_mixtures, vfloor=1e-6):
        """GMM attention.

        Args:
            kdim (int): dimension of key
            qdim (int): dimension of query
            adim: (int) dimension of the attention layer
            n_mixtures (int): number of mixtures
            vfloor (float):

        """
        super(GMMAttention, self).__init__()

        self.n_mix = n_mixtures
        self.n_heads = 1  # dummy for attention plot
        self.vfloor = vfloor
        self.mask = None
        self.myu = None

        self.ffn_gamma = nn.Linear(qdim, n_mixtures)
        self.ffn_beta = nn.Linear(qdim, n_mixtures)
        self.ffn_kappa = nn.Linear(qdim, n_mixtures)

    def reset(self):
        self.mask = None
        self.myu = None

    def forward(self, key, value, query, mask=None, aw_prev=None, mode=''):
        """Soft monotonic attention during training.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            value (FloatTensor): `[B, klen, value_dim]`
            query (FloatTensor): `[B, 1, qdim]`
            mask (ByteTensor): `[B, qmax, klen]`
            aw_prev (FloatTensor): `[B, klen, 1]`
            mode (str): dummy interface
        Return:
            cv (FloatTensor): `[B, 1, value_dim]`
            alpha (FloatTensor): `[B, klen, 1]`

        """
        bs, klen = key.size()[:2]

        if self.myu is None:
            myu_prev = key.new_zeros(bs, 1, self.n_mix)
        else:
            myu_prev = self.myu

        if self.mask is None:
            self.mask = mask

        w = torch.softmax(self.ffn_gamma(query), dim=-1)  # `[B, 1, n_mix]`
        v = torch.exp(self.ffn_beta(query))  # `[B, 1, n_mix]`
        myu = torch.exp(self.ffn_kappa(query)) + myu_prev  # `[B, 1, n_mix]`
        self.myu = myu  # register for the next step

        # Compute attention weights
        device_id = torch.cuda.device_of(next(self.parameters())).idx
        js = torch.arange(klen).unsqueeze(0).unsqueeze(2).repeat([bs, 1, self.n_mix])
        js = js.cuda(device_id).float()
        numerator = torch.exp(-torch.pow(js - myu, 2) / (2 * v + self.vfloor))
        denominator = torch.pow(2 * math.pi * v + self.vfloor, 0.5)
        aw = w * numerator / denominator  # `[B, klen, n_mix]`
        aw = aw.sum(2)  # `[B, klen]`

        # Compute context vector
        if self.mask is not None:
            aw = aw.masked_fill_(self.mask == 0, NEG_INF)
        cv = torch.bmm(aw.unsqueeze(1), value)

        return cv, aw.unsqueeze(2)
