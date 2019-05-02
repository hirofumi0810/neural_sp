#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Layer normalizazion."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalizazion.

    Args:
        input_dim (float):
        eps (float):

    """

    def __init__(self, input_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_norm = (x - mean) / (std + self.eps)
        return self.gamma * x_norm + self.beta
