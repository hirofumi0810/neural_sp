#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Swish activation.
   See details in https://arxiv.org/abs/1710.05941."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
