#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""ResNet encoder (pytorch).
   This implementation is bases on
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class ResNetEncoder(nn.Module):
    """ResNet encoder.
    Args:
    """

    def __init__(self):
        super(ResNetEncoder, self).__init__()

        raise NotImplementedError

    def forward(self, inputs):
        raise NotImplementedError
