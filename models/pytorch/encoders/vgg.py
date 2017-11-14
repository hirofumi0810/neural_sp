#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""VGG encoder.
   This implementation is bases on
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class VGGEncoder(nn.Module):
    """VGG encoder.
    Args:
    """

    def __init__(self):
        super(VGGEncoder, self).__init__()

        raise NotImplementedError

    def forward(self, inputs):
        raise NotImplementedError
