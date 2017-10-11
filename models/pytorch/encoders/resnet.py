#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""ResNet encoder.
    This implementation is bases on
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


class ResNetEnocer(torch.nn.Module):
    """ResNet encoder.
    Args:
    """

    def __init__(self):
        super(ResNetEnocer, self).__init__()

        raise NotImplementedError

    def forward(self, inputs):
        raise NotImplementedError
