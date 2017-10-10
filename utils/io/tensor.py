# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def to_np(x):
    """
    Args:
        x (FloatTensor):
    Returns:
        np.ndarray
    """
    return x.data.cpu().numpy()
