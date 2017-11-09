# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def tensor2np(x):
    """
    Args:
        x (FloatTensor):
    Returns:
        np.ndarray
    """
    return x.cpu().numpy()
