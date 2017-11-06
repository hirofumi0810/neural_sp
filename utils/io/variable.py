#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import chainer


def np2var_pytorch(inputs, volatile=False, dtype='float'):
    """Convert form np.ndarray to Variable.
    Args:
        inputs (np.ndarray): A tensor of size `[B, T, input_size]`
        volatile (bool, optional):
        type (string, optional): float or long or int
    Returns:
        inputs (torch.Variable): A tensor of size `[B, T, input_size]`
    """
    inputs = torch.from_numpy(inputs)
    if dtype == 'float':
        inputs = inputs.float()
    elif dtype == 'long':
        inputs = inputs.long()
    elif dtype == 'int':
        inputs = inputs.int()

    inputs = torch.autograd.Variable(inputs, requires_grad=False)
    # NOTE: which is better, 32-bit or 64-bit?

    if volatile:
        inputs.volatile = True

    return inputs
