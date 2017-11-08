#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
# import chainer


def np2var_pytorch(inputs, use_cuda=False, volatile=False, dtype=None):
    """Convert form np.ndarray to Variable.
    Args:
        inputs (np.ndarray): A tensor of size `[B, T, input_size]`
        use_cuda (bool, optional): if True, use CUDA
        volatile (bool, optional): if True, the history will not be saved.
            This should be used in inference model for memory efficiency.
        type (string, optional): float or long or int
    Returns:
        inputs (torch.Variable): A tensor of size `[B, T, input_size]`
    """
    inputs = torch.from_numpy(inputs)
    if dtype is not None:
        if dtype == 'float':
            inputs = inputs.float()
        elif dtype == 'long':
            inputs = inputs.long()
        elif dtype == 'int':
            inputs = inputs.int()

    inputs = torch.autograd.Variable(inputs, requires_grad=False)

    if volatile:
        inputs.volatile = True

    if use_cuda:
        inputs = inputs.cuda()

    return inputs
