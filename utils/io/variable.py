#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
# import chainer


def np2var(array, use_cuda=False, volatile=False, dtype=None):
    """Convert form np.ndarray to Variable.
    Args:
        array (np.ndarray): A tensor of any sizes
        use_cuda (bool, optional): if True, use CUDA
        volatile (bool, optional): if True, the history will not be saved.
            This should be used in inference model for memory efficiency.
        type (string, optional): float or long or int
    Returns:
        array (torch.Variable): A tensor of size `[B, T, input_size]`
    """
    array = torch.from_numpy(array)
    if dtype is not None:
        if dtype == 'float':
            array = array.float()
        elif dtype == 'long':
            array = array.long()
        elif dtype == 'int':
            array = array.int()

    array = torch.autograd.Variable(array, requires_grad=False)

    if volatile:
        array.volatile = True

    if use_cuda:
        array = array.cuda()

    return array


def var2np(var):
    """
    Args:
        var (torch.Variable):
    Returns:
        np.ndarray
    """
    return var.data.cpu().numpy()
