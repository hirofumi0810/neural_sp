#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import chainer


def np2var(array, use_cuda=False, volatile=False, dtype=None,
           backend='pytorch'):
    """Convert form np.ndarray to Variable.
    Args:
        array (np.ndarray): A tensor of any sizes
        use_cuda (bool, optional): if True, use CUDA
        volatile (bool, optional): if True, the history will not be saved.
            This should be used in inference model for memory efficiency.
        type (string, optional): float or long or int
        backend (string, optional): pytorch or chainer
    Returns:
        array (torch.Variable or list of chainer.Variable):
            pytorch => A tensor of size `[B, T, input_size]`
            chainer => list of `[T_i, input_size]`
    """
    if backend == 'pytorch':
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

    elif backend == 'chainer':
        assert isinstance(array, list)
        array = [chainer.Variable(a, requires_grad=False) for a in array]

        # NOTE: volatile argument is not supported anymore since v2.
        # Instead, use chainer.no_backprop_mode()

        # TODO: dtype, use_cuda

    else:
        raise TypeError('backend must be "pytorch" or "chainer".')

    return array


def var2np(var, backend='pytorch'):
    """Convert form Variable to np.ndarray.
    Args:
        var (Variable):
    Returns:
        np.ndarray
    """
    if backend == 'pytorch':
        return var.data.cpu().numpy()
    elif backend == 'chainer':
        return var.data.to_cpu()
    else:
        raise TypeError('backend must be "pytorch" or "chainer".')
