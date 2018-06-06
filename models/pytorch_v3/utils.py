#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable


def tensor2np(x):
    """Convert tensor to np.ndarray.
    Args:
        x (torch.FloatTensor):
    Returns:
        np.ndarray
    """
    return x.cpu().numpy()


def np2var(array, device_id=0, volatile=False):
    """Convert form np.ndarray to Variable.
    Args:
        array (np.ndarray): A tensor of any sizes
        device_id (int): ht index of the device
        volatile (bool):
    Returns:
        var (torch.autograd.Variable):
    """
    assert isinstance(array, np.ndarray)
    var = Variable(torch.from_numpy(array), requires_grad=False)
    if volatile:
        var.volatile = True
    if device_id >= 0:
        var = var.cuda(device_id)
    return var


def var2np(var):
    """Convert form Variable to np.ndarray.
    Args:
        var (torch.autograd.Variable):
    Returns:
        np.ndarray
    """
    return var.data.cpu().numpy()


def pad_list(xs, pad_value=float("nan")):
    """Convert list of Variables to Variable.
    Args:
        xs (list): A list of length `[B]`, which concains Variables of size
            `[T, input_size]`
        pad_value (flaot):
    Returns:
        xs_pad (torch.autograd.Variable): A tensor of size `[B, T, input_size]`
    """
    assert isinstance(xs[0], Variable)
    batch_size = len(xs)
    max_time = max(x.size(0) for x in xs)
    xs_pad = Variable(
        xs[0].data.new(
            batch_size, max_time, * xs[0].size()[1:]).zero_() + pad_value,
        volatile=xs[0].volatile)
    for i in range(batch_size):
        xs_pad[i, :xs[i].size(0)] = xs[i]
    return xs_pad
