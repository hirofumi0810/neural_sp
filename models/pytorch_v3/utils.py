#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


def np2var(array, device_id=-1, async=True, volatile=False):
    """Convert form np.ndarray to Variable.
    Args:
        array (np.ndarray): A tensor of any sizes
        async= (bool):
        device_id (int): ht index of the device
        volatile (bool):
    Returns:
        var (torch.autograd.Variable):
    """
    # assert isinstance(array, np.ndarray)
    if async:
        var = Variable(torch.from_numpy(array).pin_memory(),
                       requires_grad=False)
    else:
        var = Variable(torch.from_numpy(array),
                       requires_grad=False)
    if volatile:
        var.volatile = True
    if device_id < 0:
        return var
    return var.cuda(device_id, async=async)


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
    # assert isinstance(xs[0], Variable)
    batch_size = len(xs)
    max_time = max(x.size(0) for x in xs)
    xs_pad = Variable(
        xs[0].data.new(
            batch_size, max_time, * xs[0].size()[1:]).zero_() + pad_value,
        volatile=xs[0].volatile)
    for b in range(batch_size):
        xs_pad[b, :xs[b].size(0)] = xs[b]
    return xs_pad


def to_onehot(ys, num_classes, y_lens=None):
    """Convert indices into one-hot encoding.
    Args:
        ys (torch.autograd.Variable, long): Indices of labels.
            A tensor of size `[B, L]`.
        num_classes (int): the number of classes
        y_lens (list):
    Returns:
        ys (torch.autograd.Variable, float): A tensor of size
            `[B, L, num_classes]`
    """
    batch_size, num_tokens = ys.size()[:2]

    ys_onehot = Variable(ys.float().data.new(
        batch_size, num_tokens, num_classes).fill_(0.))
    for b in range(batch_size):
        for t in range(num_tokens if y_lens is None else y_lens[b]):
            ys_onehot.data[b, t, ys.data[b, t]] = 1.

    if ys.volatile:
        ys_onehot.volatile = True
    return ys_onehot
