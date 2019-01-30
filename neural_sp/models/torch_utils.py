#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def tensor2np(x):
    """Convert tensor to np.ndarray.

    Args:
        x (FloatTensor):
    Returns:
        np.ndarray

    """
    return x.cpu().numpy()


def np2tensor(array, device_id=-1):
    """Convert form np.ndarray to Variable.

    Args:
        array (np.ndarray): A tensor of any sizes
        device_id (int): ht index of the device
    Returns:
        var (Tensor):

    """
    # assert isinstance(array, np.ndarray)
    # var = torch.from_numpy(array).pin_memory())
    var = torch.from_numpy(array)
    if device_id < 0:
        return var
    # return var.cuda(device_id, async=True)
    return var.cuda(device_id)


def pad_list(xs, pad_value=0.0):
    """Convert list of Tensors to a single Tensor with padding.

    Args:
        xs (list): A list of length `[B]`, which concains Tensors of size `[T, input_size]`
        pad_value (float):
    Returns:
        xs_pad (FloatTensor): `[B, T, input_size]`

    """
    bs = len(xs)
    max_time = max(x.size(0) for x in xs)
    xs_pad = xs[0].new_zeros(bs, max_time, * xs[0].size()[1:]).fill_(pad_value)
    for b in range(bs):
        xs_pad[b, :xs[b].size(0)] = xs[b]
    return xs_pad
