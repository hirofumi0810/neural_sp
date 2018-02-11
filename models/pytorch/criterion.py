#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""My implementation of some criterion (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from models.pytorch.linear import to_onehot


def kl_div_label_smoothing(logits, distribution='uniform'):
    """KL divergence loss for label smoothing.
    Args:
        logits (torch.autograd.Variable, float):
            A tensor of size `[B, T_in, num_classes]`
        distribution (string, optional): uniform
    Returns:
        kl_loss (torch.autograd.Variable, float): A tensor of size `[]`
    """
    batch_size, label_num, num_classes = logits.size()
    if distribution == 'uniform':
        dist = Variable(torch.FloatTensor(
            batch_size, label_num, num_classes).fill_(1 / num_classes))
    elif distribution == 'normal':
        raise NotImplementedError
    else:
        raise NotImplementedError

    if logits.is_cuda:
        dist = dist.cuda()

    kl_loss = F.kl_div(F.softmax(logits, dim=-1), dist,
                       size_average=False, reduce=True)
    # kl_loss = F.kl_div(F.log_softmax(logits, dim=-1), torch.log(dist),
    #                    size_average=False, reduce=True)
    # TODO: compute at log space ?

    return kl_loss


def cross_entropy_label_smoothing(logits, ys, ignore_index=-1,
                                  distribution='uniform'):
    """Cross entropy loss for label smoothing.
    Args:
        logits (torch.autograd.Variable, float):
            A tensor of size `[B, T, num_classes]`
        ys (torch.autograd.Variable, int): A tensor of size `[B, T]`
        ignore_index (int, optional):
        distribution (string, optional): uniform
    Returns:
        xe_loss (torch.autograd.Variable, float): A tensor of size `[]`
    """
    batch_size, label_num, num_classes = logits.size()
    if distribution == 'uniform':
        dist = Variable(torch.FloatTensor(
            batch_size, label_num, num_classes).fill_(np.log(1 / num_classes)))
    elif distribution == 'normal':
        raise NotImplementedError
    else:
        raise NotImplementedError

    if logits.is_cuda:
        dist = dist.cuda()

    assert label_num == ys.size(1)
    log_probs = F.log_softmax(logits, dim=-1)
    xe_loss = 0.
    for i_batch in range(batch_size):
        t = 0
        while t < label_num:
            if ys[i_batch, t].data[0] == ignore_index:
                break

            # Convert to one-hot labels
            y_onehot = to_onehot(
                ys[i_batch: i_batch + 1, t:t + 1], num_classes)
            # `[B, 1]`

            # Compute XE loss
            xe_loss += - \
                (y_onehot * log_probs[i_batch: i_batch + 1, t:t + 1]).sum()

            t += 1

    return xe_loss
