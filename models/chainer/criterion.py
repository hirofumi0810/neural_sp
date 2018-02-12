#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""My implementation of some criterion (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from chainer import functions as F
from chainer import Variable
from models.chainer.linear import to_onehot


def kl_div_label_smoothing(logits, label_smoothing_prob,
                           distribution='uniform', use_cuda=False):
    """KL divergence loss for label smoothing.
    Args:
        logits (chainer.Variable, float):
            A tensor of size `[B, T_in, num_classes]`
        label_smoothing_prob (float, optional):
        distribution (string, optional): uniform
        size_average (bool, optional):
    Returns:
        kl_loss (chainer.Variable, float): A tensor of size `[1]`
    """
    raise NotImplementedError

    batch_size, label_num, num_classes = logits.shape
    if distribution == 'uniform':
        dist = Variable(torch.FloatTensor(
            batch_size, label_num, num_classes).fill_(1 / num_classes))
    elif distribution == 'normal':
        raise NotImplementedError
    else:
        raise NotImplementedError

    if logits.is_cuda:
        dist.to_gpu()

    kl_loss = F.kl_div(F.softmax(logits), dist,
                       size_average=False, reduce=True)
    # kl_loss = F.kl_div(F.log_softmax(logits, axis=-1), torch.log(dist),
    #                    size_average=False, reduce=True)
    # TODO: compute at log-space?

    return kl_loss


def cross_entropy_label_smoothing(logits, ys, label_smoothing_prob,
                                  distribution='uniform', size_average=False):
    """Cross entropy loss for label smoothing.
    Args:
        logits (chainer.Variable, float):
            A tensor of size `[B, T, num_classes]`
        ys (chainer.Variable, int): A tensor of size `[B, T]`
        label_smoothing_prob (float, optional):
        distribution (string, optional): uniform
        size_average (bool, optional):
    Returns:
        xe_loss (chainer.Variable, float): A tensor of size `[1]`
    """
    batch_size, label_num, num_classes = logits.shape
    if distribution == 'uniform':
        dist = 1 / num_classes
    elif distribution == 'normal':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # assert label_num == ys.shape[1]
    log_probs = F.log_softmax(logits)

    # NOTE: This cannot be used for CTC
    # xe_loss = sum([F.sum(- (to_onehot(ys[:, t:t + 1], num_classes, use_cuda=True) * (1 - label_smoothing_prob) + dist * label_smoothing_prob) * log_probs[:, t:t + 1])
    #                for t in range(label_num)])
    # print(xe_loss / batch_size)

    xe_loss = sum([F.sum(- (dist * log_probs[:, t:t + 1]))
                   for t in range(label_num)]) * label_smoothing_prob
    # print(xe_loss / batch_size)

    if size_average:
        xe_loss /= batch_size

    return xe_loss
