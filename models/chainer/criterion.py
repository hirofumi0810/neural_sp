#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""My implementation of some criterion (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from chainer import functions as F
from chainer import Variable


def kl_div_label_smoothing(logits, label_smoothing_prob,
                           distribution='uniform', size_average=False):
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
            batch_size, label_num, num_classes).fill_(1 / num_classes * label_smoothing_prob))
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

    if size_average:
        kl_loss = F.sum(kl_loss, axis=0) / len(logits)

    return kl_loss


def cross_entropy_label_smoothing(logits, y_lens, label_smoothing_prob,
                                  distribution='uniform', size_average=False):
    """Cross entropy loss for label smoothing.
    Args:
        logits (chainer.Variable, float):
            A tensor of size `[B, T, num_classes]`
        y_lens (chainer.Variable, int): A tensor of size `[B]`
        label_smoothing_prob (float, optional):
        distribution (string, optional): uniform
        size_average (bool, optional):
    Returns:
        xe_loss (chainer.Variable, float): A tensor of size `[1]`
    """
    batch_size, label_num, num_classes = logits.shape
    if distribution == 'uniform':
        dist = 1 / num_classes * label_smoothing_prob
    elif distribution == 'normal':
        raise NotImplementedError
    else:
        raise NotImplementedError

    log_probs = F.log_softmax(logits)

    xe_loss = sum([F.sum(- (dist * log_probs[b, y_lens[b].data]))
                   for b in range(batch_size)])

    if size_average:
        xe_loss /= batch_size

    return xe_loss
