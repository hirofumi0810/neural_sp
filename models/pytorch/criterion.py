#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""My implementation of some criterion (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from models.pytorch.linear import to_onehot


def kl_div_label_smoothing(logits, label_smoothing_prob,
                           distribution='uniform', size_average=False):
    """KL divergence loss for label smoothing.
    Args:
        logits (torch.autograd.Variable, float):
            A tensor of size `[B, T_in, num_classes]`
        label_smoothing_prob (float, optional):
        distribution (string, optional): uniform
        size_average (bool, optional):
    Returns:
        kl_loss (torch.autograd.Variable, float): A tensor of size `[1]`
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
                       size_average=False, reduce=True) * label_smoothing_prob
    # kl_loss = F.kl_div(F.log_softmax(logits, dim=-1), torch.log(dist),
    #                    size_average=False, reduce=True) * label_smoothing_prob
    # TODO: compute at log-space?

    if size_average:
        kl_loss /= batch_size

    return kl_loss


def cross_entropy_label_smoothing(logits, ys, label_smoothing_prob,
                                  distribution='uniform', size_average=False):
    """Cross entropy loss for label smoothing.
    Args:
        logits (torch.autograd.Variable, float):
            A tensor of size `[B, T, num_classes]`
        ys (torch.autograd.Variable, int): A tensor of size `[B, T]`
        label_smoothing_prob (float, optional):
        distribution (string, optional): uniform
        size_average (bool, optional):
    Returns:
        xe_loss (torch.autograd.Variable, float): A tensor of size `[1]`
    """
    batch_size, label_num, num_classes = logits.size()
    if distribution == 'uniform':
        dist = 1 / num_classes
    elif distribution == 'normal':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # assert label_num == ys.size(1)
    log_probs = F.log_softmax(logits, dim=-1)

    # NOTE: This cannot be used for CTC
    # xe_loss = sum([(- (to_onehot(ys[:, t:t + 1], num_classes) * (1 - label_smoothing_prob) + dist * label_smoothing_prob) * log_probs[:, t:t + 1]).sum()
    #                for t in range(label_num)])
    # print(xe_loss / batch_size)

    xe_loss = sum([(- dist * log_probs[:, t:t + 1]).sum()
                   for t in range(label_num)]) * label_smoothing_prob
    # print(xe_loss / batch_size)

    if size_average:
        xe_loss /= batch_size

    return xe_loss
