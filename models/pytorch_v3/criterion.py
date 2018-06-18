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


def kl_div_label_smoothing(logits, label_smoothing_prob,
                           distribution='uniform'):
    """KL divergence loss for label smoothing.
    Args:
        logits (torch.autograd.Variable, float):
            A tensor of size `[B, T_in, num_classes]`
        label_smoothing_prob (float, optional):
        distribution (string, optional): uniform
    Returns:
        kl_loss (torch.autograd.Variable, float): A tensor of size `[1]`
    """
    batch_size, label_num, num_classes = logits.size()
    if distribution == 'uniform':
        dist = Variable(torch.FloatTensor(
            batch_size, label_num, num_classes).fill_(1 / num_classes * label_smoothing_prob))
    elif distribution == 'normal':
        raise NotImplementedError
    else:
        raise NotImplementedError

    if logits.is_cuda:
        dist = dist.cuda()

    kl_loss_sum = F.kl_div(F.softmax(logits, dim=-1), dist,
                           size_average=False, reduce=True)
    # kl_loss_sum = F.kl_div(F.log_softmax(logits, dim=-1), torch.log(dist),
    #                    size_average=False, reduce=True)
    # TODO: compute at log-space?

    return kl_loss_sum


def cross_entropy_label_smoothing(logits, ys, y_lens, label_smoothing_prob,
                                  distribution='uniform'):
    """Cross entropy loss for label smoothing.
    Args:
        logits (torch.autograd.Variable, float):
            A tensor of size `[B, T, num_classes]`
        # ys (torch.autograd.Variables, long): A tensor of size `[B, L]`
        ys (list):
        y_lens (list): A list of length `[B]`
        label_smoothing_prob (float, optional):
        distribution (string, optional): uniform
    Returns:
        xe_loss_sum (torch.autograd.Variable, float): A tensor of size `[1]`
    """
    batch_size, label_num, num_classes = logits.size()
    if distribution == 'uniform':
        dist = 1 / num_classes
    elif distribution == 'normal':
        raise NotImplementedError
    else:
        raise NotImplementedError

    log_probs = F.log_softmax(logits, dim=-1)

    xe_loss_sum = np.sum([(- ys[b, :y_lens[b]] * log_probs[b, :y_lens[b]]).sum()
                          for b in range(batch_size)]) * (1 - label_smoothing_prob)

    xe_loss_sum_ls = np.sum([(- dist * log_probs[b, :y_lens[b]]).sum()
                             for b in range(batch_size)]) * label_smoothing_prob

    return xe_loss_sum + xe_loss_sum_ls
