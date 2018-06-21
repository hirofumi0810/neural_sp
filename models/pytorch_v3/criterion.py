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
    """Compute KL divergence loss for label smoothing.
    Args:
        logits (torch.autograd.Variable, float):
            A tensor of size `[B, T_in, num_classes]`
        label_smoothing_prob (float):
        distribution (string): uniform
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


def cross_entropy_label_smoothing(logits, ys, y_lens,
                                  label_smoothing_prob, label_smoothing_type,
                                  size_average=False):
    """Compute cross entropy loss for label smoothing.
    Args:
        logits (torch.autograd.Variable, float):
            A tensor of size `[B, T, num_classes]`
        ys (torch.autograd.Variable, long): Indices of labels.
            A tensor of size `[B, L]`.
        y_lens (list): A list of length `[B]`
        label_smoothing_prob (float):
        label_smoothing_type (string): uniform or unigram
        size_average (bool):
    Returns:
        xe_loss_sum (torch.autograd.Variable, float): A tensor of size `[1]`
    """
    batch_size, num_tokens = ys.size()[:2]
    num_classes = logits.size(-1)

    if label_smoothing_type == 'uniform':
        fill_val = label_smoothing_prob
    if label_smoothing_type == 'unigram':
        fill_val = label_smoothing_prob / num_classes
    elif label_smoothing_type == 'normal':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # Create one-hot vector
    ys_ls = Variable(ys.float().data.new(
        batch_size, num_tokens, num_classes).fill_(fill_val))
    for b in range(batch_size):
        for t in range(y_lens[b]):
            ys_ls.data[b, t, ys.data[b, t]] = 1. * (1 - label_smoothing_prob)
    if ys.volatile:
        ys_ls.volatile = True

    # Compute XE for label smoothing
    log_probs = F.log_softmax(logits, dim=-1)

    loss = np.sum([(- ys_ls[b, :y_lens[b]] * log_probs[b, :y_lens[b]]).sum()
                   for b in range(batch_size)])

    if size_average:
        loss /= batch_size
    return loss
