#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""My implementation of some criterion (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from torch.autograd import Variable
import torch.nn.functional as F


def kl_div_lsm(logits, lsm_prob, dist='uniform'):
    """Compute KL divergence loss for label smoothing.

    Args:
        logits (torch.autograd.Variable, float):
            A tensor of size `[B, T_in, n_classes]`
        lsm_prob (float):
        dist (string): uniform or unigram or normal
    Returns:
        kl_loss (torch.autograd.Variable, float): A tensor of size `[1]`

    """
    batch, n_tokens, n_classes = logits.size()
    if dist == 'uniform':
        dist = Variable(logits.data.new(
            batch, n_tokens, n_classes).fill_(1 / n_classes))
        # NOTE: multiply lsm_prob later
    elif dist == 'unigram':
        raise NotImplementedError
    elif dist == 'normal':
        raise NotImplementedError
    else:
        raise NotImplementedError

    if logits.is_cuda:
        dist = dist.cuda()

    kl_loss_sum = F.kl_div(F.softmax(logits, dim=-1), dist,
                           size_average=False, reduce=True)
    # kl_loss_sum = F.kl_div(F.log_softmax(logits, dim=-1), torch.log(dist),
    #                    size_average=False, reduce=True)
    # TODO(hirofumi): compute at log-space?

    return kl_loss_sum


def cross_entropy_lsm(logits, ys, y_lens, lsm_prob, lsm_type, size_average=False):
    """Compute cross entropy loss for label smoothing.

    Args:
        logits (torch.autograd.Variable, float):
            A tensor of size `[B, T, n_classes]`
        ys (torch.autograd.Variable, long): Indices of labels.
            A tensor of size `[B, L]`.
        y_lens (list): A list of length `[B]`
        lsm_prob (float):
        lsm_type (string): uniform or unigram or normal or rnnlm
        size_average (bool):
    Returns:
        xe_loss_sum (torch.autograd.Variable, float): A tensor of size `[1]`

    """
    batch, n_tokens = ys.size()
    n_classes = logits.size(-1)

    if lsm_type == 'uniform':
        fill_val = lsm_prob / n_classes
    elif lsm_type == 'unigram':
        raise NotImplementedError
    elif lsm_type == 'normal':
        raise NotImplementedError
    elif lsm_type == 'rnnlm':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # Create one-hot vector
    ys_ls = Variable(ys.float().data.new(batch, n_tokens, n_classes).fill_(fill_val))
    for b in range(batch):
        for t in range(y_lens[b]):
            ys_ls.data[b, t, ys.data[b, t]] = 1 - lsm_prob

    # Compute XE for label smoothing
    log_probs = F.log_softmax(logits, dim=-1)
    loss = np.sum([(- ys_ls[b, :y_lens[b]] * log_probs[b, :y_lens[b]]).sum()
                   for b in range(batch)])

    if size_average:
        loss /= batch
    return loss
