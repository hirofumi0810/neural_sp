#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Criterions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import torch
import torch.nn.functional as F

from neural_sp.models.torch_utils import to_onehot


def cross_entropy_lsm(logits, ys, ylens, lsm_prob, size_average=False):
    """Compute cross entropy loss for label smoothing of sequence-to-sequence models.

    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ys (LongTensor): Indices of labels. `[B, L]`.
        ylens (list): A list of length `[B]`
        lsm_prob (float):
        size_average (bool):
    Returns:
        loss (FloatTensor): `[1]`

    """
    bs, _, vocab = logits.size()

    # Create one-hot vector
    ys_lsm = logits.new_zeros(logits.size()).fill_(lsm_prob / (vocab - 1 - 2))
    ys_lsm[:, :, 0] = 0  # blank
    ys_lsm[:, :, 3] = 0  # pad
    for b in range(bs):
        for t in range(ylens[b]):
            ys_lsm[b, t, ys[b, t]] = 1 - lsm_prob

    # Compute XE for label smoothing
    log_probs = F.log_softmax(logits, dim=-1)
    loss = np.sum([(- ys_lsm[b, :ylens[b]] * log_probs[b, :ylens[b]]).sum() for b in range(bs)])
    if size_average:
        loss /= bs
    return loss


def kldiv_lsm_ctc(logits, ylens, size_average=False):
    """Compute KL divergence loss for label smoothing of CTC models.

    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ylens (list): A list of length `[B]`
        size_average (bool):
    Returns:
        loss (FloatTensor): `[1]`

    """
    bs, _, vocab = logits.size()

    # Create uniform distribution
    log_uniform = logits.new_zeros(logits.size()).fill_(math.log(1 / (vocab - 2)))
    log_uniform[:, :, 2] = 0  # eos
    log_uniform[:, :, 3] = 0  # pad

    # Compute XE for label smoothing
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    kl_div = torch.mul(probs, log_probs - log_uniform)
    loss = np.sum([kl_div[b, :ylens[b]].sum() for b in range(bs)])
    if size_average:
        loss /= bs
    return loss


def focal_loss(logits, ys, ylens, gamma, size_average=False):
    """Compute focal loss.

    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ys (LongTensor): Indices of labels. `[B, L]`
        ylens (list): A list of length `[B]`
        gamma (float):
        size_average (bool):
    Returns:
        loss (FloatTensor): `[1]`

    """
    bs = ys.size(0)

    # Create one-hot vector
    ys_onehot = to_onehot(ys, vocab=logits.size(-1), ylens=ylens)

    raise ValueError

    # Compute focal loss
    log_probs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    ones = torch.ones_like(log_probs)
    loss = np.sum([(- ys_onehot[b, :ylens[b]] * log_probs[b, :ylens[b]] * torch.pow(ones[b, :ylens[b]] - probs[b, :ylens[b]], gamma)).sum()
                   for b in range(bs)])
    if size_average:
        loss /= bs
    return loss
