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


def cross_entropy_lsm(logits, ys, ylens, lsm_prob, pad):
    """Compute cross entropy loss for label smoothing of sequence-to-sequence models.

    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ys (LongTensor): Indices of labels. `[B, L]`
        ylens (IntTensor): `[B]`
        lsm_prob (float): label smoothing probability
    Returns:
        loss_mean (FloatTensor): `[1]`

    """
    bs, _, vocab = logits.size()

    logits = logits.view(-1, vocab)
    ys = ys.view(-1)
    target_dist = logits.new_zeros(logits.size())
    target_dist.fill_(lsm_prob / (vocab - 1))
    ys_masked = ys.masked_fill(ys == pad, 0)
    target_dist.scatter_(1, ys_masked.unsqueeze(1), 1 - lsm_prob)

    # Compute XE for label smoothing
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -torch.mul(target_dist, log_probs)
    loss_mean = np.sum([loss[b, :ylens[b]].sum() / ylens[b] for b in range(bs)])
    return loss_mean


def distillation(logits_student, probs_teacher, ylens, temperature=1):
    """Compute cross entropy loss for knowledge distillation of sequence-to-sequence models.

    Args:
        logits_student (FloatTensor): `[B, T, vocab]`
        probs_teacher (FloatTensor): `[B, T, vocab]`
        ylens (IntTensor): `[B]`
        temperature (float):
    Returns:
        loss_mean (FloatTensor): `[1]`

    """
    bs, _, vocab = logits_student.size()

    # Compute XE for knowledge distillation
    log_probs_student = F.log_softmax(logits_student / temperature, dim=-1)
    loss = -torch.mul(probs_teacher, log_probs_student)
    loss_mean = np.sum([loss[b, :ylens[b]].sum() / ylens[b] for b in range(bs)])
    return loss_mean


def kldiv_lsm_ctc(logits, ylens):
    """Compute KL divergence loss for label smoothing of CTC and Transducer models.

    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ylens (IntTensor): `[B]`
    Returns:
        loss_mean (FloatTensor): `[1]`

    """
    bs = logits.size(0)
    vocab = logits.size(-1)

    # Create uniform distribution
    log_uniform = logits.new_zeros(logits.size()).fill_(math.log(1 / (vocab - 1)))

    # Compute KL divergence for label smoothing
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = torch.mul(probs, log_probs - log_uniform)
    loss_mean = np.sum([loss[b, :ylens[b]].sum() / ylens[b] for b in range(bs)])
    # assert loss_mean >= 0
    return loss_mean


def focal_loss(logits, ys, ylens, alpha, gamma):
    """Compute focal loss.

    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ys (LongTensor): Indices of labels. `[B, L]`
        ylens (IntTensor): `[B]`
        alpha (float):
        gamma (float):
    Returns:
        loss_mean (FloatTensor): `[1]`

    """
    bs = ys.size(0)

    # Compute focal loss
    log_probs = F.log_softmax(logits, dim=-1)
    probs_inv = -F.softmax(logits, dim=-1) + 1
    loss = -alpha * torch.mul(torch.pow(probs_inv, gamma), log_probs)
    loss_mean = np.sum([loss[b, :ylens[b]].sum() / ylens[b] for b in range(bs)])
    return loss_mean
