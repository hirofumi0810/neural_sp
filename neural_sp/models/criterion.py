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


def cross_entropy_lsm(logits, ys, lsm_prob, ignore_index, training):
    """Compute cross entropy loss for label smoothing of sequence-to-sequence models.

    Args:
        logits (FloatTensor): `[B * T, vocab]`
        ys (LongTensor): Indices of labels. `[B * L]`
        lsm_prob (float): label smoothing probability
        ignore_index (int): index for padding
    Returns:
        loss_mean (FloatTensor): `[1]`

    """
    if lsm_prob == 0 or not training:
        loss = F.cross_entropy(logits, ys.view(-1),
                               ignore_index=ignore_index, reduction='mean')
    else:
        target_dist = logits.new_zeros(logits.size())
        target_dist.fill_(lsm_prob / (logits.size(-1) - 1))
        mask = (ys == ignore_index)
        ys_masked = ys.masked_fill(mask, 0)
        target_dist.scatter_(1, ys_masked.unsqueeze(1), 1 - lsm_prob)

        log_probs = torch.log_softmax(logits, dim=-1)
        loss_sum = -torch.mul(target_dist, log_probs)
        loss = loss_sum.masked_fill(mask.unsqueeze(1), 0).sum() / (len(ys) - mask.sum())
    return loss


def distillation(logits_student, logits_teacher, ylens, temperature=5.0):
    """Compute cross entropy loss for knowledge distillation of sequence-to-sequence models.

    Args:
        logits_student (FloatTensor): `[B, T, vocab]`
        logits_teacher (FloatTensor): `[B, T, vocab]`
        ylens (IntTensor): `[B]`
        temperature (float):
    Returns:
        loss_mean (FloatTensor): `[1]`

    """
    bs, _, vocab = logits_student.size()

    log_probs_student = torch.log_softmax(logits_student, dim=-1)
    probs_teacher = torch.softmax(logits_teacher / temperature, dim=-1).data
    loss = -torch.mul(probs_teacher, log_probs_student)
    loss_mean = np.sum([loss[b, :ylens[b], :].sum() for b in range(bs)]) / ylens.sum()
    return loss_mean


def kldiv_lsm_ctc(logits, ylens):
    """Compute KL divergence loss for label smoothing of CTC and Transducer models.

    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ylens (IntTensor): `[B]`
    Returns:
        loss_mean (FloatTensor): `[1]`

    """
    bs, _, vocab = logits.size()

    log_uniform = logits.new_zeros(logits.size()).fill_(math.log(1 / (vocab - 1)))
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = torch.mul(probs, log_probs - log_uniform)
    loss_mean = np.sum([loss[b, :ylens[b], :].sum() for b in range(bs)]) / ylens.sum()
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

    log_probs = torch.log_softmax(logits, dim=-1)
    probs_inv = -torch.softmax(logits, dim=-1) + 1
    loss = -alpha * torch.mul(torch.pow(probs_inv, gamma), log_probs)
    loss_mean = np.sum([loss[b, :ylens[b], :].sum() for b in range(bs)]) / ylens.sum()
    return loss_mean
