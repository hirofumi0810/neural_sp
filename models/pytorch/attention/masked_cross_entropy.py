#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Masked XE loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def sequence_mask(seq_len, max_len=None):
    """
    Args:
        seq_len ():
        max_len ():
    Returns:

    """
    if max_len is None:
        max_len = seq_len.data.max()
    batch_size = seq_len.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if seq_len.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (seq_len.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, labels_seq_len):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (B, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (B, max_len) which contains the index of the true
            class for each corresponding step.
        labels_seq_len: A Variable containing a LongTensor of size (B,)
            which contains the length of each data in a B.
    Returns:
        loss: An average loss value masked by the length.
    """
    labels_seq_len = Variable(torch.LongTensor(labels_seq_len)).cuda()

    # logits_flat: (B * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (B * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (B * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (B * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (B, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (B, max_len)
    mask = sequence_mask(seq_len=labels_seq_len, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / labels_seq_len.float().sum()

    return loss
