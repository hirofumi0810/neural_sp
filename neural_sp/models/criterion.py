#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Criterions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from torch.autograd import Variable
import torch.nn.functional as F


def cross_entropy_lsm(logits, ys, y_lens, lsm_prob, size_average=False):
    """Compute cross entropy loss for label smoothing.

    Args:
        logits (torch.autograd.Variable, float): `[B, T, vocab]`
        ys (torch.autograd.Variable, long): Indices of labels. `[B, L]`.
        y_lens (list): A list of length `[B]`
        lsm_prob (float):
        size_average (bool):
    Returns:
        xe_loss_sum (torch.autograd.Variable, float): `[1]`

    """
    batch_size, num_tokens = ys.size()
    vocab = logits.size(-1)
    fill_val = lsm_prob / (vocab - 1)

    # Create one-hot vector
    ys_lsm = Variable(ys.float().new(batch_size, num_tokens, vocab).fill_(fill_val))
    for b in six.moves.range(batch_size):
        for t in six.moves.range(y_lens[b]):
            ys_lsm[b, t, ys[b, t]] = 1 - lsm_prob

    # Compute XE for label smoothing
    log_probs = F.log_softmax(logits, dim=-1)
    loss = np.sum([(- ys_lsm[b, :y_lens[b]] * log_probs[b, :y_lens[b]]).sum()
                   for b in six.moves.range(batch_size)])
    if size_average:
        loss /= batch_size
    return loss
