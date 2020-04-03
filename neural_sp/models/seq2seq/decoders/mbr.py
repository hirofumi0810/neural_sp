#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Minimum Bayes Risk (MBR) training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


class MBR(torch.autograd.Function):
    """Minimum Bayes Risk (MBR) training.

    Args:
        vocab (int): number of nodes in softmax layer

    """
    @staticmethod
    def forward(ctx, log_probs, hyps, exp_risk, grad):
        """Forward pass.

        Args:
            log_probs (FloatTensor): `[N_best, L, vocab]`
            hyps (LongTensor): `[N_best, L]`
            exp_risk (FloatTensor): `[1]` (for forward)
            grad (FloatTensor): `[1]` (for backward)
        Returns:
            loss (FloatTensor): `[1]`

        """
        device_id = torch.cuda.device_of(log_probs).idx
        onehot = torch.eye(log_probs.size(-1)).cuda(device_id)[hyps]
        grads = grad * onehot  # mask out other classes
        # log_probs = log_probs.requires_grad_()
        ctx.save_for_backward(log_probs, grads)
        return exp_risk

    @staticmethod
    def backward(ctx, grad_output):
        input, grads, = ctx.saved_tensors
        input.grad = grads
        return input, None, None, None
