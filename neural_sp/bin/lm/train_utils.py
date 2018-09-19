#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Training loop for RNNLMs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch

INF = float("inf")

logger = logging.getLogger('training')


class Updater(object):
    """

    Args:
        clip_grad_norm (float):
        backend (str): pytorch or chainer

    """

    def __init__(self, clip_grad_norm):
        self.clip_grad_norm = clip_grad_norm

    def __call__(self, model, ys, bptt, is_eval=False):
        """

        Args:
            model (torch.nn.Module):
            ys (np.ndarray): target labels of size `[B, L]`
            bptt (int):
            is_eval (bool):
        Returns:
            model (torch.nn.Module):
            loss_val (float):
            acc (float):

        """
        # Step for parameter update
        if is_eval:
            loss, acc = model(ys, is_eval=True)
        else:
            loss, acc = model(ys)

            # Truncate the graph
            model.module.optimizer.zero_grad()
            loss.backward()
            loss.detach()
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), self.clip_grad_norm)
            model.module.optimizer.step()
            # TODO(hirofumi): Add scheduler

        loss_val = loss.item()

        del loss

        if loss_val == INF or loss_val == -INF:
            logger.warning("WARNING: received an inf loss.")

        return model, loss_val, acc
