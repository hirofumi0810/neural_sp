#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Training loop for RNNLMs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import logging
logger = logging.getLogger('training')

import torch

try:
    import cupy
except:
    logger.warning('Install cupy.')
try:
    import chainer
except:
    logger.warning('Install chainer.')

INF = float("inf")


class Updater(object):
    """
    Args:
        clip_grad_norm (float):
        backend (string): pytorch or chainer
    """

    def __init__(self, clip_grad_norm, backend):
        self.clip_grad_norm = clip_grad_norm
        self.backend = backend

    def __call__(self, model, batch, bptt, is_eval=False):
        """
        Args:
            model (torch.nn.Module or chainer.Chain):
            batch (tuple):
            bptt (int):
            is_eval (bool):
        Returns:
            model (torch.nn.Module or chainer.Chain):
            loss_val (float):
        """
        try:
            ys = np.array(batch['ys'])
            batch_size = len(batch['input_names'])

            # Truncate
            ys = ys[:len(ys) // batch_size * batch_size]
            ys = ys.reshape((batch_size, -1))
            # ys: `[B, T]`

            num_step = ys.shape[1] // bptt
            offset = 0
            loss_val, acc = 0, 0
            for i in range(num_step):
                ys_bptt = ys[:, offset: offset + bptt]
                offset += bptt

                # Step for parameter update
                if self.backend == 'pytorch':
                    if is_eval:
                        loss_bptt, acc_bptt = model(ys_bptt, is_eval=True)
                    else:
                        model.optimizer.zero_grad()
                        if model.device_id >= 0:
                            torch.cuda.empty_cache()
                        loss_bptt, acc_bptt = model(ys_bptt)
                        loss_bptt.backward()
                        loss_bptt.detach()  # Trancate the graph
                        if self.clip_grad_norm > 0:
                            # torch.nn.utils.clip_grad_norm_(
                            #     model.parameters(), self.clip_grad_norm)
                            torch.nn.utils.clip_grad_norm(
                                model.parameters(), self.clip_grad_norm)
                        model.optimizer.step()
                        # TODO: Add scheduler

                    # loss_val += loss_bptt.item()
                    loss_val += loss_bptt.data[0]

                elif self.backend == 'chainer':
                    if is_eval:
                        loss_bptt, acc_bptt = model(ys_bptt, is_eval=True)
                    else:
                        model.optimizer.target.cleargrads()
                        loss_bptt, acc_bptt = model(ys_bptt)
                        loss_bptt.backward()
                        loss_bptt.unchain_backward()
                        model.optimizer.update()
                    loss_val += loss_bptt.data

                acc += acc_bptt
                del loss_bptt

            acc /= num_step

        except RuntimeError as e:
            logger.warning('!!!Skip mini-batch!!! (max_label_num: %d, batch: %d)' %
                           (max(len(x) for x in batch['ys']), len(batch['ys'])))
            if self.backend == 'pytorch':
                model.optimizer.zero_grad()
                if model.device_id >= 0:
                    torch.cuda.empty_cache()
            elif self.backend == 'chainer':
                model.optimizer.target.cleargrads()
            loss_val = 0.
            acc = 0.

        if loss_val == INF or loss_val == -INF:
            logger.warning("WARNING: received an inf loss.")

        # Delete features
        del batch

        return model, loss_val, acc
