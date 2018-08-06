#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Training loop for RNNLMs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
logger = logging.getLogger('training')

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
            loss_mean (float):
            acc (float):

        """
        ys = np.array(batch['ys'])
        batch_size = len(batch['input_names'])

        # Truncate
        ys = ys[:len(ys) // batch_size * batch_size]
        ys = ys.reshape((batch_size, -1))  # ys: `[B, T]`

        try:
            total_loss, total_acc = 0, 0
            num_update = 0
            while True:
                # Step for parameter update
                if self.backend == 'pytorch':
                    if is_eval:
                        loss, acc = model(ys, is_eval=True)
                    else:
                        loss, acc = model(ys)

                        # Truncate the graph
                        model.module.optimizer.zero_grad()
                        # if len(model.device_ids) >= 1:
                        #     torch.cuda.empty_cache()
                        loss.backward()
                        loss.detach()
                        if self.clip_grad_norm > 0:
                            if model.module.torch_version < 0.4:
                                torch.nn.utils.clip_grad_norm(
                                    model.module.parameters(), self.clip_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    model.module.parameters(), self.clip_grad_norm)
                        model.module.optimizer.step()
                        # TODO(hirofumi): Add scheduler

                    if model.module.torch_version < 0.4:
                        total_loss += loss.data[0]
                    else:
                        total_loss += loss.item()
                    total_acc += acc

                elif self.backend == 'chainer':
                    if is_eval:
                        loss_step, acc_step = model(ys, is_eval=True)
                    else:
                        loss_step, acc_step = model(ys)

                        # Truncate the graph
                        model.optimizer.target.cleargrads()
                        loss_step.backward()
                        loss_step.unchain_backward()
                        model.optimizer.update()
                        loss = 0

                    total_loss += loss_step.data
                    total_acc += acc_step

                ys = ys[:, bptt:]
                num_update += 1
                if len(ys[0]) == 0:
                    break

        except RuntimeError:
            logger.warning('!!!Skip mini-batch!!! (max_label_num: %d, batch: %d)' %
                           (ys.shape[1], batch_size))
            if self.backend == 'pytorch':
                model.module.optimizer.zero_grad()
                # if len(model.device_ids) >= 1:
                #     torch.cuda.empty_cache()
            elif self.backend == 'chainer':
                model.optimizer.target.cleargrads()
            total_loss, total_acc = 0., 0.

        if total_loss == INF or total_loss == -INF:
            logger.warning("WARNING: received an inf loss.")

        # Delete features
        del batch

        return model, total_loss / num_update, total_acc / num_update
