#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Training loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

    def __call__(self, model, batch, is_eval=False):
        """
        Args:
            model (torch.nn.Module or chainer.Chain):
            batch (tuple):
            is_eval (bool):
        Returns:
            model (torch.nn.Module or chainer.Chain):
            loss_val (float):
            loss_main_val (float):
            loss_sub_val (float):
            acc_main (float): Token-level accuracy in teacher-forcing in the main task
            acc_sub (float): Token-level accuracy in teacher-forcing in the sub task
        """
        loss_val, loss_main_val, loss_sub_val = 0., 0., 0.
        try:
            # Step for parameter update
            if self.backend == 'pytorch':
                if is_eval:
                    loss, loss_main, loss_sub, acc_main, acc_sub = model(
                        batch['xs'], batch['ys'], batch['ys_sub'], is_eval=True)
                else:
                    model.optimizer.zero_grad()
                    if model.device_id >= 0:
                        torch.cuda.empty_cache()
                    loss, loss_main, loss_sub, acc_main, acc_sub = model(
                        batch['xs'], batch['ys'], batch['ys_sub'])
                    loss.backward()
                    loss.detach()  # Trancate the graph
                    if self.clip_grad_norm > 0:
                        if model.torch_version < 0.4:
                            torch.nn.utils.clip_grad_norm(
                                model.parameters(), self.clip_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), self.clip_grad_norm)
                    model.optimizer.step()
                    # TODO: Add scheduler
                if model.torch_version < 0.4:
                    loss_val = loss.data[0]
                    loss_main_val = loss_main.data[0]
                    loss_sub_val = loss_sub.data[0]
                else:
                    loss_val = loss.item()
                    loss_main_val = loss_main.item()
                    loss_sub_val = loss_sub.item()

            elif self.backend == 'chainer':
                if is_eval:
                    loss, loss_main, loss_sub, acc_main, acc_sub = model(
                        batch['xs'], batch['ys'], batch['ys_sub'], is_eval=True)
                else:
                    model.optimizer.target.cleargrads()
                    loss, loss_main, loss_sub, acc_main, acc_sub = model(
                        batch['xs'], batch['ys'], batch['ys_sub'])
                    loss.backward()
                    loss.unchain_backward()
                    model.optimizer.update()

                loss_val = loss.data
                loss_main_val = loss_main.data
                loss_sub_val = loss_sub.data

            del loss
            del loss_main
            del loss_sub

        except RuntimeError as e:
            logger.warning('!!!Skip mini-batch!!! (max_frame_num: %d, batch: %d)' %
                           (max(len(x) for x in batch['xs']), len(batch['xs'])))
            if self.backend == 'pytorch':
                model.optimizer.zero_grad()
                if model.device_id >= 0:
                    torch.cuda.empty_cache()
            elif self.backend == 'chainer':
                model.optimizer.target.cleargrads()

        if loss_val == INF or loss_val == -INF:
            logger.warning(
                "WARNING: received an inf loss, setting loss value to 0 (total loss).")
            loss_val, loss_main_val, loss_sub_val = 0., 0., 0.

        # Delete features
        del batch

        return model, loss_val, loss_main_val, loss_sub_val, acc_main, acc_sub
