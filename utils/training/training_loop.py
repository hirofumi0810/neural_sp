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


def train_step(model, batch, clip_grad_norm, backend):
    """
    Args:
        model (torch.nn.Module or chainer.Chain):
        batch (tuple):
        clip_grad_norm (float):
        backend (string): pytorch or chainer
    Returns:
        model (torch.nn.Module or chainer.Chain):
        loss_train_val (float):
    """
    loss_train_val = 0.
    try:
        # Step for parameter update
        if backend == 'pytorch':
            model.optimizer.zero_grad()
            loss_train = model(batch['xs'], batch['ys'])
            loss_train.backward()
            if clip_grad_norm > 0:
                # torch.nn.utils.clip_grad_norm_(
                #     model.parameters(), clip_grad_norm)
                torch.nn.utils.clip_grad_norm(
                    model.parameters(), clip_grad_norm)
            model.optimizer.step()
            # TODO: Add scheduler

            # loss_train_val = loss_train.item()
            loss_train_val = loss_train.data[0]

        elif backend == 'chainer':
            model.optimizer.target.cleargrads()
            loss_train = model(batch['xs'], batch['ys'])
            loss_train.backward()
            loss_train.unchain_backward()
            model.optimizer.update()

            loss_train_val = loss_train.data

        del loss_train

    except RuntimeError as e:
        logger.warning('!!!Skip mini-batch!!! (max_frame_num: %d, batch: %d)' %
                       (max(len(x) for x in batch['xs']) * model.num_stack, len(batch['xs'])))
        if backend == 'pytorch':
            model.optimizer.zero_grad()
            # if model.device_id >=0:
            #     torch.cuda.empty_cache()
        elif backend == 'chainer':
            model.optimizer.target.cleargrads()

    if loss_train_val == INF or loss_train_val == -INF:
        logger.warning(
            "WARNING: received an inf loss, setting loss value to 0.")
        loss_train_val = 0

    return model, loss_train_val


def train_hierarchical_step(model, batch, clip_grad_norm, backend):
    """
    Args:
        model (torch.nn.Module or chainer.Chain):
        batch (tuple):
        clip_grad_norm (float):
        backend (string): pytorch or chainer
    Returns:
        model (torch.nn.Module or chainer.Chain):
        loss_train_val (float):
        loss_main_train_val (float):
        loss_sub_train_val (float):
    """
    loss_train_val, loss_main_train_val, loss_sub_train_val = 0., 0., 0.
    try:
        # Step for parameter update
        if backend == 'pytorch':
            model.optimizer.zero_grad()
            loss_train, loss_main_train, loss_sub_train = model(
                batch['xs'], batch['ys'])
            loss_train.backward()
            if clip_grad_norm > 0:
                # torch.nn.utils.clip_grad_norm_(
                #     model.parameters(), clip_grad_norm)
                torch.nn.utils.clip_grad_norm(
                    model.parameters(), clip_grad_norm)
            model.optimizer.step()
            # TODO: Add scheduler

            # loss_train_val = loss_train.item()
            # loss_main_train_val = loss_main_train.item()
            # loss_sub_train_val = loss_sub_train.item()
            loss_train_val = loss_train.data[0]
            loss_main_train_val = loss_main_train.data[0]
            loss_sub_train_val = loss_sub_train.data[0]

        elif backend == 'chainer':
            model.optimizer.target.cleargrads()
            loss_train, loss_main_train, loss_sub_train = model(
                batch['xs'], batch['ys'])
            loss_train.backward()
            loss_train.unchain_backward()
            model.optimizer.update()

            loss_train_val = loss_train.data
            loss_main_train_val = loss_main_train.data
            loss_sub_train_val = loss_sub_train.data

        del loss_train, loss_main_train, loss_sub_train

    except RuntimeError as e:
        logger.warning('!!!Skip mini-batch!!! (max_frame_num: %d, batch: %d)' %
                       (max(len(x) for x in batch['xs']) * model.num_stack, len(batch['xs'])))
        if backend == 'pytorch':
            model.optimizer.zero_grad()
            # if model.device_id >= 0:
            #     torch.cuda.empty_cache()
        elif backend == 'chainer':
            model.optimizer.target.cleargrads()

    if loss_train_val == INF or loss_train_val == -INF:
        logger.warning(
            "WARNING: received an inf loss, setting loss value to 0 (total loss).")
        loss_train_val, loss_main_train_val, loss_sub_train_val = 0., 0., 0.

    return model, loss_train_val, loss_main_train_val, loss_sub_train_val
