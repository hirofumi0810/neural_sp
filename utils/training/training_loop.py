#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Training loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
logger = logging.getLogger('training')
import numpy as np

import torch
import torch.nn as nn


def train_step(model, optimizer, batch, clip_grad_norm):
    """
    Args:
        model (torch.nn.Module):
        optimizer (torch.optim.Optimizer):
        batch (tuple):
        clip_grad_norm (float):
    Returns:
        model (torch.nn.Module):
        optimizer (torch.optim.Optimizer):
        loss_val_train (float):
        div_num (int):
    """
    inputs, labels, inputs_seq_len, labels_seq_len, _ = batch

    loss_val_train = 0.
    div_num = 1
    try:
        # Compute loss in the training set
        loss_train = model(inputs, labels, inputs_seq_len, labels_seq_len)
        loss_val_train = loss_train.data[0]

        # Compute gradient
        optimizer.zero_grad()
        loss_train.backward()

        # Clip gradient norm
        if clip_grad_norm > 0:
            nn.utils.clip_grad_norm(model.parameters(), clip_grad_norm)

        # Update parameters
        optimizer.step()
        # TODO: Add scheduler

        del loss_train

    except RuntimeError as e:
        if 'out of memory' in str(e):
            div_num *= 2
        else:
            raise RuntimeError

            # while True:
            #     logger.warning('!!! Divide mini-batch (batch_size: %d -> %d) !!!' %
            #                    (len(inputs), len(inputs) // div_num))
            #     try:
            #         # Divide mini-batch
            #         inputs_div = np.array_split(inputs, div_num, axis=0)
            #         labels_div = np.array_split(labels, div_num, axis=0)
            #         inputs_seq_len_div = np.array_split(
            #             inputs_seq_len, div_num, axis=0)
            #         labels_seq_len_div = np.array_split(
            #             labels_seq_len, div_num, axis=0)
            #
            #         loss_val_train = 0
            #         for i in range(div_num):
            #
            #             # torch.cuda.empty_cache()
            #
            #             # Compute loss again
            #             loss_train = model(
            #                 inputs_div[i], labels_div[i],
            #                 inputs_seq_len_div[i], labels_seq_len_div[i])
            #             loss_val_train += loss_train.data[0]
            #
            #             # Compute gradient
            #             optimizer.zero_grad()
            #             loss_train.backward()
            #
            #             # Clip gradient norm
            #             if clip_grad_norm > 0:
            #                 nn.utils.clip_grad_norm(
            #                     model.parameters(), clip_grad_norm)
            #
            #             # Update parameters
            #             optimizer.step()
            #
            #             del loss_train
            #
            #         loss_val_train /= div_num
            #         break
            #
            #     except RuntimeError as e:
            #         if 'out of memory' in str(e):
            #             div_num *= 2
            #         else:
            #             raise RuntimeError
            #
            #     if div_num > len(inputs):
            #         raise RuntimeError('batch size is too small.')

    return model, optimizer, loss_val_train, div_num


def train_hierarchical_step(model, optimizer, batch, clip_grad_norm):
    """
    Args:
        model (torch.nn.Module):
        optimizer (torch.optim.Optimizer):
        batch (tuple):
        clip_grad_norm (float):
    Returns:
        model (torch.nn.Module):
        optimizer (torch.optim.Optimizer):
        loss_val_train (float):
        loss_main_val_train (float):
        loss_sub_val_train (float):
        div_num (int):
    """
    inputs, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub, _ = batch

    loss_val_train, loss_main_val_train, loss_sub_val_train = 0., 0., 0.
    div_num = 1
    try:
        # Compute loss in the training set
        loss_train, loss_main_train, loss_sub_train = model(
            inputs, labels, labels_sub, inputs_seq_len,
            labels_seq_len, labels_seq_len_sub)
        loss_val_train = loss_train.data[0]
        loss_main_val_train = loss_main_train.data[0]
        loss_sub_val_train = loss_sub_train.data[0]

        # Compute gradient
        optimizer.zero_grad()
        loss_train.backward()

        # Clip gradient norm
        if clip_grad_norm > 0:
            nn.utils.clip_grad_norm(model.parameters(), clip_grad_norm)

        # Update parameters
        optimizer.step()
        # TODO: Add scheduler

        del loss_train, loss_main_train, loss_sub_train

    except RuntimeError as e:
        if 'out of memory' in str(e):
            div_num *= 2
        else:
            raise RuntimeError

        # while True:
        #     logger.warning('!!! Divide mini-batch (batch_size: %d -> %d) !!!' %
        #                    (len(inputs), len(inputs) // div_num))
        #     try:
        #         # Divide mini-batch
        #         inputs_div = np.array_split(inputs, div_num, axis=0)
        #         labels_div = np.array_split(labels, div_num, axis=0)
        #         labels_sub_div = np.array_split(labels_sub, div_num, axis=0)
        #         inputs_seq_len_div = np.array_split(
        #             inputs_seq_len, div_num, axis=0)
        #         labels_seq_len_div = np.array_split(
        #             labels_seq_len, div_num, axis=0)
        #         labels_seq_len_sub_div = np.array_split(
        #             labels_seq_len_sub, div_num, axis=0)
        #
        #         loss_val_train, loss_main_val_train, loss_sub_val_train = 0., 0., 0.
        #         for i in range(div_num):
        #
        #             torch.cuda.empty_cache()
        #             optimizer.zero_grad()
        #
        #             # Compute loss again
        #             loss_train, loss_main_train, loss_sub_train = model(
        #                 inputs_div[i], labels_div[i], labels_sub_div[i],
        #                 inputs_seq_len_div[i],
        #                 labels_seq_len_div[i], labels_seq_len_sub_div[i])
        #             loss_val_train += loss_train.data[0]
        #             loss_main_val_train += loss_main_train.data[0]
        #             loss_sub_val_train += loss_sub_train.data[0]
        #
        #             # Compute gradient
        #             optimizer.zero_grad()
        #             loss_train.backward()
        #
        #             # Clip gradient norm
        #             if clip_grad_norm > 0:
        #                 nn.utils.clip_grad_norm(
        #                     model.parameters(), clip_grad_norm)
        #
        #             # Update parameters
        #             optimizer.step()
        #
        #             del loss_train, loss_main_train, loss_sub_train
        #
        #         loss_val_train /= div_num
        #         loss_main_val_train /= div_num
        #         loss_sub_val_train /= div_num
        #         break
        #     except RuntimeError as e:
        #         if 'out of memory' in str(e):
        #             div_num *= 2
        #         else:
        #             raise RuntimeError
        #
        #     if div_num > len(inputs):
        #         raise RuntimeError('batch size is too small.')

    return (model, optimizer, loss_val_train,
            loss_main_val_train, loss_sub_val_train, div_num)
