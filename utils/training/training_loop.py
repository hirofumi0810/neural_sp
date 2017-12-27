#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Training loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import numpy as np

import torch.nn as nn


def train_step(model, optimizer, batch, clip_grad_norm):
    """
    Args:
        model (torch.nn.Module):
        optimizer ():
        batch (tuple):
        clip_grad_norm (float):
    Returns:
        model (torch.nn.Module):
        optimizer ():
        loss_val_train (float):
    """
    inputs, labels, inputs_seq_len, labels_seq_len, _ = batch
    del batch

    # Clear gradients before
    optimizer.zero_grad()

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

    except RuntimeError:

        # For scheduled sampling
        if hasattr(model, '_step'):
            model._step -= 1
            step_prev = model._step

        div_num = 2
        while True:
            try:
                print('!!! Divide mini-batch (div_num: %d) !!!' % div_num)

                # For scheduled sampling
                if hasattr(model, '_step'):
                    model._step = step_prev

                # Divide mini-batch
                inputs_div = np.array_split(inputs, div_num, axis=0)
                labels_div = np.array_split(labels, div_num, axis=0)
                inputs_seq_len_div = np.array_split(
                    inputs_seq_len, div_num, axis=0)
                labels_seq_len_div = np.array_split(
                    labels_seq_len, div_num, axis=0)

                loss_val_train = 0
                for i in range(div_num):
                    # Compute loss again
                    loss_train = model(
                        inputs_div[i], labels_div[i], inputs_seq_len_div[i], labels_seq_len_div[i])
                    loss_val_train += loss_train.data[0]

                    # Compute gradient
                    optimizer.zero_grad()
                    loss_train.backward()

                    # Clip gradient norm
                    if clip_grad_norm > 0:
                        nn.utils.clip_grad_norm(
                            model.parameters(), clip_grad_norm)

                    # Update parameters
                    optimizer.step()

            except:
                div_num *= 2

            if div_num > len(inputs):
                break

    del inputs, labels, inputs_seq_len, labels_seq_len
    del loss_train
    gc.collect()

    return model, optimizer, loss_val_train


def train_hierarchical_step(model, optimizer, batch, clip_grad_norm):
    """
    Args:
        model (torch.nn.Module):
        optimizer ():
        batch (tuple):
        clip_grad_norm (float):
    Returns:
        model (torch.nn.Module):
        optimizer ():
        loss_val_train (float):
        loss_main_val_train (float):
        loss_sub_val_train (float):
    """
    inputs, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub, _ = batch
    del batch

    # Clear gradients before
    optimizer.zero_grad()

    loss_val_train, loss_main_val_train, loss_sub_val_train = 0., 0., 0.
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

    except RuntimeError:

        # For scheduled sampling
        if hasattr(model, '_step'):
            model._step -= 1
            step_prev = model._step

        div_num = 2
        while True:
            try:
                print('!!! Divide mini-batch (div_num: %d) !!!' % div_num)

                # For scheduled sampling
                if hasattr(model, '_step'):
                    model._step = step_prev

                # Divide mini-batch
                inputs_div = np.array_split(inputs, div_num, axis=0)
                labels_div = np.array_split(labels, div_num, axis=0)
                labels_sub_div = np.array_split(labels_sub, div_num, axis=0)
                inputs_seq_len_div = np.array_split(
                    inputs_seq_len, div_num, axis=0)
                labels_seq_len_div = np.array_split(
                    labels_seq_len, div_num, axis=0)
                labels_seq_len_sub_div = np.array_split(
                    labels_seq_len_sub, div_num, axis=0)

                loss_val_train = 0
                for i in range(div_num):
                    # Compute loss again
                    loss_train, loss_main_train, loss_sub_train = model(
                        inputs_div[i], labels_div[i], labels_sub_div[i],
                        inputs_seq_len_div[i], labels_seq_len_div[i], labels_seq_len_sub_div[i])
                    loss_val_train += loss_train.data[0]
                    loss_main_val_train += loss_main_train.data[0]
                    loss_sub_val_train += loss_sub_train.data[0]

                    # Compute gradient
                    optimizer.zero_grad()
                    loss_train.backward()

                    # Clip gradient norm
                    if clip_grad_norm > 0:
                        nn.utils.clip_grad_norm(
                            model.parameters(), clip_grad_norm)

                    # Update parameters
                    optimizer.step()
            except:
                div_num *= 2

            if div_num > len(inputs):
                break

    del inputs, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub
    del loss_train, loss_main_train, loss_sub_train
    gc.collect()

    return model, optimizer, loss_val_train, loss_main_val_train, loss_sub_val_train
