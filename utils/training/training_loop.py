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
        # Divide mini-batch
        print('!!! Divide mini-batch !!!')
        inputs = np.array_split(inputs, 2, axis=0)
        labels = np.array_split(labels, 2, axis=0)
        inputs_seq_len = np.array_split(inputs_seq_len, 2, axis=0)
        labels_seq_len = np.array_split(labels_seq_len, 2, axis=0)

        # For scheduled sampling
        if hasattr(model, '_step'):
            model._step -= 1

        for inputs_i, labels_i, inputs_seq_len_i, labels_seq_len_i in zip(
                inputs, labels, inputs_seq_len, labels_seq_len):
            # Compute loss again
            loss_train, loss_main_train, loss_sub_train = model(
                inputs_i, labels_i, inputs_seq_len_i, labels_seq_len_i)
            loss_val_train += loss_train.data[0]

            # Compute gradient
            optimizer.zero_grad()
            loss_train.backward()

            # Clip gradient norm
            if clip_grad_norm > 0:
                nn.utils.clip_grad_norm(model.parameters(), clip_grad_norm)

            # Update parameters
            optimizer.step()
            # TODO: Add scheduler

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
        # Divide mini-batch
        print('!!! Divide mini-batch !!!')
        inputs = np.array_split(inputs, 2, axis=0)
        labels = np.array_split(labels, 2, axis=0)
        labels_sub = np.array_split(labels_sub, 2, axis=0)
        inputs_seq_len = np.array_split(inputs_seq_len, 2, axis=0)
        labels_seq_len = np.array_split(labels_seq_len, 2, axis=0)
        labels_seq_len_sub = np.array_split(labels_seq_len_sub, 2, axis=0)

        model._step -= 1

        for inputs_i, labels_i, labels_sub_i, inputs_seq_len_i, labels_seq_len_i, labels_seq_len_sub_i in zip(
                inputs, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub):
            # Compute loss again
            loss_train, loss_main_train, loss_sub_train = model(
                inputs_i, labels_i, labels_sub_i, inputs_seq_len_i,
                labels_seq_len_i, labels_seq_len_sub_i)
            loss_val_train += loss_train.data[0]
            loss_main_val_train += loss_main_train.data[0]
            loss_sub_val_train += loss_sub_train.data[0]

            # Compute gradient
            optimizer.zero_grad()
            loss_train.backward()

            # Clip gradient norm
            if clip_grad_norm > 0:
                nn.utils.clip_grad_norm(model.parameters(), clip_grad_norm)

            # Update parameters
            optimizer.step()
            # TODO: Add scheduler

    del inputs, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub
    del loss_train, loss_main_train, loss_sub_train
    gc.collect()

    return model, optimizer, loss_val_train, loss_main_val_train, loss_sub_val_train
