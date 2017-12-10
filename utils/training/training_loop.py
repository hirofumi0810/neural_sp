#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Training loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch.nn as nn


def train_step(model, optimizer, batch, clip_grad_norm):

    # Create feed dictionary for next mini batch (train)
    inputs, labels, inputs_seq_len, labels_seq_len, _ = batch

    # Clear gradients before
    optimizer.zero_grad()

    # Compute loss in the training set
    loss_train = model(inputs, labels, inputs_seq_len, labels_seq_len)

    # Compute gradient
    optimizer.zero_grad()
    loss_train.backward()

    # Clip gradient norm
    nn.utils.clip_grad_norm(model.parameters(), clip_grad_norm)

    # Update parameters
    optimizer.step()
    # TODO: Add scheduler

    # del loss_train

    return model, optimizer, loss_train.data[0]


def train_hierarchical_step(model, optimizer, batch, clip_grad_norm):
    # Create feed dictionary for next mini batch (train)
    inputs, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub, _ = data

    # Clear gradients before
    optimizer.zero_grad()

    # Compute loss in the training set
    loss_train, loss_main_train, loss_sub_train = model(
        inputs, labels, labels_sub, inputs_seq_len,
        labels_seq_len, labels_seq_len_sub)

    # Compute gradient
    optimizer.zero_grad()
    loss_train.backward()

    # Clip gradient norm
    nn.utils.clip_grad_norm(model.parameters(), clip_grad_norm)

    # Update parameters
    optimizer.step()
    # TODO: Add scheduler

    # del loss_train
    # del loss_main_train
    # del loss_sub_train

    return model, optimizer, loss_train.data[0], loss_main_train.data[0], loss_sub_train.data[0]
