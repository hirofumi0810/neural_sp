#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import abc
import torch.nn as nn
import torch.optim as optim
from models.pytorch.tmp.lr_scheduler import ReduceLROnPlateau

OPTIMIZER_CLS_NAMES = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adadelta": optim.Adadelta,
    "adagrad": optim.Adagrad,
    "rmsprop": optim.RMSprop
}
# TODO: Add yellowfin


class ModelBase(nn.Module):
    """A base class for all models. All models have to inherit this class."""

    def __init__(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def init_weights(self):
        self.num_params = 0
        for name, param in self.named_parameters():
            nn.init.uniform(
                param.data, a=-self.parameter_init, b=self.parameter_init)

            # Count total parameters
            self.num_params += param.view(-1).size(0)

    @property
    def total_parameters(self):
        return self.num_params

    def set_optimizer(self, optimizer, learning_rate_init, weight_decay,
                      lr_schedule=True, factor=0.1, patience_epoch=5):
        """
        Args:
            optimizer (string):
            learning_rate_init (float): An initial learning rate
            weight_decay (float):
            lr_schedule (bool, optional): if True, wrap optimizer with
                scheduler. Default is True.
            factor (float, optional):
            patience_epoch (int, optional):
        Returns:
            optimizer (Optimizer):
            scheduler:
        """
        optimizer = optimizer.lower()
        if optimizer not in OPTIMIZER_CLS_NAMES:
            raise ValueError(
                "Optimizer name should be one of [%s], you provided %s." %
                (", ".join(OPTIMIZER_CLS_NAMES), optimizer))

        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(),
                                       lr=learning_rate_init,
                                       weight_decay=weight_decay,
                                       nesterov=False)
        else:
            self.optimizer = OPTIMIZER_CLS_NAMES[optimizer](
                self.parameters(),
                lr=learning_rate_init,
                weight_decay=weight_decay)

        if lr_schedule:
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=factor,
                patience=patience_epoch,
                verbose=False,
                threshold=0.0001,
                threshold_mode='rel',
                cooldown=0,
                min_lr=0,
                eps=1e-08)

        return self.optimizer, scheduler

    def update(self, clip_grad=5.):
        """Update parameters.
        Args:
            clip_grad (float, optional):
        """
        # Backprop gradients
        self.loss.backward()

        # Clip norm of gradients
        # nn.utils.clip_grad_norm(model.parameters(), clip_grad)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)
        # TODO: bind all parameters before clipping
        # ex.) encoder-decoder models
        # TODO: remove optimizer.step()

        # Update parameters
        self.optimizer.step()

    def compute_loss(self, loss_fn, logits, labels):
        """Compute loss. However, do not do back-propagation yet.
        Args:
            loss_fn:
            logits:
            labels:
        Returns:
            loss:
        """
        self.loss = loss_fn(logits, labels)
