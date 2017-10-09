#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
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

    @property
    def use_cuda(self):
        return torch.cuda.is_available()

    def set_optimizer(self, optimizer, learning_rate_init, weight_decay=0,
                      lr_schedule=True, factor=0.1, patience_epoch=5):
        """Set optimizer.
        Args:
            optimizer (string): sgd or adam or adadelta or adagrad or rmsprop
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
                                       momentum=0.9,
                                       weight_decay=weight_decay,
                                       nesterov=False)
            # TODO: make nesterov parameter
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
