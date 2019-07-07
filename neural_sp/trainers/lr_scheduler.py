#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Learning rate scheduler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


class LRScheduler(object):
    """Learning rate scheduler (wrapper for optimizer).

    Args:
        optimizer (torch.optim): optimizer
        lr_max (float): maximum of learning rate
        decay_type (str): epoch/metric
            epoch: decay per epoch regardless of validation metric
            metric: decay if validation metric is not improved
        decay_start_epoch (int): the epoch to start decay
        decay_rate (float): the rate to decay the current learning rate
        decay_patient_n_epochs (int): decay learning rate if results have not been
            improved for 'decay_patient_n_epochs'
        lower_better (bool): If True, the lower, the better.
                             If False, the higher, the better.
        best_value (float): the worst value of evaluation metric
        model_size (int):
        warmup_start_lr (float):
        warmup_n_steps (int):
        lr_factor (float):
        noam (bool): schedule for Transformer

    """

    def __init__(self, optimizer, lr_max, decay_type, decay_start_epoch, decay_rate,
                 decay_patient_n_epochs=0, lower_better=True, best_value=10000,
                 model_size=1, warmup_start_lr=0, warmup_n_steps=0,
                 lr_factor=1, noam=False):

        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lower_better = lower_better
        self.best_value = best_value
        self.noam = noam

        # for warmup
        if noam:
            self.decay_type = 'warmup'
            assert warmup_n_steps > 0
            self.lr_init = lr_factor * (model_size ** -0.5)
        else:
            if warmup_n_steps > 0:
                self.lr_init = warmup_start_lr
            else:
                self.lr_init = lr_max
        self.warmup_n_steps = warmup_n_steps
        self.lr = self.lr_init

        # for decay
        self.decay_type = decay_type
        self.decay_start_epoch = decay_start_epoch
        self.decay_rate = decay_rate
        self.decay_patient_n_epochs = decay_patient_n_epochs
        self._not_improved_n_epochs = 0

        self._step = 0

    def step(self):
        self._step += 1
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def warmup(self):
        """Warm up learning rate per step.

        Args:
            epoch (int): the current epoch

        """
        if self.warmup_n_steps > 0 and self._step < self.warmup_n_steps:
            if self.noam:
                # Based on the original transformer paper
                self.lr = self.lr_init * min(self._step ** (-0.5),
                                             self._step * (self.warmup_n_steps ** (-1.5)))
            else:
                # Increase linearly
                self.lr = (self.lr_max - self.lr_init) / self.warmup_n_steps * self._step + self.lr_init

            # Update optimizer
            self._update_optimizer()

    def decay(self, epoch, value):
        """Decay learning rate per epoch.

        Args:
            epoch (int): the current epoch
            value: (float): A value to evaluate

        """
        if not self.lower_better:
            value *= -1

        if epoch < self.decay_start_epoch:
            if self.decay_type == 'metric':
                if value < self.best_value:
                    # Update the best value
                    self.best_value = value
                    # NOTE: not update learning rate here
        else:
            if self.decay_type == 'metric':
                if value < self.best_value:
                    # Improved
                    self.best_value = value
                    self._not_improved_n_epochs = 0
                elif self._not_improved_n_epochs < self.decay_patient_n_epochs:
                    # Not improved, but learning rate will be not decayed
                    self._not_improved_n_epochs += 1
                else:
                    # Not improved, and learning rate will be decayed
                    self._not_improved_n_epochs = 0
                    self.lr *= self.decay_rate
                    self._update_optimizer()

            elif self.decay_type == 'epoch':
                self.lr *= self.decay_rate
                self._update_optimizer()

    def _update_optimizer(self):
        """Update optimizer."""
        for param_group in self.optimizer.param_groups:
            if isinstance(self.optimizer, torch.optim.Adadelta):
                param_group['eps'] = self.lr
            else:
                param_group['lr'] = self.lr
