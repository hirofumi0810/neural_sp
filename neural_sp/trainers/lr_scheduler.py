#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Learning rate scheduler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
# from torch.optim.lr_scheduler import _LRScheduler


class LRScheduler(object):
    """Learning rate scheduler (wrapper for optimizer).

    Args:
        optimizer (torch.optim): optimizer
        base_lr (float): maximum of learning rate
        decay_type (str): epoch/metric
            epoch: decay per epoch regardless of validation metric
            metric: decay if validation metric is not improved
        decay_start_epoch (int): the epoch to start decay
        decay_rate (float): the rate to decay the current learning rate
        decay_patient_n_epochs (int): decay learning rate if results have not been
            improved for 'decay_patient_n_epochs'
        lower_better (bool): If True, the lower, the better.
                             If False, the higher, the better.
        warmup_start_lr (float): initial learning rate for warmup
        warmup_n_steps (int): steps for learning rate warmup
        model_size (int):
        factor (float):
        noam (bool): schedule for Transformer

    """

    def __init__(self, optimizer, base_lr, decay_type, decay_start_epoch, decay_rate,
                 decay_patient_n_epochs=0, lower_better=True,
                 warmup_start_lr=0, warmup_n_steps=0,
                 model_size=1, factor=1, noam=False):

        self.optimizer = optimizer
        self.lower_better = lower_better
        self.metric_best = 1e10 if lower_better else -1e10
        self.noam = noam

        self._step = 0
        self._epoch = 0

        # for warmup
        if noam:
            self.decay_type = 'warmup'
            assert warmup_n_steps > 0
            self.base_lr = factor * (model_size ** -0.5)
        else:
            self.base_lr = base_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_n_steps = warmup_n_steps
        self.lr = self.base_lr

        # for decay
        self.decay_type = decay_type
        self.decay_start_epoch = decay_start_epoch
        self.decay_rate = decay_rate
        self.decay_patient_n_epochs = decay_patient_n_epochs
        self.not_improved_n_epochs = 0

    def step(self):
        self._step += 1
        self.optimizer.step()
        self._warmup()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _warmup(self):
        """Warm up learning rate per step.

        Args:
            epoch (int): the current epoch

        """
        if self.warmup_n_steps > 0 and self._step < self.warmup_n_steps:
            if self.noam:
                # Based on the original transformer paper
                self.lr = self.base_lr * min(self._step ** (-0.5),
                                             self._step * (self.warmup_n_steps ** (-1.5)))
            else:
                # Increase linearly
                self.lr = (self.base_lr - self.warmup_start_lr) / \
                    self.warmup_n_steps * self._step + self.warmup_start_lr

            # Update optimizer
            self._update_optimizer()

    def epoch(self, metric):
        """Decay learning rate per epoch.

        Args:
            metric: (float): A metric to evaluate

        """
        self._epoch += 1

        if not self.lower_better:
            metric *= -1

        if self._epoch < self.decay_start_epoch:
            if self.decay_type == 'metric':
                if metric < self.metric_best:
                    self.metric_best = metric
                    # NOTE: not update learning rate here
        else:
            if self.decay_type == 'metric':
                if metric < self.metric_best:
                    # Improved
                    self.metric_best = metric
                    self.not_improved_n_epochs = 0
                elif self.not_improved_n_epochs < self.decay_patient_n_epochs:
                    # Not improved, but learning rate will be not decayed
                    self.not_improved_n_epochs += 1
                else:
                    # Not improved, and learning rate will be decayed
                    self.not_improved_n_epochs = 0
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

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

# class NoamLR(_LRScheduler):
#     def __init__(self, optimizer, warmup_n_steps):
#         self.warmup_n_steps = warmup_n_steps
#         super().__init__(optimizer)
#
#     def get_lr(self):
#         last_epoch = max(1, self.last_epoch)
#         scale = self.factor * (self.model_size ** -0.5) * min(last_epoch ** (-0.5),
#                                                               last_epoch * self.warmup_n_steps ** (-1.5))
#         return [base_lr * scale for base_lr in self.base_lrs]
