#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Learning rate controller."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


class Controller(object):
    """Controll learning rate per epoch.

    Args:
        learning_rate (float): learning rate
        decay_type (str): epoch or metric
        decay_start_epoch (int): the epoch to start decay
        decay_rate (float): the rate to decay the current learning rate
        decay_patient_n_epochs (int): decay learning rate if results have not been
            improved for 'decay_patient_n_epochs'
        lower_better (bool): If True, the lower, the better.
            If False, the higher, the better.
        best_value (float): the worst value of evaluation metric
        model_size (int):
        warmup_start_learning_rate (float):
        warmup_n_steps (int):
        factor (float):

    """

    def __init__(self, learning_rate, decay_type, decay_start_epoch, decay_rate,
                 decay_patient_n_epochs=0, lower_better=True, best_value=10000,
                 model_size=1, warmup_start_learning_rate=0, warmup_n_steps=4000,
                 factor=1):

        self.lr_max = learning_rate
        self.decay_type = decay_type
        self.decay_start_epoch = decay_start_epoch
        self.decay_rate = decay_rate
        self.decay_patient_n_epochs = decay_patient_n_epochs
        self.not_improved_epoch = 0
        self.lower_better = lower_better
        self.best_value = best_value

        # for warmup
        if decay_type == 'warmup':
            assert warmup_n_steps > 0
            if warmup_start_learning_rate > 0:
                self.lr_init = warmup_start_learning_rate
            else:
                self.lr_init = factor * (model_size ** (-0.5))
        else:
            self.lr_init = learning_rate
        self.warmup_start_lr = warmup_start_learning_rate
        self.warmup_n_steps = warmup_n_steps

    def decay_lr(self, optimizer, lr, epoch, value):
        """Decay learning rate per epoch.

        Args:
            optimizer:
            lr (float): the current learning rete
            epoch (int): the current epoch
            value: (float) A value to evaluate
        Returns:
            optimizer:
            lr (float): the decayed learning rate

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
                    self.not_improved_epoch = 0
                elif self.not_improved_epoch < self.decay_patient_n_epochs:
                    # Not improved, but learning rate will be not decayed
                    self.not_improved_epoch += 1
                else:
                    # Not improved, and learning rate will be decayed
                    self.not_improved_epoch = 0
                    lr *= self.decay_rate

                    # Update optimizer
                    for param_group in optimizer.param_groups:
                        if isinstance(optimizer, torch.optim.Adadelta):
                            param_group['eps'] = lr
                        else:
                            param_group['lr'] = lr

            elif self.decay_type == 'epoch':
                lr *= self.decay_rate

                # Update optimizer
                for param_group in optimizer.param_groups:
                    if isinstance(optimizer, torch.optim.Adadelta):
                        param_group['eps'] = lr
                    else:
                        param_group['lr'] = lr

        return optimizer, lr

    def warmup_lr(self, optimizer, lr, step):
        """Warm up learning rate per step.

        Args:
            optimizer:
            lr (float): the current learning rete
            epoch (int): the current epoch
        Returns:
            optimizer:
            lr (float): the decayed learning rate

        """
        if self.warmup_start_lr > 0:
            # linearly increse
            lr = (self.lr_max - self.warmup_start_lr) / self.warmup_n_steps * step + self.lr_init
        else:
            # based on the original transformer paper
            lr = self.lr_init * min(step ** (-0.5),
                                    step * (self.warmup_n_steps ** (-1.5)))

        # Update optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer, lr
