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
        lr (float): learning rate
        decay_type (str): epoch or metric
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
        transformer (bool):

    """

    def __init__(self, lr, decay_type, decay_start_epoch, decay_rate,
                 decay_patient_n_epochs=0, lower_better=True, best_value=10000,
                 model_size=1, warmup_start_lr=0, warmup_n_steps=4000,
                 lr_factor=1, transformer=False):

        self.lr_max = lr
        self.decay_type = decay_type
        self.decay_start_epoch = decay_start_epoch
        self.decay_rate = decay_rate
        self.decay_patient_n_epochs = decay_patient_n_epochs
        self.not_improved_n_epochs = 0
        self.lower_better = lower_better
        self.best_value = best_value
        self.transformer = transformer

        # for warmup
        if transformer:
            self.decay_type = 'warmup'
            assert warmup_n_steps > 0
            self.lr_init = lr_factor * (model_size ** -0.5)
        else:
            if warmup_start_lr > 0 and warmup_n_steps > 0:
                self.lr_init = warmup_start_lr
            else:
                self.lr_init = lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_n_steps = warmup_n_steps

        self.lr = self.lr_init

    def decay(self, optimizer, epoch, value):
        """Decay learning rate per epoch.

        Args:
            optimizer ():
            epoch (int): the current epoch
            value: (float): A value to evaluate
        Returns:
            optimizer ():

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
                    self.not_improved_n_epochs = 0
                elif self.not_improved_n_epochs < self.decay_patient_n_epochs:
                    # Not improved, but learning rate will be not decayed
                    self.not_improved_n_epochs += 1
                else:
                    # Not improved, and learning rate will be decayed
                    self.not_improved_n_epochs = 0
                    self.lr *= self.decay_rate

                    # Update optimizer
                    for param_group in optimizer.param_groups:
                        if isinstance(optimizer, torch.optim.Adadelta):
                            param_group['eps'] = self.lr
                        else:
                            param_group['lr'] = self.lr

            elif self.decay_type == 'epoch':
                self.lr *= self.decay_rate

                # Update optimizer
                for param_group in optimizer.param_groups:
                    if isinstance(optimizer, torch.optim.Adadelta):
                        param_group['eps'] = self.lr
                    else:
                        param_group['lr'] = self.lr

        return optimizer

    def warmup(self, optimizer, step):
        """Warm up learning rate per step.

        Args:
            optimizer ():
            epoch (int): the current epoch
        Returns:
            optimizer ():

        """
        if self.warmup_start_lr > 0 and not self.transformer:
            # increase linearly
            self.lr = (self.lr_max - self.warmup_start_lr) / self.warmup_n_steps * step + self.lr_init
        else:
            # based on the original transformer paper
            self.lr = self.lr_init * min(step ** (-0.5),
                                         step * (self.warmup_n_steps ** (-1.5)))

        # Update optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr

        return optimizer
