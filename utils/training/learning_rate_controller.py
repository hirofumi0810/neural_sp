#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decay learning rate per epoch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Controller(object):
    """Controll learning rate per epoch.
    Args:
        learning_rate_init: A float value, the initial learning rate
        decay_start_epoch: int, the epoch to start decay
        decay_rate: A float value,  the rate to decay the current learning rate
        decay_patient_epoch: int, decay learning rate if results have not been
            improved for 'decay_patient_epoch'
        lower_better: If True, the lower, the better.
                      If False, the higher, the better.
        worst_value: A flaot value, the worst value of evaluation
    """

    def __init__(self, learning_rate_init, decay_start_epoch, decay_rate,
                 decay_patient_epoch=1, lower_better=True, worst_value=1):
        self.learning_rate_init = learning_rate_init
        self.decay_start_epoch = decay_start_epoch
        self.decay_rate = decay_rate
        self.decay_patient_epoch = decay_patient_epoch
        self.not_improved_epoch = 0
        self.lower_better = lower_better
        self.best_value = worst_value

    def decay_lr(self, learning_rate, epoch, value):
        """Decay learning rate per epoch.
        Args:
            learning_rate: A float value, the current learning rete
            epoch: int, the current epoch
            value: A value to evaluate
        Returns:
            learning_rate_decayed: A float value, the decayed learning rate
        """
        if not self.lower_better:
            value *= -1

        if epoch < self.decay_start_epoch:
            if value < self.best_value:
                # Update
                self.best_value = value
            return learning_rate

        if value < self.best_value:
            # Improved
            self.best_value = value
            self.not_improved_epoch = 0
            return learning_rate
        elif self.not_improved_epoch < self.decay_patient_epoch:
            # Not improved, but learning rate will be not decayed
            self.not_improved_epoch += 1
            return learning_rate
        else:
            # Not improved, and learning rate will be decayed
            self.not_improved_epoch = 0
            learning_rate_decayed = learning_rate * self.decay_rate
            return learning_rate_decayed
