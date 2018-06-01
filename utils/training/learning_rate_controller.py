#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decay learning rate per epoch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Controller(object):
    """Controll learning rate per epoch.
    Args:
        learning_rate_init (float): the initial learning rate
        backend (string): pytorch or chainer
        decay_type (string): per_epoch or compare_metric
        decay_start_epoch (int): the epoch to start decay
        decay_rate (float): the rate to decay the current learning rate
        decay_patient_epoch (int): decay learning rate if results have not been
            improved for 'decay_patient_epoch'
        lower_better (bool): If True, the lower, the better.
            If False, the higher, the better.
        worst_value (float): the worst value of evaluation
    """

    def __init__(self, learning_rate_init, backend, decay_type,
                 decay_start_epoch, decay_rate,
                 decay_patient_epoch=1, lower_better=True, worst_value=1):
        self.learning_rate_init = learning_rate_init
        self.backend = backend
        self.decay_type = decay_type
        self.decay_start_epoch = decay_start_epoch
        self.decay_rate = decay_rate
        self.decay_patient_epoch = decay_patient_epoch
        self.not_improved_epoch = 0
        self.lower_better = lower_better
        self.best_value = worst_value

        assert decay_type in ['per_epoch', 'compare_metric']
        assert backend in ['pytorch', 'chainer']

    def decay_lr(self, optimizer, learning_rate, epoch, value):
        """Decay learning rate per epoch.
        Args:
            optimizer:
            learning_rate (float): the current learning rete
            epoch (int): the current epoch
            value: (float) A value to evaluate
        Returns:
            optimizer:
            learning_rate (float): the decayed learning rate
        """
        if not self.lower_better:
            value *= -1

        if epoch < self.decay_start_epoch:
            if self.decay_type == 'compare_metric':
                if value < self.best_value:
                    # Update the best value
                    self.best_value = value
                    # NOTE: not update learning rate here
        else:
            if self.decay_type == 'compare_metric':
                if value < self.best_value:
                    # Improved
                    self.best_value = value
                    self.not_improved_epoch = 0
                elif self.not_improved_epoch < self.decay_patient_epoch:
                    # Not improved, but learning rate will be not decayed
                    self.not_improved_epoch += 1
                else:
                    # Not improved, and learning rate will be decayed
                    self.not_improved_epoch = 0
                    learning_rate = learning_rate * self.decay_rate

                    # Update optimizer
                    if self.backend == 'pytorch':
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate
                    elif self.backend == 'chainer':
                        optimizer.hyperparam.lr = learning_rate

            elif self.decay_type == 'per_epoch':
                learning_rate = learning_rate * self.decay_rate

                # Update optimizer
                if self.backend == 'pytorch':
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate
                elif self.backend == 'chainer':
                    optimizer.hyperparam.lr = learning_rate
                else:
                    raise ValueError(self.backend)

        return optimizer, learning_rate
