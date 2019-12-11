#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Learning rate scheduler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from logging import getLogger
import torch
# from torch.optim.lr_scheduler import _LRScheduler

from neural_sp.trainers.optimizer import set_optimizer

logger = getLogger('training')


class LRScheduler(object):
    """Learning rate scheduler (wrapper for optimizer).

    Args:
        optimizer (torch.optim): optimizer
        base_lr (float): maximum of learning rate
        decay_type (str): always/metric
            always: decay per epoch regardless of validation metric
            metric: decay if validation metric is not improved
        decay_start_epoch (int): the epoch to start decay
        decay_rate (float): the rate to decay the current learning rate
        decay_patient_n_epochs (int): decay learning rate if results have not been
            improved for 'decay_patient_n_epochs'
        early_stop_patient_n_epochs (int): number of epochs to tolerate stopping training
            when validation perfomance is not improved
        lower_better (bool): If True, the lower, the better.
                             If False, the higher, the better.
        warmup_start_lr (float): initial learning rate for warmup
        warmup_n_steps (int): steps for learning rate warmup
        model_size (int): d_model
        factor (float): factor of learning rate for Transformer
        noam (bool): schedule for Transformer

    """

    def __init__(self, optimizer, base_lr, decay_type, decay_start_epoch, decay_rate,
                 decay_patient_n_epochs=0, early_stop_patient_n_epochs=-1, lower_better=True,
                 warmup_start_lr=0, warmup_n_steps=0,
                 model_size=1, factor=1, noam=False):

        self.optimizer = optimizer
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
        self.lower_better = lower_better
        self.decay_type = decay_type
        self.decay_start_epoch = decay_start_epoch
        self.decay_rate = decay_rate
        self.decay_patient_n_epochs = decay_patient_n_epochs
        self.metric_best = 1e10 if lower_better else -1e10
        self._is_best = False
        self.not_improved_n_epochs = 0
        self.early_stop_patient_n_epochs = early_stop_patient_n_epochs
        self._is_early_stop = False

    @property
    def n_steps(self):
        return self._step

    @property
    def n_epochs(self):
        return self._epoch

    @property
    def is_best(self):
        return self._is_best

    @property
    def is_early_stop(self):
        return self._is_early_stop

    def step(self):
        self._step += 1
        self.optimizer.step()
        if self.noam:
            self._noam_lr()
        else:
            self._warmup_lr()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _noam_lr(self):
        """Warm up and decay learning rate per step based on Transformer."""
        self.lr = self.base_lr * min(self._step ** (-0.5),
                                     self._step * (self.warmup_n_steps ** (-1.5)))
        self._update_lr()

    def _warmup_lr(self):
        """Warm up learning rate per step by incresing linearly."""
        if self.warmup_n_steps > 0 and self._step <= self.warmup_n_steps:
            self.lr = (self.base_lr - self.warmup_start_lr) / \
                self.warmup_n_steps * self._step + self.warmup_start_lr
            self._update_lr()

    def epoch(self, metric=None):
        """Decay learning rate per epoch.

        Args:
            metric: (float): A metric to evaluate

        """
        self._epoch += 1
        self._is_best = False

        if not self.lower_better:
            metric *= -1

        if metric is not None and metric < self.metric_best:
            self.metric_best = metric
            self._is_best = True
            logger.info('||||| Best Score |||||')

        if not self.noam and self._epoch >= self.decay_start_epoch:
            if self.decay_type == 'metric':
                if self._is_best:
                    # Improved
                    self.not_improved_n_epochs = 0
                elif self.not_improved_n_epochs < self.decay_patient_n_epochs:
                    # Not improved, but learning rate is not decayed
                    self.not_improved_n_epochs += 1
                else:
                    # Not improved, and learning rate is decayed
                    self.not_improved_n_epochs = 0
                    self.lr *= self.decay_rate
                    self._update_lr()
                    logger.info('Epoch %d: reducing learning rate to %.7f'
                                % (self._epoch, self.lr))
            elif self.decay_type == 'always':
                self.lr *= self.decay_rate
                self._update_lr()
                logger.info('Epoch %d: reducing learning rate to %.7f'
                            % (self._epoch, self.lr))

    def _update_lr(self):
        """Reduce learning rate."""
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
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.

        """
        self.__dict__.update(state_dict)

    def convert_to_sgd(self, model, lr, weight_decay, decay_type, decay_rate):
        self.lr = lr
        self.decay_type = decay_type
        self.decay_rate = decay_rate

        # weight_decay = self.optimizer.defaults['weight_decay']
        self.optimizer = set_optimizer(model, 'sgd', lr, weight_decay)
        logger.info('========== Convert to SGD ==========')
