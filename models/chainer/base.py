#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import chainer
from chainer import Chain
from chainer import optimizers
from chainer import serializers


class ModelBase(Chain):
    """A base class for all models. All models have to inherit this class."""

    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

    def init_weights(self):
        raise NotImplementedError
    #     self.num_params = 0
    #     for name, param in self.named_parameters():
    #         nn.init.uniform(
    #             param.data, a=-self.parameter_init, b=self.parameter_init)
    #
    #         # Count total parameters
    #         self.num_params += param.view(-1).size(0)

    @property
    def total_parameters(self):
        return self.num_params

    def set_optimizer(self, optimizer, learning_rate_init):
        """Set the optimizer and add hooks
        Args:
            optimizer (string): adadelta or adagrad or adam or sgd or nesterov
                or rmsprop or rmspropgraves
            learning_rate_init (float): An initial learning rate
        Returns:
            optimizer (Optimizer):
        """
        optimizer = optimizer.lower()

        if optimizer == 'adadelta':
            self.optimizer = optimizers.AdaDelta(rho=0.95, eps=1e-06)
            # NOTE: these are default settings.
            # TODO: check learning rate
        elif optimizer == 'adagrad':
            self.optimizer = optimizers.AdaGrad(lr=learning_rate_init, eps=1e-08)
        elif optimizer == 'adam':
            self.optimizer = optimizers.Adam(
                alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
            # NOTE: these are default settings.
            # TODO: check learning rate
        elif optimizer == 'sgd':
            self.optimizer = optimizers.MomentumSGD(
                lr=learning_rate_init, momentum=0.9)
        elif optimizer == 'nesterov':
            self.optimizer = optimizers.NesterovAG(
                lr=learning_rate_init, momentum=0.9)
        elif optimizer == 'rmsprop':
            self.optimizer = optimizers.RMSprop(
                lr=learning_rate_init, alpha=0.99, eps=1e-08)
        elif optimizer == 'rmspropgraves':
            self.optimizer = optimizers.RMSpropGraves(
                lr=learning_rate_init, alpha=0.95, momentum=0.9, eps=0.0001)
        else:
            raise NotImplementedError

        # TODO: Add learning scheduler

        return self.optimizer

    def compute_loss(self, logits, labels, *args):
        raise NotImplementedError

    def save(self, epoch=None):
        serializers.save_npz('model', self)

    def restore(self, epoch=None):
        serializers.load_npz('model', self)
