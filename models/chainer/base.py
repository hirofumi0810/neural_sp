#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import chainer
from chainer import Chain
from chainer import optimizers

OPTIMIZER_CLS_NAMES = {
    "sgd": optimizers.MomentumSGD,
    "nesterov": optimizers.NesterovAG,
    "adam": optimizers.Adam,
    "adadelta": optimizers.AdaDelta,
    "adagrad": optimizers.AdaGrad,
    "rmsprop": optimizers.RMSprop
}
# TODO: Add yellowfin


class ModelBase(Chain):
    """A base class for all models. All models have to inherit this class."""

    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

    # def init_weights(self):
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

    def set_optimizer(self, optimizer, learning_rate_init, weight_decay,
                      clip_grad=5,
                      lr_schedule=True, factor=0.1, patience_epoch=5):
        """
        Args:
            optimizer (string):
            learning_rate_init (float): An initial learning rate
            weight_decay (float):
            clip_grad (float, optional):
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

        self.optimizer = OPTIMIZER_CLS_NAMES[optimizer](lr=learning_rate_init)
        self.optimizer.setup(self)
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(clip_grad))

        # if lr_schedule:
        #     # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     scheduler = ReduceLROnPlateau(
        #         optimizer,
        #         mode='min',
        #         factor=factor,
        #         patience=patience_epoch,
        #         verbose=False,
        #         threshold=0.0001,
        #         threshold_mode='rel',
        #         cooldown=0,
        #         min_lr=0,
        #         eps=1e-08)

        return self.optimizer

    def update(self, clip_grad=5.):
        """Update parameters.
        Args:
            clip_grad (float, optional):
        """
        # Clear the parameter gradients
        self.optimizer.target.cleargrads()

        # Backprop gradients
        self.loss.backward()

        # Truncate the graph
        # loss.unchain_backward()

        # Update parameters
        self.optimizer.update()

    def compute_loss(self, logits, labels):
        """Compute loss. However, do not do back-propagation yet.
        Args:
            logits (chianer.Variable): A tensor of size `()`
            labels (chainer.Variable): A tensor of size `()`
        Returns:
            loss:
        """
        assert isinstance(logits, chainer.Variable), 'logits must be chainer.Variable.'
        assert isinstance(labels, chainer.Variable), 'labels must be chainer.Variable.'

        self.loss = self.optimizer.target(logits, labels)
