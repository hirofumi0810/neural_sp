#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Select optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from logging import getLogger
import torch

logger = getLogger('training')


def set_optimizer(model, optimizer, lr, weight_decay=0.):
    """Set optimizer.

    Args:
        model (): model class
        optimizer (str): name of optimizer
        lr (float): learning rate
        weight_decay (float): L2 penalty for weight decay

    Returns:
        opt (torch.optim): optimizer

    """
    parameters = [p for p in model.parameters() if p.requires_grad]
    logger.info("===== Freezed parameters =====")
    for n in [n for n, p in model.named_parameters() if not p.requires_grad]:
        logger.info("%s" % n)

    if optimizer == 'sgd':
        opt = torch.optim.SGD(parameters,
                              lr=lr,
                              weight_decay=weight_decay,
                              nesterov=False)
    elif optimizer == 'momentum':
        opt = torch.optim.SGD(parameters,
                              lr=lr,
                              momentum=0.9,
                              weight_decay=weight_decay,
                              nesterov=False)
    elif optimizer == 'nesterov':
        opt = torch.optim.SGD(parameters,
                              lr=lr,
                              #  momentum=0.9,
                              momentum=0.99,
                              weight_decay=weight_decay,
                              nesterov=True)
    elif optimizer == 'adadelta':
        opt = torch.optim.Adadelta(parameters,
                                   rho=0.9,  # pytorch default
                                   # rho=0.95,  # chainer default
                                   # eps=1e-8,  # pytorch default
                                   # eps=1e-6,  # chainer default
                                   eps=lr,
                                   weight_decay=weight_decay)

    elif optimizer == 'adam':
        opt = torch.optim.Adam(parameters,
                               lr=lr,
                               weight_decay=weight_decay)

    elif optimizer == 'noam':
        opt = torch.optim.Adam(parameters,
                               lr=0,
                               betas=(0.9, 0.98),
                               eps=1e-09,
                               weight_decay=weight_decay)

    elif optimizer == 'adagrad':
        opt = torch.optim.Adagrad(parameters,
                                  lr=lr,
                                  weight_decay=weight_decay)

    elif optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(parameters,
                                  lr=lr,
                                  weight_decay=weight_decay)

    else:
        raise NotImplementedError(optimizer)

    return opt
