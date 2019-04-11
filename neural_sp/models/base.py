#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for all models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import numpy as np
import torch
import torch.nn as nn

np.random.seed(1)

OPTIMIZER_CLS_NAMES = {
    "sgd": torch.optim.SGD,
    "momentum": torch.optim.SGD,
    "nesterov": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "rmsprop": torch.optim.RMSprop
}

logger = logging.getLogger('training')


class ModelBase(nn.Module):
    """A base class for all models. All models have to inherit this class."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def torch_version(self):
        return float('.'.join(torch.__version__.split('.')[:2]))

    @property
    def num_params_dict(self):
        if not hasattr(self, '_nparams_dict'):
            self._nparams_dict = {}
            for n, p in self.named_parameters():
                self._nparams_dict[n] = p.view(-1).size(0)
        return self._nparams_dict

    @property
    def total_parameters(self):
        if not hasattr(self, '_nparams'):
            self._nparams = 0
            for n, p in self.named_parameters():
                self._nparams += p.view(-1).size(0)
        return self._nparams

    @property
    def use_cuda(self):
        return torch.cuda.is_available()

    @property
    def device_id(self):
        return torch.cuda.device_of(next(self.parameters()).data).idx

    def init_weights(self, param_init, dist, keys=[None], ignore_keys=[None]):
        """Initialize parameters. All biases are initialized with zero unless otherwise specified.

        Args:
            param_init (float):
            dist (str): uniform or normal or orthogonal or constant
            keys (list):
            ignore_keys (list):

        """
        for n, p in self.named_parameters():
            if keys != [None] and len(list(filter(lambda k: k in n, keys))) == 0:
                continue
            if ignore_keys != [None] and len(list(filter(lambda k: k in n, ignore_keys))) > 0:
                continue

            if p.data.dim() == 1:
                if dist == 'constant':
                    torch.nn.init.constant_(p.data, val=param_init)
                    logger.info('Initialize %s / %s / %.3f' % (n, dist, param_init))
                else:
                    p.data.zero_()
            else:
                if dist == 'uniform':
                    nn.init.uniform_(p.data, a=-param_init, b=param_init)
                elif dist == 'normal':
                    assert param_init > 0
                    torch.nn.init.normal_(p.data, mean=0, std=param_init)
                elif dist == 'orthogonal':
                    torch.nn.init.orthogonal_(p.data, gain=1)
                elif dist == 'constant':
                    torch.nn.init.constant_(p.data, val=param_init)
                elif dist == 'lecun':
                    if p.data.dim() == 2:
                        n = p.data.size(1)  # linear weight
                        p.data.normal_(0, 1. / math.sqrt(n))
                    elif p.data.dim() == 4:
                        # conv weight
                        n = p.data.size(1)
                        for k in p.data.size()[2:]:
                            n *= k
                        p.data.normal_(0, 1. / math.sqrt(n))
                    else:
                        raise NotImplementedError(p.data.dim())
                elif dist == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(p.data, gain=1.0)
                elif dist == 'xavier_normal':
                    torch.nn.init.xavier_normal_(p.data, gain=1.0)
                elif dist == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(p.data, mode='fan_in', nonlinearity='relu')
                elif dist == 'kaiming_normal':
                    torch.nn.init.kaiming_normal_(p.data, mode='fan_in', nonlinearity='relu')
                else:
                    raise NotImplementedError(dist)
                logger.info('Initialize %s / %s / %.3f' % (n, dist, param_init))

    def init_forget_gate_bias_with_one(self):
        """Initialize bias in forget gate with 1. See detail in

            https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745

        """
        for n, p in self.named_parameters():
            if 'lstm' in n and 'bias' in n:
                n = p.size(0)
                start, end = n // 4, n // 2
                p.data[start:end].fill_(1.)

    def set_cuda(self, deterministic=True, benchmark=False):
        """Set model to the GPU version.

        Args:
            deterministic (bool):
            benchmark (bool):

        """
        if self.use_cuda:
            if benchmark:
                torch.backends.cudnn.benchmark = True
                logger.info('GPU mode (benchmark)')
            elif deterministic:
                logger.info('GPU deterministic mode (no cudnn)')
                torch.backends.cudnn.enabled = False
                # NOTE: this is slower than GPU mode.
            else:
                logger.info('GPU mode')
            self = self.cuda(self.device_id)
        else:
            logger.warning('CPU mode')

    def set_optimizer(self, optimizer, learning_rate, weight_decay=0.0,
                      transformer=False):
        """Set optimizer.

        Args:
            optimizer (str): sgd or adam or adadelta or adagrad or rmsprop
            learning_rate (float): learning rate
            weight_decay (float): L2 penalty
            transformer (bool):

        """
        optimizer = optimizer.lower()
        parameters = [p for p in self.parameters() if p.requires_grad]
        logger.info("===== Freezed parameters =====")
        for n in [n for n, p in self.named_parameters() if not p.requires_grad]:
            logger.info("%s" % n)

        if optimizer not in OPTIMIZER_CLS_NAMES:
            raise ValueError(
                "Optimizer n should be one of [%s], you provided %s." %
                (", ".join(OPTIMIZER_CLS_NAMES), optimizer))

        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(parameters,
                                             lr=learning_rate,
                                             weight_decay=weight_decay,
                                             nesterov=False)
        elif optimizer == 'momentum':
            self.optimizer = torch.optim.SGD(parameters,
                                             lr=learning_rate,
                                             momentum=0.9,
                                             weight_decay=weight_decay,
                                             nesterov=False)
        elif optimizer == 'nesterov':
            self.optimizer = torch.optim.SGD(parameters,
                                             lr=learning_rate,
                                             #  momentum=0.9,
                                             momentum=0.99,
                                             weight_decay=weight_decay,
                                             nesterov=True)
        elif optimizer == 'adadelta':
            self.optimizer = torch.optim.Adadelta(
                parameters,
                # rho=0.9,  # pytorch default
                rho=0.95,  # chainer default
                # eps=1e-8,  # pytorch default
                # eps=1e-6,  # chainer default
                eps=learning_rate,
                weight_decay=weight_decay)

        elif optimizer == 'adam':
            if transformer:
                self.optimizer = torch.optim.Adam(
                    parameters,
                    lr=learning_rate,
                    betas=(0.9, 0.997),
                    eps=1e-09,
                    weight_decay=weight_decay)
            else:
                self.optimizer = torch.optim.Adam(
                    parameters,
                    lr=learning_rate,
                    weight_decay=weight_decay)
        else:
            self.optimizer = OPTIMIZER_CLS_NAMES[optimizer](
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay)

        # if lr_schedule:
        #     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     scheduler = ReduceLROnPlateau(self.optimizer,
        #                                   mode='min',
        #                                   factor=factor,
        #                                   patience=patience_epoch,
        #                                   verbose=False,
        #                                   threshold=0.0001,
        #                                   threshold_mode='rel',
        #                                   cooldown=0,
        #                                   min_lr=0,
        #                                   eps=1e-08)
