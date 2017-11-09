#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile, basename
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from models.pytorch.tmp.lr_scheduler import ReduceLROnPlateau

OPTIMIZER_CLS_NAMES = {
    "sgd": optim.SGD,
    "momentum": optim.SGD,
    "nesterov": optim.SGD,
    "adam": optim.Adam,
    "adadelta": optim.Adadelta,
    "adagrad": optim.Adagrad,
    "rmsprop": optim.RMSprop
}
# TODO: Add yellowfin


class ModelBase(nn.Module):
    """A base class for all models. All models have to inherit this class."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, inputs):
        raise NotImplementedError

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform(
                param.data, a=-self.parameter_init, b=self.parameter_init)

    @property
    def num_params_dict(self):
        if not hasattr(self, '_num_params_dict'):
            self._num_params_dict = {}

            # Register each parameter
            for name, param in self.named_parameters():
                self._num_params_dict[name] = param.view(-1).size(0)

        return self._num_params_dict

    @property
    def total_parameters(self):
        if not hasattr(self, '_num_params'):
            self._num_params = 0

            # Count total parameters
            for name, param in self.named_parameters():
                self._num_params += param.view(-1).size(0)

        return self._num_params

    @property
    def use_cuda(self):
        return torch.cuda.is_available()

    def set_cuda(self, deterministic=False):
        """Set model to the GPU version.
        Args:
            deterministic (bool, optional):
        """
        if self.use_cuda and deterministic:
            print('GPU deterministic mode (no cudnn)')
            torch.backends.cudnn.enabled = False
            # NOTE: this is slower than GPU mode.
        elif self.use_cuda:
            print('GPU mode')
        else:
            print('CPU mode')

        if self.use_cuda:
            self = self.cuda()

    def set_optimizer(self, optimizer, learning_rate_init, weight_decay=0,
                      lr_schedule=True, factor=0.1, patience_epoch=5):
        """Set optimizer.
        Args:
            optimizer (string): sgd or adam or adadelta or adagrad or rmsprop
            learning_rate_init (float): An initial learning rate
            weight_decay (float, optional): L2 penalty
            lr_schedule (bool, optional): if True, wrap optimizer with
                scheduler. Default is True.
            factor (float, optional):
            patience_epoch (int, optional):
        Returns:
            optimizer (Optimizer):
            scheduler ():
        """
        optimizer = optimizer.lower()
        if optimizer not in OPTIMIZER_CLS_NAMES:
            raise ValueError(
                "Optimizer name should be one of [%s], you provided %s." %
                (", ".join(OPTIMIZER_CLS_NAMES), optimizer))

        if optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.parameters(),
                lr=learning_rate_init,
                weight_decay=weight_decay,
                nesterov=False)
        elif optimizer == 'momentum':
            self.optimizer = optim.SGD(
                self.parameters(),
                lr=learning_rate_init,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=False)
        elif optimizer == 'nesterov':
            self.optimizer = optim.SGD(
                self.parameters(),
                lr=learning_rate_init,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=True)
        else:
            self.optimizer = OPTIMIZER_CLS_NAMES[optimizer](
                self.parameters(),
                lr=learning_rate_init,
                weight_decay=weight_decay)

        if lr_schedule:
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=factor,
                patience=patience_epoch,
                verbose=False,
                threshold=0.0001,
                threshold_mode='rel',
                cooldown=0,
                min_lr=0,
                eps=1e-08)
            # TODO: fix bug
        else:
            scheduler = None

        return self.optimizer, scheduler

    def save_checkpoint(self, save_path, epoch):
        """
        Args:
            save_path (string): path to save a model (directory)
            epoch (int): the epoch to save the model
        Returns:
            model (string): path to the saved model (file)
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch
        }
        model_path = join(save_path, 'model.epoch-' + str(epoch))
        torch.save(checkpoint, model_path)
        return model_path

    def load_checkpoint(self, save_path, epoch):
        """
        Args:
            save_path (string):
            epoch (int):
        """
        if int(epoch) == -1:
            models = [(int(basename(x).split('-')[-1]), x)
                      for x in glob(join(save_path, 'model.*'))]

            if len(models) == 0:
                raise ValueError

            # Restore the model in the last eppch
            epoch = sorted(models, key=lambda x: x[0])[-1][0]

        model_path = join(save_path, 'model.epoch-' + str(epoch))
        if isfile(join(model_path)):
            print("=> Loading checkpoint (epoch:%d): %s" %
                  (epoch, model_path))
            checkpoint = torch.load(
                model_path, map_location=lambda storage, loc: storage)
        else:
            raise ValueError("No checkpoint found at %s" % model_path)
        return checkpoint
