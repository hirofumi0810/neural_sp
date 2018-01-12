#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for all models (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join, isfile, basename
from glob import glob

import logging
logger = logging.getLogger('training')

import chainer
from chainer import optimizers
from chainer import serializers
from chainer import cuda

from utils.directory import mkdir


OPTIMIZER_CLS_NAMES = {
    "sgd": optimizers.SGD,
    "momentum": optimizers.MomentumSGD,
    "nesterov": optimizers.NesterovAG,
    "adam": optimizers.Adam,
    "adadelta": optimizers.AdaDelta,
    "adagrad": optimizers.AdaGrad,
    "rmsprop": optimizers.RMSprop,
    "rmsprop_graves": optimizers.RMSpropGraves
}


class ModelBase(chainer.Chain):
    """A base class for all models. All models have to inherit this class."""

    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

    def _inject_weight_noise(self, mean, std):
        raise NotImplementedError

    @property
    def num_params_dict(self):
        if not hasattr(self, '_num_params_dict'):
            self._num_params_dict = {}
            for name, param in self.namedparams():
                self._num_params_dict[name] = param.reshape(-1).shape[0]
        return self._num_params_dict

    @property
    def total_parameters(self):
        if not hasattr(self, '_num_params'):
            self._num_params = 0
            for name, param in self.namedparams():
                self._num_params += param.reshape(-1).shape[0]
        return self._num_params

    @property
    def use_cuda(self):
        return cuda.available and cuda.cudnn_enabled

    def set_cuda(self, gpu_id=0, deterministic=False, benchmark=True):
        """Set model to the GPU version.
        Args:
            gpu_id (int)
            deterministic (bool, optional):
            benchmark (bool, optional):
        """
        if self.use_cuda:
            # if benchmark:
            #     torch.backends.cudnn.benchmark = True
            #     logger.info('GPU mode (benchmark)')
            # elif deterministic:
            #     logger.info('GPU deterministic mode (no cudnn)')
            #     torch.backends.cudnn.enabled = False
            #     # NOTE: this is slower than GPU mode.
            # else:
            #     logger.info('GPU mode')
            cuda.get_device_from_id(gpu_id).use()
            self.to_gpu()
        else:
            logger.info('CPU mode')

    def set_optimizer(self, optimizer, learning_rate_init,
                      weight_decay=0, clip_grad_norm=5,
                      lr_schedule=None, factor=None, patience_epoch=None):
        """Set the optimizer and add hooks
        Args:
            optimizer (string): sgd or adam or adadelta or adagrad or rmsprop
            learning_rate_init (float): An initial learning rate
            weight_decay (float, optional): L2 penalty
            clip_grad_norm (float):
            lr_schedule: not used here
            factor: not used here
            patience_epoch: not used here
        Returns:
            scheduler ():
        """
        optimizer = optimizer.lower()
        if optimizer not in OPTIMIZER_CLS_NAMES:
            raise ValueError(
                "Optimizer name should be one of [%s], you provided %s." %
                (", ".join(OPTIMIZER_CLS_NAMES), optimizer))

        if optimizer == 'adadelta':
            self.optimizer = optimizers.AdaDelta(rho=0.95, eps=1e-6)
            # TODO: check learning rate
        elif optimizer == 'adagrad':
            self.optimizer = optimizers.AdaGrad(
                lr=learning_rate_init, eps=1e-8)
        elif optimizer == 'adam':
            self.optimizer = optimizers.Adam(
                alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
            # TODO: check learning rate
        elif optimizer == 'sgd':
            self.optimizer = optimizers.MomentumSGD(
                lr=learning_rate_init, momentum=0.9)
        elif optimizer == 'nesterov':
            self.optimizer = optimizers.NesterovAG(
                lr=learning_rate_init, momentum=0.9)
        elif optimizer == 'rmsprop':
            self.optimizer = optimizers.RMSprop(
                lr=learning_rate_init, alpha=0.99, eps=1e-8)
        elif optimizer == 'rmspropgraves':
            self.optimizer = optimizers.RMSpropGraves(
                lr=learning_rate_init, alpha=0.95, momentum=0.9, eps=0.0001)
        else:
            raise NotImplementedError

        self.optimizer.setup(self)

        # Add hook
        self.optimizer.add_hook(
            chainer.optimizer.GradientClipping(clip_grad_norm))
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
        # self.optimizer.add_hook(chainer.optimizer.GradientNoise(eta=0.01))

        return None

    def set_save_path(self, save_path):
        # Reset model directory
        model_index = 0
        save_path_tmp = save_path
        while True:
            if isfile(join(save_path_tmp, 'complete.txt')):
                # Training of the first model have been finished
                model_index += 1
                save_path_tmp = save_path + '_' + str(model_index)
            elif isfile(join(save_path_tmp, 'config.yml')):
                # Training of the first model have not been finished yet
                model_index += 1
                save_path_tmp = save_path + '_' + str(model_index)
            else:
                break
        self.save_path = mkdir(save_path_tmp)

    def save_checkpoint(self, save_path, epoch):
        """Save checkpoint.
        Args:
            save_path (string): path to save a model (directory)
            epoch (int): the epoch to save the model
        Returns:
            model (string): path to the saved model (file)
        """
        model_path = join(save_path, 'model.epoch-' + str(epoch))
        serializers.save_npz(model_path, self)
        serializers.save_npz('optimizer.epoch-' + str(epoch), self.optimizer)

        logger.info("=> Saved checkpoint (epoch:%d): %s" %
                    (epoch, model_path))

    def load_checkpoint(self, save_path, epoch):
        """Load checkpoint.
        Args:
            save_path (string):
            epoch (int):
        """
        raise NotImplementedError
        # serializers.load_npz('model.epoch-' + str(epoch), self)

        # if int(epoch) == -1:
        #     models = [(int(basename(x).split('-')[-1]), x)
        #               for x in glob(join(save_path, 'model.*'))]
        #
        #     if len(models) == 0:
        #         raise ValueError
        #
        #     # Restore the model in the last eppch
        #     epoch = sorted(models, key=lambda x: x[0])[-1][0]
        #
        # model_path = join(save_path, 'model.epoch-' + str(epoch))
        # if isfile(join(model_path)):
        #     logger.info("=> Loading checkpoint (epoch:%d): %s" %
        #                 (epoch, model_path))
        #     checkpoint = torch.load(
        #         model_path, map_location=lambda storage, loc: storage)
        # else:
        #     raise ValueError("No checkpoint found at %s" % model_path)
        # return checkpoint
