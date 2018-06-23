#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for all models (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join, isfile, basename
from glob import glob
import pickle

import logging
logger = logging.getLogger('training')

import numpy as np
import chainer
from chainer import Variable
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

    def init_weights(self, parameter_init, distribution,
                     keys=[None], ignore_keys=[None]):
        """Initialize parameters.
        Args:
            parameter_init (float):
            distribution (string): uniform or normal or orthogonal or constant
            keys (list):
            ignore_keys (list):
        """
        for name, param in self.namedparams():
            if keys != [None] and len(list(filter(lambda k: k in name, keys))) == 0:
                continue

            if ignore_keys != [None] and len(list(filter(lambda k: k in name, ignore_keys))) > 0:
                continue

            xp = cuda.get_array_module(param.data)
            if distribution == 'uniform':
                param.data[...] = xp.random.uniform(
                    low=-parameter_init, high=parameter_init, size=param.data.shape)
            elif distribution == 'normal':
                assert parameter_init > 0
                param.data[...] = xp.random.normal(
                    loc=0, scale=parameter_init, size=param.data.shape)
            elif distribution == 'orthogonal':
                flat_shape = (len(param.data), int(
                    np.prod(param.data.shape[1:])))
                a = np.random.normal(size=flat_shape)
                # we do not have cupy.linalg.svd for now
                u, _, v = np.linalg.svd(a, full_matrices=False)
                # TODO: fix bugs
                # pick the one with the correct shape
                q = u if u.shape == flat_shape else v
                param.data[...] = xp.asarray(q.reshape(param.data.shape))
                param.data *= 1.1
            elif distribution == 'constant':
                param.data[...] = xp.asarray(parameter_init)
            else:
                raise NotImplementedError

    def init_forget_gate_bias_with_one(self):
        """Initialize bias in forget gate with 1."""
        for name, param in self.namedparams():
            if 'lstm' in name and ('b1' in name or 'b5' in name):
                param.data[:] = 1.0

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

    def set_cuda(self, deterministic=False, benchmark=True):
        """Set model to the GPU version.
        Args:
            gpu_id (int):
            deterministic (bool):
            benchmark (bool): not used
        """
        if self.use_cuda:
            gpu_id = 0
            # TODO: add multi-GPU mode

            chainer.config.type_check = False

            if deterministic:
                chainer.config.cudnn_deterministic = True
                logger.info('GPU deterministic mode (no cudnn)')
            else:
                chainer.config.cudnn_deterministic = False
                logger.info('GPU mode')
            cuda.get_device_from_id(gpu_id).use()
            self.to_gpu()

            if not chainer.cuda.available:
                raise ValueError('cuda is not available')
            if not chainer.cuda.cudnn_enabled:
                raise ValueError('cudnn is not available')
        else:
            logger.warning('CPU mode')

    def set_optimizer(self, optimizer, learning_rate_init,
                      weight_decay=0, clip_grad_norm=5,
                      lr_schedule=None, factor=None, patience_epoch=None):
        """Set the optimizer and add hooks
        Args:
            optimizer (string): sgd or adam or adadelta or adagrad or rmsprop
            learning_rate_init (float): An initial learning rate
            weight_decay (float): L2 penalty
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

    def save_checkpoint(self, save_path, epoch, step, lr, metric_dev_best,
                        remove_old_checkpoints=False):
        """Save checkpoint.
        Args:
            save_path (string): path to save a model (directory)
            epoch (int): the currnet epoch
            step (int): the current step
            lr (float):
            metric_dev_best (float):
            remove_old_checkpoints (bool): if True, all checkpoints
                other than the best one will be deleted
        Returns:
            model (string): path to the saved model (file)
        """
        model_path = join(save_path, 'model.epoch-' + str(epoch))

        # Remove old checkpoints
        if remove_old_checkpoints:
            for path in glob(join(save_path, 'model.epoch-*')):
                os.remove(path)

        checkpoint = {
            "epoch": epoch,
            "step": step,
            "lr": lr,
            "metric_dev_best": metric_dev_best
        }

        # Save parameters, optimizer, step index etc.
        # serializers.save_npz(model_path, self)
        # serializers.save_npz(
        #     join(save_path, 'optimizer.epoch-' + str(epoch)), self.optimizer)

        serializer = serializers.DictionarySerializer()

        pickled_params = np.frombuffer(
            pickle.dumps(checkpoint), dtype=np.uint8)
        serializer("checkpoint", pickled_params)

        serializer["model"].save(self)
        serializer["optimizer"].save(self.optimizer)

        np.savez_compressed(model_path, **serializer.target)

        logger.info("=> Saved checkpoint (epoch:%d): %s" % (epoch, model_path))

    def load_checkpoint(self, save_path, epoch=-1, restart=False,
                        load_pretrained_model=False):
        """Load checkpoint.
        Args:
            save_path (string): path to the saved models
            epoch (int): if -1 means the last saved model
            restart (bool): if True, restore the save optimizer
            load_pretrained_model (bool): if True, load all parameters
                which match those of the new model's parameters
        Returns:
            epoch (int): the currnet epoch
            step (int): the current step
            lr (float):
            metric_dev_best (float):
        """
        if int(epoch) == -1:
            # Restore the last saved model
            epochs = [(int(basename(x).split('-')[-1].split('.')[0]), x)
                      for x in glob(join(save_path, 'model.*'))]

            if len(epochs) == 0:
                raise ValueError

            epoch = sorted(epochs, key=lambda x: x[0])[-1][0]

        model_path = join(save_path, 'model.epoch-' + str(epoch) + '.npz')

        if isfile(join(model_path)):
            with np.load(model_path) as f:
                deserializer = serializers.NpzDeserializer(f)

                pickled_params = deserializer("checkpoint", None)
                checkpoint = pickle.loads(pickled_params.tobytes())
                # type: HyperParameters

                # Restore parameters
                if load_pretrained_model:
                    logger.info(
                        "=> Loading pre-trained checkpoint (epoch:%d): %s" % (epoch, model_path))

                    # TODO:
                    # pretrained_dict = checkpoint['state_dict']
                    # model_dict = self.state_dict()
                    #
                    # # 1. filter out unnecessary keys and params which do not match size
                    # pretrained_dict = {
                    #     k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
                    # # 2. overwrite entries in the existing state dict
                    # model_dict.update(pretrained_dict)
                    # # 3. load the new state dict
                    # self.load_state_dict(model_dict)

                    # for k in pretrained_dict.keys():
                    #     logger.info(k)

                deserializer["model"].load(self)

                # Restore optimizer
                if restart:
                    if hasattr(self, 'optimizer'):
                        deserializer["optimizer"].load(self.optimizer)
                    else:
                        raise ValueError('Set optimizer.')
                else:
                    print("=> Loading checkpoint (epoch:%d): %s" %
                          (epoch, model_path))

        else:
            raise ValueError("No checkpoint found at %s" % model_path)

        return (checkpoint['epoch'] + 1, checkpoint['step'] + 1,
                checkpoint['lr'], checkpoint['metric_dev_best'])

    def _create_zero_var(self, size, dtype=np.float32):
        """Initialize a variable with zero.
        Args:
            size (tuple):
            fill_value (int or float):
            dtype ():
        Returns:
            (chainer.Variable, float):
        """
        return Variable(self.xp.full(size, 0, dtype=dtype))

    def _create_var(self, size, fill_value=0., dtype=np.float32):
        """Initialize a variable.
        Args:
            size (tuple):
            fill_value (int or float):
            dtype ():
        Returns:
            (chainer.Variable, float):
        """
        return Variable(self.xp.full(size, fill_value, dtype=dtype))

    def np2var(self, array, use_cuda=False, volatile=False, dtype=None):
        """Convert form np.ndarray to Variable.
        Args:
            array (np.ndarray): A tensor of any sizes
            use_cuda (bool): if True, use CUDA
            volatile (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
            type (string): float or long or int
            backend (string): pytorch or chainer
        Returns:
            array (chainer.Variable or list of chainer.Variable):
                pytorch => A tensor of size `[B, T, input_size]`
                chainer => list of `[T_i, input_size]`
        """
        if self.use_cuda:
            if isinstance(array, list):
                raise ValueError
            array = Variable(
                chainer.cuda.to_gpu(array), requires_grad=False)
        else:
            if isinstance(array, list):
                raise ValueError
            array = Variable(array, requires_grad=False)
        return array

    def var2np(slef, var):
        """Convert form Variable to np.ndarray.
        Args:
            var (chainer.Variable):
        Returns:
            np.ndarray
        """
        return chainer.cuda.to_cpu(var.data)
