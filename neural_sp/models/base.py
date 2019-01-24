#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for all models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob
import logging
import math
import numpy as np
import os
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
logger = logging.getLogger('decoding')


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
        """Initialize parameters.

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

            if dist == 'uniform':
                nn.init.uniform_(p.data, a=-param_init, b=param_init)
            elif dist == 'normal':
                assert param_init > 0
                torch.nn.init.normal(p.data, mean=0, std=param_init)
            elif dist == 'orthogonal':
                if p.dim() >= 2:
                    torch.nn.init.orthogonal(p.data, gain=1)
            elif dist == 'constant':
                torch.nn.init.constant_(p.data, val=param_init)
            elif dist == 'lecun':
                if p.data.dim() == 1:
                    p.data.zero_()  # bias
                elif p.data.dim() == 2:
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
            else:
                raise NotImplementedError(dist)

    def init_forget_gate_bias_with_one(self):
        """Initialize bias in forget gate with 1. See detail in

            https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745

        """
        for n, p in self.named_parameters():
            if 'lstm' in n and 'bias' in n:
                n = p.size(0)
                start, end = n // 4, n // 2
                p.data[start:end].fill_(1.)

    def gaussian_noise_trigger(self):
        self._gaussian_noise = True

    def gaussian_noise(self, mean=0., std=0.0625):
        """Inject Gaussian noise to weight matrices.

        Args:
            mean (float): mean
            std (float): standard deviation

        """
        if self._gaussian_noise:
            for n, p in self.named_parameters():
                # NOTE: skip bias parameters
                if p.data.dim() == 1:
                    continue

                noise = np.random.normal(loc=mean, scale=std, size=p.size())
                noise = torch.FloatTensor(noise)
                if self.use_cuda:
                    noise = noise.cuda(self.device_id)
                p.data += noise

            # m = torch.distributions.Normal(
            #     torch.Tensor([mean]), torch.Tensor([std]))
            # for n, p in self.named_parameters():
            #     noise = m.sample()
            #     if self.use_cuda:
            #         noise = noise.cuda(self.device_id)
            #     p.data += noise

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

    def set_optimizer(self, optimizer, learning_rate_init,
                      weight_decay=0.0, clip_grad_norm=5.0,
                      lr_schedule=True, factor=0.1, patience_epoch=5):
        """Set optimizer.

        Args:
            optimizer (str): sgd or adam or adadelta or adagrad or rmsprop
            learning_rate_init (float): An initial learning rate
            weight_decay (float): L2 penalty
            clip_grad_norm (float): not used here
            lr_schedule (bool): if True, wrap optimizer with
                scheduler. Default is True.
            factor (float):
            patience_epoch (int):
        Returns:
            scheduler ():

        """
        optimizer = optimizer.lower()
        parameters = [p for p in self.parameters() if p.requires_grad]
        names = [n for n, p in self.named_parameters() if p.requires_grad]
        logger.info("===== Update parameters =====")
        for n in names:
            logger.info("%s" % n)

        if optimizer not in OPTIMIZER_CLS_NAMES:
            raise ValueError(
                "Optimizer n should be one of [%s], you provided %s." %
                (", ".join(OPTIMIZER_CLS_NAMES), optimizer))

        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(parameters,
                                             lr=learning_rate_init,
                                             weight_decay=weight_decay,
                                             nesterov=False)
        elif optimizer == 'momentum':
            self.optimizer = torch.optim.SGD(parameters,
                                             lr=learning_rate_init,
                                             momentum=0.9,
                                             weight_decay=weight_decay,
                                             nesterov=False)
        elif optimizer == 'nesterov':
            self.optimizer = torch.optim.SGD(parameters,
                                             lr=learning_rate_init,
                                             momentum=0.9,
                                             weight_decay=weight_decay,
                                             nesterov=True)
        elif optimizer == 'adadelta':
            self.optimizer = torch.optim.Adadelta(
                parameters,
                # rho=0.9,  # pytorch default
                rho=0.95,  # chainer default
                # eps=1e-8,  # pytorch default
                # eps=1e-6,  # chainer default
                eps=learning_rate_init,
                weight_decay=weight_decay)

        else:
            self.optimizer = OPTIMIZER_CLS_NAMES[optimizer](
                parameters,
                lr=learning_rate_init,
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
        scheduler = None

        return scheduler

    def set_save_path(self, save_path):
        # Reset model directory
        model_index = 0
        save_path_tmp = save_path
        while True:
            if os.path.isfile(os.path.join(save_path_tmp, '.done_training')):
                # Training of the first model have been finished
                model_index += 1
                save_path_tmp = save_path + '_' + str(model_index)
            elif os.path.isfile(os.path.join(save_path_tmp, 'config.yml')):
                # Training of the first model have not been finished yet
                model_index += 1
                save_path_tmp = save_path + '_' + str(model_index)
            else:
                break
        if not os.path.isdir(save_path_tmp):
            os.mkdir(save_path_tmp)
        self.save_path = save_path_tmp

    def save_checkpoint(self, save_path, epoch, step, lr, metric_dev_best,
                        remove_old_checkpoints=False):
        """Save checkpoint.

        Args:
            save_path (str): path to save a model (directory)
            epoch (int): the currnet epoch
            step (int): the current step
            lr (float):
            metric_dev_best (float):
            remove_old_checkpoints (bool): if True, all checkpoints
                other than the best one will be deleted
        Returns:
            model (str): path to the saved model (file)

        """
        model_path = os.path.join(save_path, 'model.epoch-' + str(epoch))

        # Remove old checkpoints
        if remove_old_checkpoints:
            for path in glob(os.path.join(save_path, 'model.epoch-*')):
                os.remove(path)

        # Save parameters, optimizer, step index etc.
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "lr": lr,
            "metric_dev_best": metric_dev_best
        }
        torch.save(checkpoint, model_path)

        logger.info("=> Saved checkpoint (epoch:%d): %s" % (epoch, model_path))

    def load_checkpoint(self, save_path, epoch=-1, restart=False):
        """Load checkpoint.

        Args:
            save_path (str): path to the saved models
            epoch (int): negative values mean the offset from the last saved model
            restart (bool): if True, restore the save optimizer
        Returns:
            epoch (int): the currnet epoch
            step (int): the current step
            lr (float):
            metric_dev_best (float)

        """
        if int(epoch) < 0:
            models = [(int(os.path.basename(x).split('-')[-1]), x)
                      for x in glob(os.path.join(save_path, 'model.*'))]
            if len(models) == 0:
                raise ValueError('There is no checkpoint')

            # Sort in the discending order
            epoch = sorted(models, key=lambda x: x[0])[epoch][0]

        checkpoint_path = os.path.join(save_path, 'model.epoch-' + str(epoch))

        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        else:
            raise ValueError("No checkpoint found at %s" % checkpoint_path)

        # Restore parameters
        self.load_state_dict(checkpoint['state_dict'])

        # Restore optimizer
        if restart:
            logger.info("=> Loading checkpoint (epoch:%d): %s" % (epoch, checkpoint_path))

            if hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(checkpoint['optimizer'])

                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda(self.device_id)
                # NOTE: from https://github.com/pytorch/pytorch/issues/2830
            else:
                raise ValueError('Set optimizer.')
        else:
            logger.info("=> Loading checkpoint (epoch:%d): %s" % (epoch, checkpoint_path))

        return (epoch + 1, checkpoint['step'] + 1,
                checkpoint['lr'], checkpoint['metric_dev_best'])
