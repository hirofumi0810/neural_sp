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
import numpy as np
import os
import torch
import torch.nn as nn

# from src.models.pytorch_v3.tmp.lr_scheduler import ReduceLROnPlateau
from neural_sp.utils.general import mkdir

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
        raise NotImplementedError

    @property
    def torch_version(self):
        return float('.'.join(torch.__version__.split('.')[:2]))

    @property
    def num_params_dict(self):
        if not hasattr(self, '_num_params_dict'):
            self._num_params_dict = {}
            for name, param in self.named_parameters():
                self._num_params_dict[name] = param.view(-1).size(0)
        return self._num_params_dict

    @property
    def total_parameters(self):
        if not hasattr(self, '_num_params'):
            self._num_params = 0
            for name, param in self.named_parameters():
                self._num_params += param.view(-1).size(0)
        return self._num_params

    @property
    def use_cuda(self):
        return torch.cuda.is_available()

    @property
    def device_id(self):
        return torch.cuda.device_of(next(self.parameters()).data).idx

    def init_weights(self, param_init, dist,
                     keys=[None], ignore_keys=[None]):
        """Initialize parameters.

        Args:
            param_init (float):
            dist (string): uniform or normal or orthogonal or constant
            keys (list):
            ignore_keys (list):

        """
        for name, param in self.named_parameters():
            if keys != [None] and len(list(filter(lambda k: k in name, keys))) == 0:
                continue

            if ignore_keys != [None] and len(list(filter(lambda k: k in name, ignore_keys))) > 0:
                continue

            if dist == 'uniform':
                nn.init.uniform_(param.data, a=-param_init, b=param_init)
            elif dist == 'normal':
                assert param_init > 0
                torch.nn.init.normal(param.data, mean=0, std=param_init)
            elif dist == 'orthogonal':
                if param.dim() >= 2:
                    torch.nn.init.orthogonal(param.data, gain=1)
            elif dist == 'constant':
                torch.nn.init.constant_(param.data, val=param_init)
            else:
                raise NotImplementedError

    def init_forget_gate_bias_with_one(self):
        """Initialize bias in forget gate with 1. See detail in

            https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745

        """
        for name, param in self.named_parameters():
            if 'lstm' in name and 'bias' in name:
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)

    def inject_weight_noise(self, mean, std):
        # m = torch.distributions.Normal(
        #     torch.Tensor([mean]), torch.Tensor([std]))
        # for name, param in self.named_parameters():
        #     noise = m.sample()
        #     if self.use_cuda:
        #         noise = noise.cuda(self.device_id)
        #     param.data += noise

        for name, param in self.named_parameters():
            noise = np.random.normal(loc=mean, scale=std, size=param.size())
            noise = torch.FloatTensor(noise)
            if self.use_cuda:
                noise = noise.cuda(self.device_id)
            param.data += noise

    def set_cuda(self, deterministic=False, benchmark=True):
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
                      weight_decay=0, clip_grad_norm=5,
                      lr_schedule=True, factor=0.1, patience_epoch=5):
        """Set optimizer.

        Args:
            optimizer (string): sgd or adam or adadelta or adagrad or rmsprop
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
                "Optimizer name should be one of [%s], you provided %s." %
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
                # rho=0.9,  # default
                rho=0.95,
                # eps=1e-6,  # default
                eps=1e-8,
                lr=learning_rate_init,
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
            metric_dev_best (float)

        """
        if int(epoch) == -1:
            # Restore the last saved model
            epochs = [(int(os.path.basename(x).split('-')[-1]), x)
                      for x in glob(os.path.join(save_path, 'model.*'))]

            if len(epochs) == 0:
                raise ValueError('There is no checkpoint')

            epoch = sorted(epochs, key=lambda x: x[0])[-1][0]

        checkpoint_path = os.path.join(save_path, 'model.epoch-' + str(epoch))

        try:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        except:
            raise ValueError("No checkpoint found at %s" % checkpoint_path)

        # Restore parameters
        if load_pretrained_model:
            logger.info("=> Loading pre-trained checkpoint (epoch:%d): %s" % (epoch, checkpoint_path))

            pretrained_dict = checkpoint['state_dict']
            model_dict = self.state_dict()

            # 1. filter out unnecessary keys and params which do not match size
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys() and v.size() == model_dict[k].size()}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.load_state_dict(model_dict)

            for k in pretrained_dict.keys():
                logger.info(k)
            logger.info('=> Finished loading.')
        else:
            self.load_state_dict(checkpoint['state_dict'])

        # Restore optimizer
        if restart:
            logger.info("=> Loading checkpoint (epoch:%d): %s" %
                        (epoch, checkpoint_path))

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
            print("=> Loading checkpoint (epoch:%d): %s" %
                  (epoch, checkpoint_path))

        return (checkpoint['epoch'] + 1, checkpoint['step'] + 1,
                checkpoint['lr'], checkpoint['metric_dev_best'])
