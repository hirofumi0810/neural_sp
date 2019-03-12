#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utility functions for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import numpy as np
import os
import seaborn as sns
from tensorboardX import SummaryWriter
import torch
import yaml

plt.style.use('ggplot')
grey = '#878f99'
blue = '#4682B4'
orange = '#D2691E'
green = '#82b74b'

logger = logging.getLogger('training')


class Reporter(object):
    """"Report loss, accuracy etc. during training.

    Args:
        save_path (str):
        tensorboard (bool): use tensorboard logging

    """

    def __init__(self, save_path, tensorboard=True):
        self.save_path = save_path
        self.tensorboard = tensorboard

        if tensorboard:
            self.tf_writer = SummaryWriter(save_path)

        # report per step
        self._step = 0
        self.observation_train = {'loss': {}, 'acc': {}, 'ppl': {}}
        self.observation_train_local = {'loss': {}, 'acc': {}, 'ppl': {}}
        self.observation_dev = {'loss': {}, 'acc': {}, 'ppl': {}}
        self.steps = []

        # report per epoch
        self._epoch = 0
        self.observation_eval = []
        self.epochs = []

    def add(self, observation, is_eval):
        """Restore values per step.

            Args:
                observation (dict):
                is_eval (bool):

        """
        for k, v in observation.items():
            if v is None:
                continue
            metric, name = k.split('.')
            # NOTE: metric: loss, acc, ppl

            if v == float("inf") or v == -float("inf"):
                logger.warning("WARNING: received an inf loss for %s." % k)

            if not is_eval:
                if name not in self.observation_train_local[metric].keys():
                    self.observation_train_local[metric][name] = []
                self.observation_train_local[metric][name].append(v)
            else:
                # avarage for training
                if name not in self.observation_train[metric].keys():
                    self.observation_train[metric][name] = []
                self.observation_train[metric][name].append(
                    np.mean(self.observation_train_local[metric][name]))
                logger.info('%s (train, mean): %.3f' % (k, np.mean(self.observation_train_local[metric][name])))

                if name not in self.observation_dev[metric].keys():
                    self.observation_dev[metric][name] = []
                self.observation_dev[metric][name].append(v)
                logger.info('%s (dev): %.3f' % (k, v))

                # Logging by tensorboard
                if self.tensorboard:
                    if not is_eval:
                        self.tf_writer.add_scalar('train/' + metric + '/' + name, v, self._step)
                    else:
                        self.tf_writer.add_scalar('dev/' + metric + '/' + name, v, self._step)
                # for n, p in model.module.named_parameters():
                #     n = n.replace('.', '/')
                #     if p.grad is not None:
                #         tf_writer.add_histogram(n, p.data.cpu().numpy(), self._step + 1)
                #         tf_writer.add_histogram(n + '/grad', p.grad.data.cpu().numpy(), self._step + 1)

    def step(self, is_eval):
        self._step += 1
        if is_eval:
            self.steps.append(self._step)

            # reset
            self.observation_train_local = {'loss': {}, 'acc': {}, 'ppl': {}}

    def epoch(self, wer):
        self._epoch += 1
        self.epochs.append(self._epoch)

        # register
        self.observation_eval.append(wer)

        plt.clf()
        plt.plot(self.epochs, self.observation_eval, orange,
                 label='dev', linestyle='-')
        plt.xlabel('epoch', fontsize=12)
        plt.ylabel('WER', fontsize=12)
        plt.ylim([0,  max(100, max(self.observation_eval))])
        plt.legend(loc="upper right", fontsize=12)
        if os.path.isfile(os.path.join(self.save_path, 'wer' + ".png")):
            os.remove(os.path.join(self.save_path, 'wer' + ".png"))
        plt.savefig(os.path.join(self.save_path, 'wer' + ".png"), dvi=500)

    def snapshot(self):
        # linestyles = ['solid', 'dashed', 'dotted', 'dashdotdotted']
        linestyles = ['-', '--', '-.', ':', ':', ':', ':', ':', ':', ':', ':', ':']
        for metric in self.observation_train.keys():
            plt.clf()
            upper = 0
            for i, (k, v) in enumerate(sorted(self.observation_train[metric].items())):
                # skip non-observed values
                if np.mean(self.observation_train[metric][k]) == 0:
                    continue

                plt.plot(self.steps, self.observation_train[metric][k], blue,
                         label=k + " (train)", linestyle=linestyles[i])
                plt.plot(self.steps, self.observation_dev[metric][k], orange,
                         label=k + " (dev)", linestyle=linestyles[i])
                upper = max(upper, max(self.observation_train[metric][k]))
                upper = max(upper, max(self.observation_dev[metric][k]))

                # Save as csv file
                if os.path.isfile(os.path.join(self.save_path, metric + '-' + k + ".csv")):
                    os.remove(os.path.join(self.save_path, metric + '-' + k + ".csv"))
                loss_graph = np.column_stack(
                    (self.steps, self.observation_train[metric][k], self.observation_dev[metric][k]))
                np.savetxt(os.path.join(self.save_path, metric + '-' + k + ".csv"), loss_graph, delimiter=",")

            upper = min(upper + 10, 300)

            plt.xlabel('step', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.ylim([0, upper])
            plt.legend(loc="upper right", fontsize=12)
            if os.path.isfile(os.path.join(self.save_path, metric + ".png")):
                os.remove(os.path.join(self.save_path, metric + ".png"))
            plt.savefig(os.path.join(self.save_path, metric + ".png"), dvi=500)


class Controller(object):
    """Controll learning rate per epoch.

    Args:
        learning_rate (float): learning rate
        decay_type (str): epoch or metric
        decay_start_epoch (int): the epoch to start decay
        decay_rate (float): the rate to decay the current learning rate
        decay_patient_epoch (int): decay learning rate if results have not been
            improved for 'decay_patient_epoch'
        lower_better (bool): If True, the lower, the better.
            If False, the higher, the better.
        best_value (float): the worst value of evaluation metric
        model_size (int):
        warmup_start_learning_rate (float):
        warmup_nsteps (int):
        factor (float):

    """

    def __init__(self, learning_rate, decay_type, decay_start_epoch, decay_rate,
                 decay_patient_epoch=0, lower_better=True, best_value=10000,
                 model_size=1, warmup_start_learning_rate=0, warmup_nsteps=4000,
                 factor=1):

        self.lr_max = learning_rate
        self.decay_type = decay_type
        self.decay_start_epoch = decay_start_epoch
        self.decay_rate = decay_rate
        self.decay_patient_epoch = decay_patient_epoch
        self.not_improved_epoch = 0
        self.lower_better = lower_better
        self.best_value = best_value

        # for warmup
        if decay_type == 'warmup':
            assert warmup_nsteps > 0
            if warmup_start_learning_rate > 0:
                self.lr_init = warmup_start_learning_rate
            else:
                self.lr_init = factor * (model_size ** (-0.5))
        else:
            self.lr_init = learning_rate
        self.warmup_start_lr = warmup_start_learning_rate
        self.warmup_nsteps = warmup_nsteps

    def decay_lr(self, optimizer, lr, epoch, value):
        """Decay learning rate per epoch.

        Args:
            optimizer:
            lr (float): the current learning rete
            epoch (int): the current epoch
            value: (float) A value to evaluate
        Returns:
            optimizer:
            lr (float): the decayed learning rate

        """
        if not self.lower_better:
            value *= -1

        if epoch < self.decay_start_epoch:
            if self.decay_type == 'metric':
                if value < self.best_value:
                    # Update the best value
                    self.best_value = value
                    # NOTE: not update learning rate here
        else:
            if self.decay_type == 'metric':
                if value < self.best_value:
                    # Improved
                    self.best_value = value
                    self.not_improved_epoch = 0
                elif self.not_improved_epoch < self.decay_patient_epoch:
                    # Not improved, but learning rate will be not decayed
                    self.not_improved_epoch += 1
                else:
                    # Not improved, and learning rate will be decayed
                    self.not_improved_epoch = 0
                    lr *= self.decay_rate

                    # Update optimizer
                    for param_group in optimizer.param_groups:
                        if isinstance(optimizer, torch.optim.Adadelta):
                            param_group['eps'] = lr
                        else:
                            param_group['lr'] = lr

            elif self.decay_type == 'epoch':
                lr *= self.decay_rate

                # Update optimizer
                for param_group in optimizer.param_groups:
                    if isinstance(optimizer, torch.optim.Adadelta):
                        param_group['eps'] = lr
                    else:
                        param_group['lr'] = lr

        return optimizer, lr

    def warmup_lr(self, optimizer, lr, step):
        """Warm up learning rate per step.

        Args:
            optimizer:
            lr (float): the current learning rete
            epoch (int): the current epoch
        Returns:
            optimizer:
            lr (float): the decayed learning rate

        """
        if self.warmup_start_lr > 0:
            # linearly increse
            lr = (self.lr_max - self.warmup_start_lr) / self.warmup_nsteps * step + self.lr_init
        else:
            # based on the original transformer paper
            lr = self.lr_init * min(step ** (-0.5),
                                    step * (self.warmup_nsteps ** (-1.5)))

        # Update optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer, lr


def load_config(config_path):
    """Load a configration yaml file.

    Args:
        config_path (str):
    Returns:
        params (dict):

    """
    with open(config_path, "r") as f:
        config = yaml.load(f)

        # Load the parent config file
        if 'parent' in config.keys():
            with open(config['parent'], "r") as fp:
                config_parent = yaml.load(fp)
            params = config_parent['param']

            # Override
            for key in config['param'].keys():
                params[key] = config['param'][key]
        else:
            params = config['param']
    return params


def save_config(config, save_path):
    """Save a configuration file as a yaml file.

    Args:
        config (dict):

    """
    with open(os.path.join(save_path, 'config.yml'), "w") as f:
        f.write(yaml.dump({'param': config}, default_flow_style=False))


def set_logger(save_path, key):
    """Set logger.

    Args:
        save_path (str):
        key (str):
    Returns:
        logger ():

    """
    logger = logging.getLogger(key)
    sh = logging.StreamHandler()
    fh = logging.FileHandler(save_path)

    logger.setLevel(logging.DEBUG)
    sh.setLevel(logging.WARNING)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s line:%(lineno)d %(levelname)s: %(message)s')
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger
