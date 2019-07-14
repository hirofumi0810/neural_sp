#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Reporter during training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from logging import getLogger
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import seaborn as sns
from tensorboardX import SummaryWriter

plt.style.use('ggplot')
grey = '#878f99'
blue = '#4682B4'
orange = '#D2691E'
green = '#82b74b'

logger = getLogger('training')


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
                logger.warning("WARNING: received an inf %s for %s." % (metric, k))

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

    def step(self, is_eval=False):
        self._step += 1
        if is_eval:
            self.steps.append(self._step)

            # reset
            self.observation_train_local = {'loss': {}, 'acc': {}, 'ppl': {}}

    def epoch(self, metric=None, name='wer'):
        self._epoch += 1
        if metric is None:
            return
        self.epochs.append(self._epoch)

        # register
        self.observation_eval.append(metric)

        plt.clf()
        plt.plot(self.epochs, self.observation_eval, orange,
                 label='dev', linestyle='-')
        plt.xlabel('epoch', fontsize=12)
        plt.ylabel(name.upper(), fontsize=12)
        plt.ylim([0, min(100, max(self.observation_eval) + 1)])
        plt.legend(loc="upper right", fontsize=12)
        if os.path.isfile(os.path.join(self.save_path, name + ".png")):
            os.remove(os.path.join(self.save_path, name + ".png"))
        plt.savefig(os.path.join(self.save_path, name + ".png"), dvi=500)

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
