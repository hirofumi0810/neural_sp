#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Reporter during training."""

from tensorboardX import SummaryWriter
import os
import numpy as np
from matplotlib import pyplot as plt
import logging
import matplotlib
matplotlib.use('Agg')

plt.style.use('ggplot')
grey = '#878f99'
blue = '#4682B4'
orange = '#D2691E'
green = '#82b74b'

logger = logging.getLogger(__name__)


class Reporter(object):
    """"Report loss, accuracy etc. during training.

    Args:
        save_path (str):

    """

    def __init__(self, save_path):
        self.save_path = save_path

        # tensorboard
        self.tf_writer = SummaryWriter(save_path)

        # report per step
        self._step = 0
        self.obsv_train = {'loss': {}, 'acc': {}, 'ppl': {}}
        self.obsv_train_local = {'loss': {}, 'acc': {}, 'ppl': {}}
        self.obsv_dev = {'loss': {}, 'acc': {}, 'ppl': {}}
        self.steps = []

        # report per epoch
        self._epoch = 0
        self.obsv_eval = []
        self.epochs = []

    def add(self, observation, is_eval=False):
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
                if name not in self.obsv_train_local[metric].keys():
                    self.obsv_train_local[metric][name] = []
                self.obsv_train_local[metric][name].append(v)
            else:
                # avarage for training
                if name not in self.obsv_train[metric].keys():
                    self.obsv_train[metric][name] = []
                self.obsv_train[metric][name].append(
                    np.mean(self.obsv_train_local[metric][name]))
                logger.info('%s (train): %.3f' % (k, np.mean(self.obsv_train_local[metric][name])))

                if name not in self.obsv_dev[metric].keys():
                    self.obsv_dev[metric][name] = []
                self.obsv_dev[metric][name].append(v)
                logger.info('%s (dev): %.3f' % (k, v))

            if is_eval:
                self.add_tensorboard_scalar('train' + '/' + metric + '/' + name, v)
            else:
                self.add_tensorboard_scalar('dev' + '/' + metric + '/' + name, v)

    def add_tensorboard_scalar(self, key, value):
        """Add scalar value to tensorboard."""
        self.tf_writer.add_scalar(key, value, self._step)

    def add_tensorboard_histogram(self, key, value):
        """Add histogram value to tensorboard."""
        self.tf_writer.add_histogram(key, value, self._step)

    def step(self, is_eval=False):
        self._step += 1
        if is_eval:
            self.steps.append(self._step)

            # reset
            self.obsv_train_local = {'loss': {}, 'acc': {}, 'ppl': {}}

    def epoch(self, metric=None, name='wer'):
        self._epoch += 1
        if metric is None:
            return
        self.epochs.append(self._epoch)

        # register
        self.obsv_eval.append(metric)

        plt.clf()
        upper = 0.1
        plt.plot(self.epochs, self.obsv_eval, orange,
                 label='dev', linestyle='-')
        plt.xlabel('epoch', fontsize=12)
        plt.ylabel(name, fontsize=12)
        if max(self.obsv_eval) > 1:
            upper = min(100, max(self.obsv_eval) + 1)
        else:
            upper = min(upper, max(self.obsv_eval))
        plt.ylim([0, upper])
        plt.legend(loc="upper right", fontsize=12)
        if os.path.isfile(os.path.join(self.save_path, name + ".png")):
            os.remove(os.path.join(self.save_path, name + ".png"))
        plt.savefig(os.path.join(self.save_path, name + ".png"), dvi=500)

    def snapshot(self):
        # linestyles = ['solid', 'dashed', 'dotted', 'dashdotdotted']
        linestyles = ['-', '--', '-.', ':', ':', ':', ':', ':', ':', ':', ':', ':']
        for metric in self.obsv_train.keys():
            plt.clf()
            upper = 0.1
            for i, (k, v) in enumerate(sorted(self.obsv_train[metric].items())):
                # skip non-observed values
                if np.mean(self.obsv_train[metric][k]) == 0:
                    continue

                plt.plot(self.steps, self.obsv_train[metric][k], blue,
                         label=k + " (train)", linestyle=linestyles[i])
                plt.plot(self.steps, self.obsv_dev[metric][k], orange,
                         label=k + " (dev)", linestyle=linestyles[i])
                upper = max(upper, max(self.obsv_train[metric][k]))
                upper = max(upper, max(self.obsv_dev[metric][k]))

                # Save as csv file
                if os.path.isfile(os.path.join(self.save_path, metric + '-' + k + ".csv")):
                    os.remove(os.path.join(self.save_path, metric + '-' + k + ".csv"))
                loss_graph = np.column_stack(
                    (self.steps, self.obsv_train[metric][k], self.obsv_dev[metric][k]))
                np.savetxt(os.path.join(self.save_path, metric + '-' + k + ".csv"), loss_graph, delimiter=",")

            if upper > 1:
                upper = min(upper + 10, 300)  # for CE, CTC loss

            plt.xlabel('step', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.ylim([0, upper])
            plt.legend(loc="upper right", fontsize=12)
            if os.path.isfile(os.path.join(self.save_path, metric + ".png")):
                os.remove(os.path.join(self.save_path, metric + ".png"))
            plt.savefig(os.path.join(self.save_path, metric + ".png"), dvi=500)
