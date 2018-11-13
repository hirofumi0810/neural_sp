#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utility functions for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import seaborn as sns

plt.style.use('ggplot')
blue = '#4682B4'
orange = '#D2691E'

INF = float("inf")

logger = logging.getLogger('training')


class Updater(object):
    """.

    Args:
        clip_grad_norm (float):

    """

    def __init__(self, clip_grad_norm):
        self.clip_grad_norm = clip_grad_norm

    def __call__(self, model, batch, is_eval=False):
        """.

        Args:
            model (torch.nn.Module):
            batch (tuple):
            is_eval (bool):
        Returns:
            model (torch.nn.Module):
            loss_val (float):
            loss_att_val (float):
            loss_ctc_val (float):
            acc (float): Token-level accuracy in teacher-forcing

        """
        # Step for parameter update
        if is_eval:
            loss, loss_acc_fwd, loss_acc_bwd, loss_acc_sub = model(
                batch['xs'], batch['ys'], batch['ys_sub'], is_eval=True)
        else:
            model.module.optimizer.zero_grad()
            loss, loss_acc_fwd, loss_acc_bwd, loss_acc_sub = model(
                batch['xs'], batch['ys'], batch['ys_sub'])
            if len(model.device_ids) > 1:
                loss.backward(torch.ones(len(model.device_ids)))
            else:
                loss.backward()
            loss.detach()  # Trancate the graph
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), self.clip_grad_norm)
            model.module.optimizer.step()
            # TODO(hirofumi): Add scheduler

        if model.module.bwd_weight < 0.5:
            loss_att_val = loss_acc_fwd['loss_att']
            loss_ctc_val = loss_acc_fwd['loss_ctc']
            acc = loss_acc_fwd['acc']
        else:
            loss_att_val = loss_acc_bwd['loss_att']
            loss_ctc_val = loss_acc_bwd['loss_ctc']
            acc = loss_acc_bwd['acc']

        del loss
        del loss_acc_fwd
        del loss_acc_bwd
        del loss_acc_sub

        # logger.warning('!!!Skip mini-batch!!! (max_x_len: %d, bs: %d)' %
        #                (max(len(x) for x in batch['xs']), len(batch['xs'])))
        # torch.cuda.empty_cache()
        # loss_att_val = 0.
        # acc = 0.

        if loss_att_val == INF or loss_att_val == -INF:
            logger.warning("WARNING: received an inf loss.")

        del batch

        return model, loss_att_val, loss_ctc_val, acc


class Reporter(object):
    """"Report loss, accuracy etc. during training.

    Args:
        save_path (str):
        max_loss (int): the maximum value of loss to plot

    """

    def __init__(self, save_path, max_loss=300):
        self.save_path = save_path
        self.max_loss = max_loss

        self.steps = []
        self.losses_train = []
        self.losses_sub_train = []
        self.losses_dev = []
        self.losses_sub_dev = []
        self.accs_train = []
        self.accs_sub_train = []
        self.accs_dev = []
        self.accs_sub_dev = []

    def step(self, step, loss_train, loss_dev, acc_train, acc_dev):
        self.steps.append(step)
        self.losses_train.append(loss_train)
        self.losses_dev.append(loss_dev)
        self.accs_train.append(acc_train)
        self.accs_dev.append(acc_dev)

    def epoch(self):
        # Plot loss
        plt.clf()
        plt.plot(self.steps, self.losses_train, blue, label="Train")
        plt.plot(self.steps, self.losses_dev, orange, label="Dev")
        plt.xlabel('step', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.ylim([0, self.max_loss])
        plt.legend(loc="upper right", fontsize=12)
        if os.path.isfile(os.path.join(self.save_path, "loss.png")):
            os.remove(os.path.join(self.save_path, "loss.png"))
        plt.savefig(os.path.join(self.save_path, "loss.png"), dvi=500)

        # Save loss as csv file
        if os.path.isfile(os.path.join(self.save_path, "loss.csv")):
            os.remove(os.path.join(self.save_path, "loss.csv"))
        loss_graph = np.column_stack((self.steps, self.losses_train, self.losses_dev))
        np.savetxt(os.path.join(self.save_path, "loss.csv"), loss_graph, delimiter=",")

        # Plot accuracy
        plt.clf()
        plt.plot(self.steps, self.accs_train, blue, label="Train")
        plt.plot(self.steps, self.accs_dev, orange, label="Dev")
        plt.xlabel('step', fontsize=12)
        plt.ylabel('accuracy', fontsize=12)
        plt.legend(loc="upper right", fontsize=12)
        if os.path.isfile(os.path.join(self.save_path, 'accuracy.png')):
            os.remove(os.path.join(self.save_path, 'accuracy.png'))
        plt.savefig(os.path.join(self.save_path, 'accuracy.png'), dvi=500)

        # Save accuracy as csv file
        acc_graph = np.column_stack((self.steps, self.accs_train, self.accs_dev))
        if os.path.isfile(os.path.join(self.save_path, "accuracy.csv")):
            os.remove(os.path.join(self.save_path, "accuracy.csv"))
        np.savetxt(os.path.join(self.save_path, "accuracy.csv"), acc_graph, delimiter=",")


class Controller(object):
    """Controll learning rate per epoch.

    Args:
        learning_rate_init (float): the initial learning rate
        decay_type (str): per_epoch or compare_metric
        decay_start_epoch (int): the epoch to start decay
        decay_rate (float): the rate to decay the current learning rate
        decay_patient_epoch (int): decay learning rate if results have not been
            improved for 'decay_patient_epoch'
        lower_better (bool): If True, the lower, the better.
            If False, the higher, the better.
        best_value (float): the worst value of evaluation metric

    """

    def __init__(self, learning_rate_init, decay_type,
                 decay_start_epoch, decay_rate,
                 decay_patient_epoch=1, lower_better=True, best_value=10000):
        self.learning_rate_init = learning_rate_init
        self.decay_type = decay_type
        self.decay_start_epoch = decay_start_epoch
        self.decay_rate = decay_rate
        self.decay_patient_epoch = decay_patient_epoch
        self.not_improved_epoch = 0
        self.lower_better = lower_better
        self.best_value = best_value

        assert decay_type in ['per_epoch', 'compare_metric']

    def decay_lr(self, optimizer, learning_rate, epoch, value):
        """Decay learning rate per epoch.

        Args:
            optimizer:
            learning_rate (float): the current learning rete
            epoch (int): the current epoch
            value: (float) A value to evaluate
        Returns:
            optimizer:
            learning_rate (float): the decayed learning rate

        """
        if not self.lower_better:
            value *= -1

        if epoch < self.decay_start_epoch:
            if self.decay_type == 'compare_metric':
                if value < self.best_value:
                    # Update the best value
                    self.best_value = value
                    # NOTE: not update learning rate here
        else:
            if self.decay_type == 'compare_metric':
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
                    learning_rate = learning_rate * self.decay_rate

                    # Update optimizer
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate

            elif self.decay_type == 'per_epoch':
                learning_rate = learning_rate * self.decay_rate

                # Update optimizer
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

        return optimizer, learning_rate
