#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Report loss, accuracy etc. during training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
blue = '#4682B4'
orange = '#D2691E'


class Reporter(object):
    """"Report loss, accuracy etc. during training.
    Args:

    """

    def __init__(self, save_path, max_loss=500):
        self.save_path = save_path
        self.max_loss = max_loss

        self.steps = []
        self.losses_train = []
        self.losses_main_train = []
        self.losses_sub_train = []
        self.losses_dev = []
        self.losses_main_dev = []
        self.losses_sub_dev = []
        self.accs_main_train = []
        self.accs_sub_train = []
        self.accs_main_dev = []
        self.accs_sub_dev = []

    def step(self, step, loss_train, loss_main_train, loss_sub_train,
             loss_dev, loss_main_dev, loss_sub_dev,
             acc_main_train, acc_sub_train,
             acc_main_dev, acc_sub_dev):
        self.steps.append(step)

        self.losses_train.append(loss_train)
        self.losses_main_train.append(loss_main_train)
        self.losses_sub_train.append(loss_sub_train)

        self.losses_dev.append(loss_dev)
        self.losses_main_dev.append(loss_main_dev)
        self.losses_sub_dev.append(loss_sub_dev)

        self.accs_main_train.append(acc_main_train)
        self.accs_sub_train.append(acc_sub_train)

        self.accs_main_dev.append(acc_main_dev)
        self.accs_sub_dev.append(acc_sub_dev)

    def epoch(self):
        # Plot loss
        plt.clf()
        plt.plot(self.steps, self.losses_train, blue, label="Train")
        plt.plot(self.steps, self.losses_main_train,
                 blue, label="Train (main)", ls="--")
        plt.plot(self.steps, self.losses_sub_train,
                 blue, label="Train (sub)", ls=":")
        plt.plot(self.steps, self.losses_dev, orange, label="Dev")
        plt.plot(self.steps, self.losses_main_dev,
                 orange, label="Dev (main)", ls="--")
        plt.plot(self.steps, self.losses_sub_dev,
                 orange, label="Dev (sub)", ls=":")
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
        loss_graph = np.column_stack(
            (self.steps, self.losses_train, self.losses_main_train, self.losses_sub_train,
             self.losses_dev, self.losses_main_dev, self.losses_sub_dev))
        np.savetxt(os.path.join(self.save_path, "loss.csv"),
                   loss_graph, delimiter=",")

        # Plot accuracy
        plt.clf()
        plt.plot(self.steps, self.accs_main_train,
                 blue, label="Train (main)")
        plt.plot(self.steps, self.accs_sub_train,
                 blue, label="Train (sub)", ls=":")
        plt.plot(self.steps, self.accs_main_dev,
                 orange, label="Dev (main)")
        plt.plot(self.steps, self.accs_sub_dev,
                 orange, label="Dev (sub)", ls=":")
        plt.xlabel('step', fontsize=12)
        plt.ylabel('accuracy', fontsize=12)
        plt.legend(loc="upper right", fontsize=12)
        if os.path.isfile(os.path.join(self.save_path, 'accuracy.png')):
            os.remove(os.path.join(self.save_path, 'accuracy.png'))
        plt.savefig(os.path.join(self.save_path, 'accuracy.png'), dvi=500)

        # Save accuracy as csv file
        acc_graph = np.column_stack(
            (self.steps, self.accs_main_train, self.accs_sub_train,
             self.accs_main_dev, self.accs_sub_dev))
        if os.path.isfile(os.path.join(self.save_path, "accuracy.csv")):
            os.remove(os.path.join(self.save_path, "accuracy.csv"))
        np.savetxt(os.path.join(self.save_path, "accuracy.csv"),
                   acc_graph, delimiter=",")
