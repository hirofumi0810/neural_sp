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

    def __init__(self, save_path):
        self.save_path = save_path
        self.steps = []
        self.losses_train = []
        self.losses_dev = []
        self.accs_train = []
        self.accs_dev = []

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
        plt.legend(loc="upper right", fontsize=12)
        if os.path.isfile(os.path.join(self.save_path, "loss.png")):
            os.remove(os.path.join(self.save_path, "loss.png"))
        plt.savefig(os.path.join(self.save_path, "loss.png"), dvi=500)

        # Save loss as csv file
        if os.path.isfile(os.path.join(self.save_path, "loss.csv")):
            os.remove(os.path.join(self.save_path, "loss.csv"))
        loss_graph = np.column_stack(
            (self.steps, self.losses_train, self.losses_dev))
        np.savetxt(os.path.join(self.save_path, "loss.csv"),
                   loss_graph, delimiter=",")

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
        acc_graph = np.column_stack(
            (self.steps, self.accs_train, self.accs_dev))
        if os.path.isfile(os.path.join(self.save_path, "accuracy.csv")):
            os.remove(os.path.join(self.save_path, "accuracy.csv"))
        np.savetxt(os.path.join(self.save_path, "accuracy.csv"),
                   acc_graph, delimiter=",")
