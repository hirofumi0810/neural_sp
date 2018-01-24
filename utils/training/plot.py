#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

blue = '#4682B4'
orange = '#D2691E'


def plot_loss(train_losses, dev_losses, steps, save_path):
    """Save history of training & dev loss as figure.
    Args:
        train_losses (list): train losses
        dev_losses (list): dev losses
        steps (list): steps
    """
    # Save as csv file
    loss_graph = np.column_stack((steps, train_losses, dev_losses))
    if os.path.isfile(os.path.join(save_path, "loss.csv")):
        os.remove(os.path.join(save_path, "loss.csv"))
    np.savetxt(os.path.join(save_path, "loss.csv"), loss_graph, delimiter=",")
    # TODO: change to chainer reporter

    # TODO: error check for inf loss

    # Plot & save as png file
    plt.clf()
    plt.plot(steps, train_losses, blue, label="Train")
    plt.plot(steps, dev_losses, orange, label="Dev")
    plt.xlabel('step', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc="upper right", fontsize=12)
    if os.path.isfile(os.path.join(save_path, "loss.png")):
        os.remove(os.path.join(save_path, "loss.png"))
    plt.savefig(os.path.join(save_path, "loss.png"), dvi=500)


def plot_ler(train_lers, dev_lers, steps, label_type, save_path):
    """Save history of training & dev LERs as figure.
    Args:
        train_lers (list): train losses
        dev_lers (list): dev losses
        steps (list): steps
    """
    if 'word' in label_type:
        name = 'WER'
    elif 'char' in label_type:
        name = 'CER'
    elif 'phone' in label_type:
        name = 'PER'
    else:
        raise ValueError

    # Save as csv file
    loss_graph = np.column_stack((steps, train_lers, dev_lers))
    if os.path.isfile(os.path.join(save_path, "ler.csv")):
        os.remove(os.path.join(save_path, "ler.csv"))
    np.savetxt(os.path.join(save_path, "ler.csv"), loss_graph, delimiter=",")

    # Plot & save as png file
    plt.clf()
    plt.plot(steps, train_lers, blue, label="Train")
    plt.plot(steps, dev_lers, orange, label="Dev")
    plt.xlabel('step', fontsize=12)
    plt.ylabel(name, fontsize=12)
    plt.legend(loc="upper right", fontsize=12)
    if os.path.isfile(os.path.join(save_path, name.lower() + '.png')):
        os.remove(os.path.join(save_path, name.lower() + '.png'))
    plt.savefig(os.path.join(save_path, name.lower() + '.png'), dvi=500)
