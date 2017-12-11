#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot the CTC posteriors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_style("white")
blue = '#4682B4'
orange = '#D2691E'
green = '#006400'


def plot_ctc_probs(probs, frame_num, num_stack, save_path=None):
    """
    Args:
        probs (np.ndarray): A tensor of size `[T, num_classes]`
        frame_num (int):
        num_stack (int):
        save_path (string): path to save a figure of CTC posterior (utterance)
    """
    plt.clf()
    plt.figure(figsize=(10, 4))
    times_probs = np.arange(frame_num) * num_stack / 100

    # NOTE: index 0 is reserved for blank in warpctc_pytorch
    plt.plot(times_probs, probs[:, 0],
             ':', label='blank', color='grey')
    for i in range(1, probs.shape[-1], 1):
        plt.plot(times_probs, probs[:, i])
    plt.xlabel('Time [sec]', fontsize=12)
    plt.ylabel('Posteriors', fontsize=12)
    plt.xlim([0, frame_num * num_stack / 100])
    plt.ylim([0.05, 1.05])
    plt.xticks(
        list(range(0, int(frame_num * num_stack / 100) + 1, 1)))
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path, dvi=500)

    plt.close()


def plot_hieracical_ctc_probs(probs, probs_sub, frame_num, num_stack,
                              save_path=None):
    raise NotImplementedError
