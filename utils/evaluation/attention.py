#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot attention weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_style("white")
blue = '#4682B4'
orange = '#D2691E'
green = '#006400'


def plot_attention_weights(attention_weights, label_list,
                           save_path=None, fig_size=(10, 4)):
    """
    Args:
        attention_weights (np.ndarray): A tensor of size `[T_out, T_in]`
        label_list (list):
        save_path (string): path to save a figure of CTC posterior (utterance)
        fig_size (tuple):
    """
    plt.clf()
    plt.figure(figsize=fig_size)
    sns.heatmap(attention_weights,
                # cmap='Blues',
                cmap='viridis',
                xticklabels=False,
                yticklabels=label_list)

    plt.xlabel('Input frames [sec]', fontsize=12)
    plt.ylabel('Output labels (top to bottom)', fontsize=12)

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path, dvi=500)

    plt.close()
