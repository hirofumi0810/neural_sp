#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot attention weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_style("white")
blue = '#4682B4'
orange = '#D2691E'
green = '#006400'

# sns.set(font='IPAMincho')
sns.set(font='Noto Sans CJK JP')


def plot_attention_weights(spectrogram, attention_weights, label_list,
                           save_path=None, fig_size=(10, 4)):
    """
    Args:
        spectrogram (np.ndarray): A tensor of size `[T_in, input_size]`
        attention_weights (np.ndarray): A tensor of size `[T_out, T_in]`
        label_list (list):
        save_path (string): path to save a figure of CTC posterior (utterance)
        fig_size (tuple):
    """
    plt.clf()
    plt.figure(figsize=fig_size)

    # plot attention weights
    # plt.subplot(211)
    sns.heatmap(attention_weights,
                # cmap='Blues',
                cmap='viridis',
                xticklabels=False,
                yticklabels=label_list)
    plt.ylabel('Output labels (←)', fontsize=12)

    # plot spectrogram
    # plt.subplot(212)
    # plt.plot(spectrogram)
    # plt.xlabel('Input frames [sec]', fontsize=12)
    # plt.ylabel('Frequency', fontsize=12)

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path, dvi=500)

    plt.close()


def plot_hierarchical_attention_weights(spectrogram,
                                        attention_weights, attention_weights_sub,
                                        label_list, label_list_sub,
                                        save_path=None, fig_size=(20, 8)):
    """
    Args:
        spectrogram (np.ndarray): A tensor of size `[T_in, input_size]`
        attention_weights (np.ndarray): A tensor of size `[T_out, T_in]`
        attention_weights_sub (np.ndarray): A tensor of size `[T_out_sub, T_in]`
        label_list (list):
        label_list_sub (list):
        save_path (string): path to save a figure of CTC posterior (utterance)
        fig_size (tuple):
    """
    plt.clf()
    plt.figure(figsize=fig_size)

    # plot attention weights
    plt.subplot(211)
    sns.heatmap(attention_weights,
                cmap='viridis',
                xticklabels=False,
                yticklabels=label_list)
    plt.ylabel('Output labels (←)', fontsize=12)

    plt.subplot(212)
    sns.heatmap(attention_weights_sub,
                cmap='viridis',
                xticklabels=False,
                yticklabels=label_list_sub)
    plt.xlabel('Input frames [sec]', fontsize=12)
    plt.ylabel('Output labels (←)', fontsize=12)

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path, dvi=500)

    plt.close()
