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

# sns.set(font='IPAMincho')
sns.set(font='Noto Sans CJK JP')


def plot_attention_weights(attention_weights, frame_num, num_stack, label_list,
                           spectrogram=None, save_path=None, figsize=(10, 4)):
    """Plot attention weights.
    Args:
        attention_weights (np.ndarray): A tensor of size `[T_out, T_in]`
        frame_num (int):
        num_stack (int): the number of frames to stack
        label_list (list):
        spectrogram (np.ndarray, optional): A tensor of size `[T, feature_dim]`
        save_path (string): path to save a figure of CTC posterior (utterance)
        figsize (tuple):
    """
    plt.clf()
    plt.figure(figsize=figsize)

    if spectrogram is None:
        # Plot attention weights
        sns.heatmap(attention_weights,
                    # cmap='Blues',
                    cmap='viridis',
                    xticklabels=False,
                    yticklabels=label_list)
        # cbar_kws={"orientation": "horizontal"}
        plt.ylabel('Output labels (←)', fontsize=12)
        plt.yticks(rotation=0)
    else:
        # Plot attention weights
        plt.subplot(211)
        sns.heatmap(attention_weights,
                    cmap='viridis',
                    xticklabels=False,
                    yticklabels=label_list)
        plt.ylabel('Output labels (←)', fontsize=12)
        plt.yticks(rotation=0)

        # Plot spectrogram
        plt.subplot(212)
        imgplot = plt.imshow(spectrogram.T, aspect='auto', origin='lower')
        imgplot.set_cmap('nipy_spectral')
        plt.xlabel('Time [msec]', fontsize=12)
        plt.ylabel('Frequency bin', fontsize=12)
        plt.colorbar()
        plt.grid('off')

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path, dvi=500)

    plt.close()


def plot_hierarchical_attention_weights(attention_weights, attention_weights_sub,
                                        label_list, label_list_sub,
                                        spectrogram=None, save_path=None, figsize=(20, 8)):
    """Plot attention weights for the hierarchical model.
    Args:
        spectrogram (np.ndarray): A tensor of size `[T_in, input_size]`
        attention_weights (np.ndarray): A tensor of size `[T_out, T_in]`
        attention_weights_sub (np.ndarray): A tensor of size `[T_out_sub, T_in]`
        label_list (list):
        label_list_sub (list):
        spectrogram (np.ndarray, optional): A tensor of size `[T, feature_dim]`
        save_path (string): path to save a figure of CTC posterior (utterance)
        figsize (tuple):
    """
    plt.clf()
    plt.figure(figsize=figsize)

    if spectrogram is None:
        # Plot attention weights
        plt.subplot(211)
        sns.heatmap(attention_weights,
                    cmap='viridis',
                    xticklabels=False,
                    yticklabels=label_list)
        plt.ylabel('Output labels (main) (←)', fontsize=12)
        plt.yticks(rotation=0)

        plt.subplot(212)
        sns.heatmap(attention_weights_sub,
                    cmap='viridis',
                    xticklabels=False,
                    yticklabels=label_list_sub)
        plt.xlabel('Time [sec]', fontsize=12)
        plt.ylabel('Output labels (sub) (←)', fontsize=12)
        plt.yticks(rotation=0)
    else:
        # Plot attention weights
        plt.subplot(311)
        sns.heatmap(attention_weights,
                    cmap='viridis',
                    xticklabels=False,
                    yticklabels=label_list)
        plt.ylabel('Output labels (main) (←)', fontsize=12)
        plt.yticks(rotation=0)

        plt.subplot(312)
        sns.heatmap(attention_weights_sub,
                    cmap='viridis',
                    xticklabels=False,
                    yticklabels=label_list_sub)
        plt.ylabel('Output labels (sub) (←)', fontsize=12)
        plt.yticks(rotation=0)

        # Plot spectrogram
        plt.subplot(313)
        plt.imshow(spectrogram.T, aspect='auto', origin='lower')
        plt.xlabel('Time [msec]', fontsize=12)
        plt.ylabel('Frequency bin', fontsize=12)
        plt.colorbar()
        plt.grid('off')

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path, dvi=500)

    plt.close()
