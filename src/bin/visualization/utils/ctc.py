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


def plot_ctc_probs(probs, frame_num, num_stack, space_index=None, str_pred='',
                   spectrogram=None, save_path=None, figsize=(10, 4)):
    """Plot CTC posteriors.
    Args:
        probs (np.ndarray): A tensor of size `[T, num_classes]`
        frame_num (int):
        num_stack (int): the number of frames to stack
        space_index (int, optional):
        str_pred (string, optional):
        save_path (string, optional): path to save a figure of CTC posterior (utterance)
        figsize (tuple, optional):
    """
    plt.clf()
    plt.figure(figsize=figsize)
    times_probs = np.arange(frame_num) * num_stack / 100
    if len(str_pred) > 0:
        plt.title(str_pred)

    if spectrogram is None:
        # NOTE: index 0 is reserved for blank
        plt.plot(times_probs, probs[:, 0],
                 ':', label='blank', color='grey')
        for i in range(1, probs.shape[-1], 1):
            if space_index is not None and i == space_index:
                plt.plot(times_probs, probs[:, space_index],
                         label='space', color='black')
            else:
                plt.plot(times_probs, probs[:, i])
        plt.xlabel('Time [sec]', fontsize=12)
        plt.ylabel('Posteriors', fontsize=12)
        plt.xlim([0, frame_num * num_stack / 100])
        plt.ylim([0.05, 1.05])
        plt.xticks(list(range(0, int(frame_num * num_stack / 100) + 1, 1)))
        plt.yticks(list(range(0, 2, 1)))
        plt.legend(loc="upper right", fontsize=12)
    else:
        # Plot CTC posteriors
        plt.subplot(211)
        plt.plot(times_probs, probs[:, 0],
                 ':', label='blank', color='grey')
        # NOTE: index 0 is reserved for blank
        for i in range(1, probs.shape[-1], 1):
            if space_index is not None and i == space_index:
                plt.plot(times_probs, probs[:, space_index],
                         label='space', color='black')
            else:
                plt.plot(times_probs, probs[:, i])
        plt.ylabel('Posteriors', fontsize=12)
        plt.xlim([0, frame_num * num_stack / 100])
        plt.ylim([0.05, 1.05])
        plt.tick_params(labelbottom=False)
        plt.yticks(list(range(0, 2, 1)))
        plt.legend(loc="upper right", fontsize=12)

        # Plot spectrogram
        plt.subplot(212)
        plt.imshow(spectrogram.T,
                   cmap='viridis',
                   aspect='auto', origin='lower')
        plt.xlabel('Time [msec]', fontsize=12)
        plt.ylabel('Frequency bin', fontsize=12)
        # plt.colorbar()
        plt.grid('off')

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path, dvi=500)

    plt.close()


def plot_hierarchical_ctc_probs(probs, probs_sub, frame_num, num_stack,
                                space_index=None, str_pred='', str_pred_sub='',
                                spectrogram=None, save_path=None, figsize=(20, 8)):
    """Plot CTC posteriors for the hierarchical model.
    Args:
        probs (np.ndarray): A tensor of size `[T, num_classes]`
        probs_sub (np.ndarray): A tensor of size `[T, num_classes_sub]`
        frame_num (int):
        num_stack (int):
        save_path (string): path to save a figure of CTC posterior (utterance)
        figsize (tuple):
        space_index (int, optional):
    """
    # TODO: add spectrogram

    plt.clf()
    plt.figure(figsize=figsize)
    times_probs = np.arange(frame_num) * num_stack / 100
    if len(str_pred) > 0:
        plt.title(str_pred)
        # plt.title(str_pred_sub)

    # NOTE: index 0 is reserved for blank
    plt.subplot(211)
    plt.plot(times_probs, probs[:, 0],
             ':', label='blank', color='grey')

    # Plot only top-k
    # indices = np.argmax(probs[:, 1:], axis=-1)
    # plt.plot(times_probs, probs[:, indices])
    # for i in range(1, probs.shape[-1], 1):
    for i in range(1, 100, 1):
        plt.plot(times_probs, probs[:, i])

    plt.ylabel('Posteriors (Word)', fontsize=12)
    plt.xlim([0, frame_num * num_stack / 100])
    plt.ylim([0.05, 1.05])
    plt.xticks(
        list(range(0, int(frame_num * num_stack / 100) + 1, 1)))
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)

    plt.subplot(212)
    plt.plot(times_probs, probs_sub[:, 0],
             ':', label='blank', color='grey')
    for i in range(1, probs_sub.shape[-1], 1):
        if space_index is not None and i == space_index:
            plt.plot(times_probs, probs_sub[:, space_index],
                     label='space', color='black')
        else:
            plt.plot(times_probs, probs_sub[:, i])
    plt.xlabel('Time [sec]', fontsize=12)
    plt.ylabel('Posteriors (Char)', fontsize=12)
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
