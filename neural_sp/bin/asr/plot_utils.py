#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Plot attention weights & ctc probabilities."""

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

# sns.set(font='IPAMincho')
sns.set(font='Noto Sans CJK JP')


def plot_attention_weights(aw, label_list=[], spectrogram=None, text_ref=None,
                           save_path=None, figsize=(10, 4)):
    """Plot attention weights.

    Args:
        aw (np.ndarray): A tensor of size `[L, T]`
        label_list (list):
        spectrogram (np.ndarray): A tensor of size `[T, feature_dim]`
        text_ref (str):
        save_path (str): path to save a figure of CTC posterior (utterance)
        figsize (tuple):

    """
    plt.clf()
    plt.figure(figsize=figsize)

    if spectrogram is None:
        # Plot attention weights
        sns.heatmap(aw, cmap='viridis',
                    xticklabels=False,
                    yticklabels=label_list if len(label_list) > 0 else False)
        # cbar_kws={"orientation": "horizontal"}
        plt.ylabel(u'Output labels (←)', fontsize=12)
        plt.yticks(rotation=0)
    else:
        # Plot attention weights
        plt.subplot(211)
        sns.heatmap(aw, cmap='viridis',
                    xticklabels=False,
                    yticklabels=label_list if len(label_list) > 0 else False)
        plt.ylabel(u'Output labels (←)', fontsize=12)
        plt.yticks(rotation=0)

        # Plot spectrogram
        plt.subplot(212)
        plt.imshow(spectrogram.T, cmap='viridis', aspect='auto', origin='lower')
        plt.xlabel(u'Time [msec]', fontsize=12)
        plt.ylabel(u'Frequency bin', fontsize=12)
        plt.colorbar()
        plt.grid('off')

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path, dvi=500)

    plt.close()


def plot_hierarchical_attention_weights(aw, aw_sub, label_list=[], label_list_sub=[],
                                        spectrogram=None, text_ref=None,
                                        save_path=None, figsize=(20, 8)):
    """Plot attention weights for the hierarchical model.

    Args:
        spectrogram (np.ndarray): A tensor of size `[T, input_size]`
        aw (np.ndarray): A tensor of size `[L, T]`
        aw_sub (np.ndarray): A tensor of size `[L_sub, T]`
        label_list (list):
        label_list_sub (list):
        spectrogram (np.ndarray): A tensor of size `[T, feature_dim]`
        text_ref (str):
        save_path (str): path to save a figure of CTC posterior (utterance)
        figsize (tuple):

    """
    plt.clf()
    plt.figure(figsize=figsize)

    if spectrogram is None:
        # Plot attention weights
        plt.subplot(211)
        sns.heatmap(aw, cmap='viridis',
                    xticklabels=False,
                    yticklabels=label_list if len(label_list) > 0 else False)
        plt.ylabel(u'Output labels (main) (←)', fontsize=12)
        plt.yticks(rotation=0)

        plt.subplot(212)
        sns.heatmap(aw_sub, cmap='viridis',
                    xticklabels=False,
                    yticklabels=label_list_sub if len(label_list_sub) > 0 else False)
        plt.xlabel(u'Time [sec]', fontsize=12)
        plt.ylabel(u'Output labels (sub) (←)', fontsize=12)
        plt.yticks(rotation=0)
    else:
        # Plot attention weights
        plt.subplot(311)
        sns.heatmap(aw, cmap='viridis',
                    xticklabels=False,
                    yticklabels=label_list if len(label_list) > 0 else False)
        plt.ylabel(u'Output labels (main) (←)', fontsize=12)
        plt.yticks(rotation=0)

        plt.subplot(312)
        sns.heatmap(aw_sub, cmap='viridis',
                    xticklabels=False,
                    yticklabels=label_list_sub if len(label_list_sub) > 0 else False)
        plt.ylabel(u'Output labels (sub) (←)', fontsize=12)
        plt.yticks(rotation=0)

        # Plot spectrogram
        plt.subplot(313)
        plt.imshow(spectrogram.T, cmap='viridis', aspect='auto', origin='lower')
        plt.xlabel(u'Time [msec]', fontsize=12)
        plt.ylabel(u'Frequency bin', fontsize=12)
        plt.colorbar()
        plt.grid('off')

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path, dvi=500)

    plt.close()


def plot_nested_attention_weights(aw, label_list=[], label_list_sub=[],
                                  save_path=None, figsize=(10, 4)):
    """Plot attention weights from word-level decoder to character-level decoder.

    Args:
        aw (np.ndarray): A tensor of size `[L, T]`
        label_list (list):
        label_list_sub (list):
        save_path (str): path to save a figure of CTC posterior (utterance)
        figsize (tuple):

    """
    plt.clf()
    plt.figure(figsize=figsize)

    # Plot attention weights
    sns.heatmap(awmap='viridis',
                xticklabels=label_list_sub,
                yticklabels=label_list)
    # cbar_kws={"orientation": "horizontal"}
    plt.ylabel(u'Output characters (→)', fontsize=12)
    plt.ylabel(u'Output words (←)', fontsize=12)
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path, dvi=500)

    plt.close()


def plot_ctc_probs(probs, frame_num, num_stack, space_index=None, text_hyp='',
                   spectrogram=None, save_path=None, figsize=(10, 4)):
    """Plot CTC posteriors.

    Args:
        probs (np.ndarray): A tensor of size `[T, num_classes]`
        frame_num (int):
        num_stack (int): the number of frames to stack
        space_index (int):
        text_hyp (str):
        save_path (str): path to save a figure of CTC posterior (utterance)
        figsize (tuple):

    """
    plt.clf()
    plt.figure(figsize=figsize)
    times_probs = np.arange(frame_num) * num_stack / 100
    if len(text_hyp) > 0:
        plt.title(text_hyp)

    if spectrogram is None:
        # NOTE: index 0 is reserved for blank
        plt.plot(times_probs, probs[:, 0], ':', label='blank', color='grey')
        for i in range(1, probs.shape[-1], 1):
            if space_index is not None and i == space_index:
                plt.plot(times_probs, probs[:, space_index], label='space', color='black')
            else:
                plt.plot(times_probs, probs[:, i])
        plt.xlabel(u'Time [sec]', fontsize=12)
        plt.ylabel('Posteriors', fontsize=12)
        plt.xlim([0, frame_num * num_stack / 100])
        plt.ylim([0.05, 1.05])
        plt.xticks(list(range(0, int(frame_num * num_stack / 100) + 1, 1)))
        plt.yticks(list(range(0, 2, 1)))
        plt.legend(loc="upper right", fontsize=12)
    else:
        # Plot CTC posteriors
        plt.subplot(211)
        plt.plot(times_probs, probs[:, 0], ':', label='blank', color='grey')
        # NOTE: index 0 is reserved for blank
        for i in range(1, probs.shape[-1], 1):
            if space_index is not None and i == space_index:
                plt.plot(times_probs, probs[:, space_index], label='space', color='black')
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
        plt.imshow(spectrogram.T, cmap='viridis', aspect='auto', origin='lower')
        plt.xlabel(u'Time [msec]', fontsize=12)
        plt.ylabel(u'Frequency bin', fontsize=12)
        # plt.colorbar()
        plt.grid('off')

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path, dvi=500)

    plt.close()


def plot_hierarchical_ctc_probs(probs, probs_sub, frame_num, num_stack,
                                space_index=None, text_hyp='', text_hyp_sub='',
                                spectrogram=None, save_path=None, figsize=(20, 8)):
    """Plot CTC posteriors for the hierarchical model.

    Args:
        probs (np.ndarray): A tensor of size `[T, num_classes]`
        probs_sub (np.ndarray): A tensor of size `[T, num_classes_sub]`
        frame_num (int):
        num_stack (int):
        save_path (str): path to save a figure of CTC posterior (utterance)
        figsize (tuple):
        space_index (int):

    """
    # TODO: add spectrogram

    plt.clf()
    plt.figure(figsize=figsize)
    times_probs = np.arange(frame_num) * num_stack / 100
    if len(text_hyp) > 0:
        plt.title(text_hyp)
        # plt.title(text_hyp_sub)

    # NOTE: index 0 is reserved for blank
    plt.subplot(211)
    plt.plot(times_probs, probs[:, 0], ':', label='blank', color='grey')

    # Plot only top-k
    # indices = np.argmax(probs[:, 1:], axis=-1)
    # plt.plot(times_probs, probs[:, indices])
    # for i in range(1, probs.shape[-1], 1):
    for i in range(1, 100, 1):
        plt.plot(times_probs, probs[:, i])

    plt.ylabel('Posteriors (Word)', fontsize=12)
    plt.xlim([0, frame_num * num_stack / 100])
    plt.ylim([0.05, 1.05])
    plt.xticks(list(range(0, int(frame_num * num_stack / 100) + 1, 1)))
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)

    plt.subplot(212)
    plt.plot(times_probs, probs_sub[:, 0], ':', label='blank', color='grey')
    for i in range(1, probs_sub.shape[-1], 1):
        if space_index is not None and i == space_index:
            plt.plot(times_probs, probs_sub[:, space_index], label='space', color='black')
        else:
            plt.plot(times_probs, probs_sub[:, i])
    plt.xlabel(u'Time [sec]', fontsize=12)
    plt.ylabel('Posteriors (Char)', fontsize=12)
    plt.xlim([0, frame_num * num_stack / 100])
    plt.ylim([0.05, 1.05])
    plt.xticks(list(range(0, int(frame_num * num_stack / 100) + 1, 1)))
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path, dvi=500)

    plt.close()
