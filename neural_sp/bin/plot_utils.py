#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Plot attention weights & ctc probabilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('ggplot')
sns.set_style("white")
blue = '#4682B4'
orange = '#D2691E'
green = '#006400'

# sns.set(font='IPAMincho')
sns.set(font='Noto Sans CJK JP')


def plot_attention_weights(aw, tokens=[], spectrogram=None, ref=None,
                           save_path=None, figsize=(20, 8),
                           ctc_probs=None, ctc_topk_ids=None):
    """Plot attention weights.

    Args:
        aw (np.ndarray): A tensor of size `[L, T, n_heads]
        tokens (list): hypothesis tokens
        spectrogram (np.ndarray): A tensor of size `[T, feature_dim]`
        ref (str): reference text
        save_path (str): path to save a figure
        figsize (tuple):

    """
    plt.clf()
    plt.figure(figsize=figsize)

    n_heads = aw.shape[2]
    n_col = n_heads
    if spectrogram is not None:
        n_col += 1
    if ctc_probs is not None:
        n_col += 1

    if spectrogram is None:
        sns.heatmap(aw[:, :, 0], cmap='viridis',
                    xticklabels=False,
                    yticklabels=tokens if len(tokens) > 0 else False,
                    annot_kws={'size': 5})
        plt.ylabel(u'Output labels (←)', fontsize=12)
        plt.yticks(rotation=0)
    else:
        for h in range(1, n_heads + 1, 1):
            plt.subplot(n_col, 1, h)
            sns.heatmap(aw[:, :, h - 1], cmap='viridis',
                        xticklabels=False,
                        yticklabels=tokens if len(tokens) > 0 else False,
                        cbar_kws={"orientation": "horizontal"})
            plt.ylabel(u'Output labels (←)', fontsize=12)
            plt.yticks(rotation=0)

        # CTC propabilities for joint CTC-attention
        if ctc_probs is not None:
            plt.subplot(n_col, 1, n_col - 1)
            times_probs = np.arange(ctc_probs.shape[0])
            for idx in set(ctc_topk_ids.reshape(-1).tolist()):
                if idx == 0:
                    plt.plot(times_probs, ctc_probs[:, 0], ':', label='<blank>', color='grey')
                else:
                    plt.plot(times_probs, ctc_probs[:, idx])
            plt.ylabel('CTC posteriors', fontsize=12)
            plt.tick_params(labelbottom=False)
            plt.yticks(list(range(0, 2, 1)))
            plt.xlim(0, ctc_probs.shape[0])

        # Plot spectrogram
        plt.subplot(n_col, 1, n_col)
        plt.imshow(spectrogram.T, cmap='viridis', aspect='auto', origin='lower')
        plt.xlabel(u'Time [msec]', fontsize=12)
        plt.ylabel(u'Frequency bin', fontsize=12)
        # plt.colorbar()
        plt.grid('off')

    if ref is not None:
        plt.title(ref)

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path, dvi=500)

    plt.close()


def plot_hierarchical_attention_weights(aw, aw_sub, tokens=[], tokens_sub=[],
                                        spectrogram=None, ref=None,
                                        save_path=None, figsize=(20, 8)):
    """Plot attention weights for the hierarchical model.

    Args:
        spectrogram (np.ndarray): A tensor of size `[T, input_size]`
        aw (np.ndarray): A tensor of size `[L, T]`
        aw_sub (np.ndarray): A tensor of size `[L_sub, T]`
        tokens (list):
        tokens_sub (list):
        spectrogram (np.ndarray): A tensor of size `[T, feature_dim]`
        ref (str):
        save_path (str): path to save a figure
        figsize (tuple):

    """
    plt.clf()
    plt.figure(figsize=figsize)

    if spectrogram is None:
        plt.subplot(211)
        sns.heatmap(aw, cmap='viridis',
                    xticklabels=False,
                    yticklabels=tokens if len(tokens) > 0 else False)
        plt.ylabel(u'Output labels (main) (←)', fontsize=12)
        plt.yticks(rotation=0)

        plt.subplot(212)
        sns.heatmap(aw_sub, cmap='viridis',
                    xticklabels=False,
                    yticklabels=tokens_sub if len(tokens_sub) > 0 else False)
        plt.xlabel(u'Time [sec]', fontsize=12)
        plt.ylabel(u'Output labels (sub) (←)', fontsize=12)
        plt.yticks(rotation=0)
    else:
        plt.subplot(311)
        sns.heatmap(aw, cmap='viridis',
                    xticklabels=False,
                    yticklabels=tokens if len(tokens) > 0 else False)
        plt.ylabel(u'Output labels (main) (←)', fontsize=12)
        plt.yticks(rotation=0)

        plt.subplot(312)
        sns.heatmap(aw_sub, cmap='viridis',
                    xticklabels=False,
                    yticklabels=tokens_sub if len(tokens_sub) > 0 else False)
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


def plot_ctc_probs(ctc_probs, topk_ids, subsample_factor, space=-1, hyp='',
                   spectrogram=None, save_path=None, figsize=(20, 8), topk=None):
    """Plot CTC posteriors.

    Args:
        ctc_probs (np.ndarray): A tensor of size `[T, vocab]`
        topk_ids ():
        subsample_factor (int): the number of frames to stack
        space (int): index for space mark
        hyp (str):
        save_path (str): path to save a figure
        figsize (tuple):
        topk (int):

    """
    plt.clf()
    plt.figure(figsize=figsize)
    n_frames = ctc_probs.shape[0]
    times_probs = np.arange(n_frames) * subsample_factor / 100
    if len(hyp) > 0:
        plt.title(hyp)

    plt.xlim([0, n_frames * subsample_factor / 100])
    plt.ylim([0.05, 1.05])
    plt.legend(loc="upper right", fontsize=12)

    if spectrogram is None:
        # NOTE: index 0 is reserved for blank
        for idx in set(topk_ids.reshape(-1).tolist()):
            if idx == 0:
                plt.plot(times_probs, ctc_probs[:, 0], ':', label='<blank>', color='grey')
            elif idx == space:
                plt.plot(times_probs, ctc_probs[:, space], label='<space>', color='black')
            else:
                plt.plot(times_probs, ctc_probs[:, idx])
        plt.xlabel(u'Time [sec]', fontsize=12)
        plt.ylabel('Posteriors', fontsize=12)
        plt.xticks(list(range(0, int(n_frames * subsample_factor / 100) + 1, 1)))
        plt.yticks(list(range(0, 2, 1)))
    else:
        plt.subplot(211)
        # NOTE: index 0 is reserved for blank
        for idx in set(topk_ids.reshape(-1).tolist()):
            if idx == 0:
                plt.plot(times_probs, ctc_probs[:, 0], ':', label='<blank>', color='grey')
            elif idx == space:
                plt.plot(times_probs, ctc_probs[:, space], label='<space>', color='black')
            else:
                plt.plot(times_probs, ctc_probs[:, idx])
        plt.ylabel('Posteriors', fontsize=12)
        plt.tick_params(labelbottom=False)
        plt.yticks(list(range(0, 2, 1)))

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


def plot_hierarchical_ctc_probs(ctc_probs, topk_ids, ctc_probs_sub, topk_ids_sub,
                                subsample_factor, space=-1, space_sub=-1, hyp='', hyp_sub='',
                                spectrogram=None, save_path=None, figsize=(20, 8)):
    """Plot CTC posteriors for the hierarchical model.

    Args:
        ctc_probs (np.ndarray): A tensor of size `[T, vocab]`
        ctc_probs_sub (np.ndarray): A tensor of size `[T, num_classes_sub]`
        n_frames (int):
        subsample_factor (int):
        save_path (str): path to save a figure
        figsize (tuple):
        space (int): index for space mark

    """
    # TODO(hirofumi): add spectrogram

    plt.clf()
    plt.figure(figsize=figsize)
    n_frames = ctc_probs.shape[0]
    times_probs = np.arange(n_frames) * subsample_factor / 100
    if len(hyp) > 0:
        plt.title(hyp)
        # plt.title(hyp_sub)

    # NOTE: index 0 is reserved for blank
    plt.subplot(211)
    plt.plot(times_probs, ctc_probs[:, 0], ':', label='<blank>', color='grey')

    # Plot only top-k
    for idx in set(topk_ids.reshape(-1).tolist()):
        if idx == 0:
            plt.plot(times_probs, ctc_probs[:, 0], ':', label='<blank>', color='grey')
        elif idx == space:
            plt.plot(times_probs, ctc_probs[:, space], label='<space>', color='black')
        else:
            plt.plot(times_probs, ctc_probs[:, idx])

    plt.ylabel('Posteriors (Word)', fontsize=12)
    plt.xlim([0, n_frames * subsample_factor / 100])
    plt.ylim([0.05, 1.05])
    plt.xticks(list(range(0, int(n_frames * subsample_factor / 100) + 1, 1)))
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)

    plt.subplot(212)
    plt.plot(times_probs, ctc_probs_sub[:, 0], ':', label='<blank>', color='grey')
    for idx in set(topk_ids_sub.reshape(-1).tolist()):
        if idx == 0:
            plt.plot(times_probs, ctc_probs_sub[:, 0], ':', label='<blank>', color='grey')
        elif idx == space_sub:
            plt.plot(times_probs, ctc_probs_sub[:, space_sub], label='<space>', color='black')
        else:
            plt.plot(times_probs, ctc_probs_sub[:, idx])
    plt.xlabel(u'Time [sec]', fontsize=12)
    plt.ylabel('Posteriors (Char)', fontsize=12)
    plt.xlim([0, n_frames * subsample_factor / 100])
    plt.ylim([0.05, 1.05])
    plt.xticks(list(range(0, int(n_frames * subsample_factor / 100) + 1, 1)))
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path, dvi=500)

    plt.close()
