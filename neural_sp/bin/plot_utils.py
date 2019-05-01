#! /usr/bin/env python
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


def plot_cache_weights(cache_probs, keys=[], queries=[],
                       save_path=None, figsize=(20, 8), mask=None):
    """Plot weights over cache.

    Args:
        cache_probs (np.ndarray): A tensor of size `[n_keys, n_queries]`
        keys (list):
        queries (list):
        save_path (str): path to save a figure
        figsize (tuple):

    """
    plt.clf()
    plt.figure(figsize=figsize)
    assert len(keys) == cache_probs.shape[0], "key: %d, cache: (%d, %d)" % (
        len(keys), cache_probs.shape[0], cache_probs.shape[1])
    assert len(queries) == cache_probs.shape[1], "query: %d, cache: (%d, %d)" % (
        len(queries), cache_probs.shape[0], cache_probs.shape[1])
    sns.heatmap(cache_probs.transpose(1, 0),
                # cmap='viridis',
                xticklabels=keys,
                yticklabels=queries,
                linewidths=0.01,
                mask=mask.transpose(1, 0) if mask is not None else None,
                vmin=0, vmax=1)
    # cbar_kws={"orientation": "horizontal"}
    plt.ylabel(u'Query (←)', fontsize=8)
    plt.yticks(rotation=0)

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path, dvi=500)

    plt.close()


def plot_attention_weights(aw, tokens=[], spectrogram=None, ref=None,
                           save_path=None, figsize=(20, 8)):
    """Plot attention weights.

    Args:
        aw (np.ndarray): A tensor of size `[L, T]`
        tokens (list):
        spectrogram (np.ndarray): A tensor of size `[T, feature_dim]`
        ref (str):
        save_path (str): path to save a figure
        figsize (tuple):

    """
    plt.clf()
    plt.figure(figsize=figsize)

    if spectrogram is None:
        sns.heatmap(aw, cmap='viridis',
                    xticklabels=False,
                    yticklabels=tokens if len(tokens) > 0 else False)
        # cbar_kws={"orientation": "horizontal"}
        plt.ylabel(u'Output labels (←)', fontsize=8)
        plt.yticks(rotation=0)
    else:
        plt.subplot(211)
        sns.heatmap(aw, cmap='viridis',
                    xticklabels=False,
                    yticklabels=tokens if len(tokens) > 0 else False)
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


def plot_ctc_probs(ctc_probs, indices_topk, nframes, subsample_factor, space_id=None, hyp='',
                   spectrogram=None, save_path=None, figsize=(20, 8), topk=None):
    """Plot CTC posteriors.

    Args:
        ctc_probs (np.ndarray): A tensor of size `[T, vocab]`
        nframes (int):
        subsample_factor (int): the number of frames to stack
        space_id (int):
        hyp (str):
        save_path (str): path to save a figure
        figsize (tuple):
        topk (int):

    """
    plt.clf()
    plt.figure(figsize=figsize)
    times_probs = np.arange(nframes) * subsample_factor / 100
    if len(hyp) > 0:
        plt.title(hyp)

    plt.xlim([0, nframes * subsample_factor / 100])
    plt.ylim([0.05, 1.05])
    plt.legend(loc="upper right", fontsize=12)

    if spectrogram is None:
        # NOTE: index 0 is reserved for blank
        if 0 not in indices_topk:
            plt.plot(times_probs, ctc_probs[:, 0], ':', label='<blank>', color='grey')
        for i in indices_topk:
            if i == 0:
                plt.plot(times_probs, ctc_probs[:, 0], ':', label='<blank>', color='grey')
            elif space_id is not None and i == space_id:
                plt.plot(times_probs, ctc_probs[:, space_id], label='<space>', color='black')
            else:
                plt.plot(times_probs, ctc_probs[:, i])
        plt.xlabel(u'Time [sec]', fontsize=12)
        plt.ylabel('Posteriors', fontsize=12)
        plt.xticks(list(range(0, int(nframes * subsample_factor / 100) + 1, 1)))
        plt.yticks(list(range(0, 2, 1)))
    else:
        plt.subplot(211)
        # NOTE: index 0 is reserved for blank
        if 0 not in indices_topk:
            plt.plot(times_probs, ctc_probs[:, 0], ':', label='<blank>', color='grey')
        for i in indices_topk:
            if i == 0:
                plt.plot(times_probs, ctc_probs[:, 0], ':', label='<blank>', color='grey')
            elif space_id is not None and i == space_id:
                plt.plot(times_probs, ctc_probs[:, space_id], label='<space>', color='black')
            else:
                plt.plot(times_probs, ctc_probs[:, i])
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


def plot_hierarchical_ctc_probs(ctc_probs, ctc_probs_sub, nframes, subsample_factor,
                                space_id=None, hyp='', hyp_sub='',
                                spectrogram=None, save_path=None, figsize=(20, 8)):
    """Plot CTC posteriors for the hierarchical model.

    Args:
        ctc_probs (np.ndarray): A tensor of size `[T, vocab]`
        ctc_probs_sub (np.ndarray): A tensor of size `[T, num_classes_sub]`
        nframes (int):
        subsample_factor (int):
        save_path (str): path to save a figure
        figsize (tuple):
        space_id (int):

    """
    # TODO: add spectrogram

    plt.clf()
    plt.figure(figsize=figsize)
    times_probs = np.arange(nframes) * subsample_factor / 100
    if len(hyp) > 0:
        plt.title(hyp)
        # plt.title(hyp_sub)

    # NOTE: index 0 is reserved for blank
    plt.subplot(211)
    plt.plot(times_probs, ctc_probs[:, 0], ':', label='<blank>', color='grey')

    # Plot only top-k
    # indices = np.argmax(ctc_probs[:, 1:], axis=-1)
    # plt.plot(times_probs, ctc_probs[:, indices])
    # for i in range(1, ctc_probs.shape[-1], 1):
    for i in range(1, 100, 1):
        plt.plot(times_probs, ctc_probs[:, i])

    plt.ylabel('Posteriors (Word)', fontsize=12)
    plt.xlim([0, nframes * subsample_factor / 100])
    plt.ylim([0.05, 1.05])
    plt.xticks(list(range(0, int(nframes * subsample_factor / 100) + 1, 1)))
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)

    plt.subplot(212)
    plt.plot(times_probs, ctc_probs_sub[:, 0], ':', label='<blank>', color='grey')
    for i in range(1, ctc_probs_sub.shape[-1], 1):
        if space_id is not None and i == space_id:
            plt.plot(times_probs, ctc_probs_sub[:, space_id], label='<space>', color='black')
        else:
            plt.plot(times_probs, ctc_probs_sub[:, i])
    plt.xlabel(u'Time [sec]', fontsize=12)
    plt.ylabel('Posteriors (Char)', fontsize=12)
    plt.xlim([0, nframes * subsample_factor / 100])
    plt.ylim([0.05, 1.05])
    plt.xticks(list(range(0, int(nframes * subsample_factor / 100) + 1, 1)))
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path, dvi=500)

    plt.close()
