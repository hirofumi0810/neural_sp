# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Plot attention weights & ctc probabilities."""

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use('Agg')

plt.style.use('ggplot')
sns.set_style("white")
blue = '#4682B4'
orange = '#D2691E'
green = '#006400'

# sns.set(font='IPAMincho')
sns.set(font='Noto Sans CJK JP')


def plot_attention_weights(aw, tokens=[], spectrogram=None, factor=4,
                           save_path=None, figsize=(20, 6),
                           ref=None, ctc_probs=None, ctc_topk_ids=None):
    """Plot attention weights.

    Args:
        aw (np.ndarray): A tensor of size `[H, L, T]
        tokens (list): tokens in hypothesis
        spectrogram (np.ndarray): A tensor of size `[T, input_dim]`
        factor (int): subsampling factor
        save_path (str): path to save a figure
        figsize (tuple):
        ref (str): reference text

    """
    n_heads = aw.shape[0]
    n_col = n_heads
    if spectrogram is not None:
        n_col += 1
    if ctc_probs is not None:
        n_col += 1
    if n_heads > 1:
        figsize = (20, 16)

    plt.clf()
    plt.figure(figsize=figsize)
    # Plot attention weights
    for h in range(1, n_heads + 1):
        plt.subplot(n_col, 1, n_heads - h + 1)
        sns.heatmap(aw[h - 1, :, :], cmap='viridis',
                    xticklabels=False,
                    yticklabels=tokens if len(tokens) > 0 else False,
                    cbar=False,
                    cbar_kws={"orientation": "horizontal"})
        plt.ylabel(u'Output labels (←)', fontsize=12 if n_heads == 1 else 8)
        plt.yticks(rotation=0, fontsize=6)

    # Plot CTC propabilities for joint CTC-attention
    if ctc_probs is not None:
        plt.subplot(n_col, 1, n_heads + 1)
        times_probs = np.arange(ctc_probs.shape[0])
        for idx in set(ctc_topk_ids.reshape(-1).tolist()):
            if idx == 0:
                plt.plot(times_probs, ctc_probs[:, 0], ':', label='<blank>', color='grey')
            else:
                plt.plot(times_probs, ctc_probs[:, idx])
        plt.ylabel('CTC posteriors', fontsize=12 if n_heads == 1 else 8)
        plt.tick_params(labelbottom=False)
        plt.yticks(list(range(0, 2, 1)))
        plt.xlim(0, ctc_probs.shape[0])

    # Plot spectrogram
    if spectrogram is not None:
        ax = plt.subplot(n_col, 1, n_col)
        plt.imshow(spectrogram.T, cmap='viridis', aspect='auto', origin='lower')
        freq = 200  # for plot
        plt.xticks(np.arange(0, len(spectrogram) + 1, freq),
                   np.arange(0, len(spectrogram) * factor + 1, freq * factor))
        plt.xlabel(u'Time [frame/10ms]', fontsize=12)
        plt.ylabel(u'Frequency bin', fontsize=12 if n_heads == 1 else 8)
        # plt.colorbar()
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    # if ref is not None:
    #     plt.title('REF: ' + ref + '\n' + 'HYP: ' + ' '.join(tokens).replace('▁', ' '), fontsize=12)

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path)

    plt.close()


def plot_hierarchical_attention_weights(aw, aw_sub, tokens=[], tokens_sub=[],
                                        spectrogram=None, factor=4,
                                        save_path=None, figsize=(20, 6),
                                        ref=None):
    """Plot attention weights for the hierarchical model.

    Args:
        aw (np.ndarray): A tensor of size `[H, L, T]`
        aw_sub (np.ndarray): A tensor of size `[H_sub, L_sub, T]`
        tokens (list): tokens in hypothesis for the main task
        tokens_sub (list): tokens in hypothesis for the auxiliary task
        spectrogram (np.ndarray): A tensor of size `[T, input_dim]`
        factor (int): subsampling factor
        save_path (str): path to save a figure
        figsize (tuple):
        ref (str):

    """
    n_col = 2
    if spectrogram is not None:
        n_col += 1

    plt.clf()
    plt.figure(figsize=figsize)

    plt.subplot(n_col, 1, 1)
    sns.heatmap(aw[0], cmap='viridis',
                xticklabels=False,
                yticklabels=tokens if len(tokens) > 0 else False)
    plt.ylabel(u'Output labels (main) (←)', fontsize=12)
    plt.yticks(rotation=0, fontsize=6)

    plt.subplot(n_col, 1, 2)
    sns.heatmap(aw_sub[0], cmap='viridis',
                xticklabels=False,
                yticklabels=tokens_sub if len(tokens_sub) > 0 else False)
    if spectrogram is None:
        plt.xlabel(u'Time [frame]', fontsize=12)
    plt.ylabel(u'Output labels (sub) (←)', fontsize=12)
    plt.yticks(rotation=0, fontsize=6)

    # Plot spectrogram
    if spectrogram is not None:
        ax = plt.subplot(n_col, 1, 3)
        plt.imshow(spectrogram.T, cmap='viridis', aspect='auto', origin='lower')
        freq = 200  # for plot
        plt.xticks(np.arange(0, len(spectrogram) + 1, freq),
                   np.arange(0, len(spectrogram) * factor + 1, freq * factor))
        plt.xlabel(u'Time [frame/10ms]', fontsize=12)
        plt.ylabel(u'Frequency bin', fontsize=12)
        # plt.colorbar()
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path)

    plt.close()


def plot_ctc_probs(ctc_probs, topk_ids, spectrogram=None, factor=4,
                   save_path=None, figsize=(20, 6),
                   topk=None, space=-1, hyp=None):
    """Plot CTC posteriors.

    Args:
        ctc_probs (np.ndarray): A tensor of size `[T, vocab]`
        topk_ids ():
        spectrogram (np.ndarray): A tensor of size `[T, input_dim]`
        factor (int): subsampling factor
        save_path (str): path to save a figure
        figsize (tuple):
        topk (int):
        space (int): index for space mark
        hyp (str):

    """
    n_col = 1
    if spectrogram is not None:
        n_col += 1

    plt.clf()
    plt.figure(figsize=figsize)
    if hyp is not None:
        plt.title(hyp)

    plt.ylim([0.05, 1.05])
    plt.legend(loc="upper right", fontsize=12)

    if spectrogram is not None:
        plt.subplot(n_col, 1, 1)
    times_probs = np.arange(ctc_probs.shape[0])
    # NOTE: index 0 is reserved for blank
    for idx in set(topk_ids.reshape(-1).tolist()):
        if idx == 0:
            plt.plot(times_probs, ctc_probs[:, 0], ':', label='<blank>', color='grey')
        elif idx == space:
            plt.plot(times_probs, ctc_probs[:, space], label='<space>', color='black')
        else:
            plt.plot(times_probs, ctc_probs[:, idx])
    if spectrogram is None:
        plt.xlabel(u'Time [frame]', fontsize=12)
    plt.ylabel('Posteriors', fontsize=12)
    plt.tick_params(labelbottom=False)
    plt.yticks(list(range(0, 2, 1)))

    # Plot spectrogram
    if spectrogram is not None:
        ax = plt.subplot(n_col, 1, 2)
        plt.imshow(spectrogram.T, cmap='viridis', aspect='auto', origin='lower')
        freq = 200  # for plot
        plt.xticks(np.arange(0, len(spectrogram) + 1, freq),
                   np.arange(0, len(spectrogram) * factor + 1, freq * factor))
        plt.xlabel(u'Time [frame/10ms]', fontsize=12)
        plt.ylabel(u'Frequency bin', fontsize=12)
        # plt.colorbar()
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path)

    plt.close()


def plot_hierarchical_ctc_probs(ctc_probs, topk_ids, ctc_probs_sub, topk_ids_sub,
                                spectrogram=None, factor=4,
                                save_path=None, figsize=(20, 6),
                                space=-1, space_sub=-1, hyp=None, hyp_sub=None):
    """Plot CTC posteriors for the hierarchical model.

    Args:
        ctc_probs (np.ndarray): A tensor of size `[T, vocab]`
        topk_ids ():
        ctc_probs_sub (np.ndarray): A tensor of size `[T, vocab_sub]`
        topk_ids_sub ():
        spectrogram (np.ndarray): A tensor of size `[T, input_dim]`
        factor (int): subsampling factor
        save_path (str): path to save a figure
        figsize (tuple):
        space (int): index for space mark

    """
    n_col = 2
    if spectrogram is not None:
        n_col += 1

    plt.clf()
    plt.figure(figsize=figsize)
    if hyp is not None:
        plt.title(hyp)
        # plt.title(hyp_sub)

    # NOTE: index 0 is reserved for blank
    plt.subplot(n_col, 1, 1)
    times_probs = np.arange(ctc_probs.shape[0])
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
    plt.ylim([0.05, 1.05])
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)

    plt.subplot(n_col, 1, 2)
    plt.plot(times_probs, ctc_probs_sub[:, 0], ':', label='<blank>', color='grey')
    for idx in set(topk_ids_sub.reshape(-1).tolist()):
        if idx == 0:
            plt.plot(times_probs, ctc_probs_sub[:, 0], ':', label='<blank>', color='grey')
        elif idx == space_sub:
            plt.plot(times_probs, ctc_probs_sub[:, space_sub], label='<space>', color='black')
        else:
            plt.plot(times_probs, ctc_probs_sub[:, idx])
    if spectrogram is None:
        plt.xlabel(u'Time [frame]', fontsize=12)
    plt.ylabel('Posteriors (Char)', fontsize=12)
    plt.ylim([0.05, 1.05])
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)

    # Plot spectrogram
    if spectrogram is not None:
        ax = plt.subplot(n_col, 1, 3)
        plt.imshow(spectrogram.T, cmap='viridis', aspect='auto', origin='lower')
        freq = 200  # for plot
        plt.xticks(np.arange(0, len(spectrogram) + 1, freq),
                   np.arange(0, len(spectrogram) * factor + 1, freq * factor))
        plt.xlabel(u'Time [frame/10ms]', fontsize=12)
        plt.ylabel(u'Frequency bin', fontsize=12)
        # plt.colorbar()
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    # Save as a png file
    if save_path is not None:
        plt.savefig(save_path)

    plt.close()
