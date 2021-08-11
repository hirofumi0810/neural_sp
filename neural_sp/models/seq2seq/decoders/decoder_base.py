# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for decoders."""

import logging
import numpy as np
import os
import shutil

from neural_sp.models.base import ModelBase

import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


class DecoderBase(ModelBase):
    """Base class for decoders."""

    def __init__(self):

        super(ModelBase, self).__init__()

        logger.info('Overriding DecoderBase class.')

    def reset_session(self):
        self._new_session = True

    def trigger_scheduled_sampling(self):
        logger.info('Activate scheduled sampling')
        self._ss_prob = getattr(self, 'ss_prob', 0)

    def trigger_quantity_loss(self):
        if getattr(self, 'attn_type', '') == 'mocha':
            logger.info('Activate quantity loss')
            self._quantity_loss_weight = getattr(self, 'quantity_loss_weight', 0)

    def trigger_latency_loss(self):
        if getattr(self, 'attn_type', '') == 'mocha':
            logger.info('Activate latency loss')
            self._latency_loss_weight = getattr(self, 'latency_loss_weight', 0)

    def trigger_stableemit(self):
        if getattr(self, 'attn_type', '') == 'mocha':
            if hasattr(self, 'score'):
                self.score.trigger_stableemit()
            elif hasattr(self, 'layers'):
                pass  # TODO(hirofumi): MMA

    def greedy(self, eouts, elens, max_len_ratio):
        raise NotImplementedError

    def embed_token_id(self, indices):
        raise NotImplementedError

    def cache_embedding(self, device):
        raise NotImplementedError

    def initialize_beam(self, hyp, lmstate):
        raise NotImplementedError

    def beam_search(self, eouts, elens, params, idx2token):
        raise NotImplementedError

    def _plot_attention(self, save_path=None, n_cols=2):
        """Plot attention for each head in all decoder layers."""
        if len(getattr(self, 'aws_dict', {}).keys()) == 0:
            return

        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator

        elens = self.data_dict['elens']
        ylens = self.data_dict['ylens']
        # ys = self.data_dict['ys']
        aws_dict = self.aws_dict

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        for k, aw in aws_dict.items():
            if aw is None:
                continue

            plt.clf()
            n_heads = aw.shape[1]
            n_cols_tmp = 1 if n_heads == 1 else n_cols * max(1, n_heads // 4)
            fig, axes = plt.subplots(max(1, n_heads // n_cols_tmp), n_cols_tmp,
                                     figsize=(20 * max(1, n_heads // 4), 8), squeeze=False)
            for h in range(n_heads):
                ax = axes[h // n_cols_tmp, h % n_cols_tmp]
                if 'yy' in k:
                    ax.imshow(aw[-1, h, :ylens[-1], :ylens[-1]], aspect="auto")
                else:
                    ax.imshow(aw[-1, h, :ylens[-1], :elens[-1]], aspect="auto")
                # NOTE: show the last utterance in a mini-batch
                ax.grid(False)
                ax.set_xlabel("Input (head%d)" % h)
                ax.set_ylabel("Output (head%d)" % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                # ax.set_yticks(np.linspace(0, ylens[-1] - 1, ylens[-1]))
                # ax.set_yticks(np.linspace(0, ylens[-1] - 1, 1), minor=True)
                # ax.set_yticklabels(ys + [''])

            fig.tight_layout()
            if save_path is not None:
                fig.savefig(os.path.join(save_path, '%s.png' % k))
            plt.close()

    def _plot_ctc(self, save_path=None, topk=10):
        """Plot CTC posterior probabilities."""
        if self.ctc_weight == 0:
            return
        if len(self.ctc.prob_dict.keys()) == 0:
            return

        from matplotlib import pyplot as plt

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        elen = self.ctc.data_dict['elens'][-1]
        probs = self.ctc.prob_dict['probs'][-1, :elen]  # `[T, vocab]`
        # NOTE: show the last utterance in a mini-batch

        topk_ids = np.argsort(probs, axis=1)

        plt.clf()
        n_frames = probs.shape[0]
        times_probs = np.arange(n_frames)
        plt.figure(figsize=(20, 8))

        # NOTE: index 0 is reserved for blank
        for idx in set(topk_ids.reshape(-1).tolist()):
            if idx == 0:
                plt.plot(times_probs, probs[:, 0], ':', label='<blank>', color='grey')
            else:
                plt.plot(times_probs, probs[:, idx])
        plt.xlabel(u'Time [frame]', fontsize=12)
        plt.ylabel('Posteriors', fontsize=12)
        plt.xticks(list(range(0, int(n_frames) + 1, 10)))
        plt.yticks(list(range(0, 2, 1)))

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'prob.png'))
        plt.close()
