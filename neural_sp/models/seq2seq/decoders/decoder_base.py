#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for decoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import os
import torch
import shutil

from neural_sp.models.base import ModelBase
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list
from neural_sp.utils import mkdir_join

import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


class DecoderBase(ModelBase):
    """Base class for decoders."""

    def __init__(self):

        super(ModelBase, self).__init__()

        logger.info('Overriding DecoderBase class.')

    @property
    def device_id(self):
        return torch.cuda.device_of(next(self.parameters()).data).idx

    def reset_parameters(self, param_init):
        raise NotImplementedError

    def reset_session(self):
        self.new_session = True

    def greedy(self, eouts, elens, max_len_ratio):
        raise NotImplementedError

    def beam_search(self, eouts, elens, params, idx2token):
        raise NotImplementedError

    def _plot_attention(self, save_path, n_cols=2):
        """Plot attention for each head in all decoder layers."""
        if getattr(self, 'att_weight', 0) == 0:
            return
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator

        _save_path = mkdir_join(save_path, 'dec_att_weights')

        # Clean directory
        if _save_path is not None and os.path.isdir(_save_path):
            shutil.rmtree(_save_path)
            os.mkdir(_save_path)

        elens = self.data_dict['elens']
        ylens = self.data_dict['ylens']
        # ys = self.data_dict['ys']

        for k, aw in self.aws_dict.items():
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
            fig.savefig(os.path.join(_save_path, '%s.png' % k), dvi=500)
            plt.close()

    def _plot_ctc(self, save_path, topk=10):
        """Plot CTC posteriors."""
        if self.ctc_weight == 0:
            return
        from matplotlib import pyplot as plt

        _save_path = mkdir_join(save_path, 'ctc')

        # Clean directory
        if _save_path is not None and os.path.isdir(_save_path):
            shutil.rmtree(_save_path)
            os.mkdir(_save_path)

        elen = self.ctc.data_dict['elens'][-1]
        probs = self.ctc.prob_dict['probs'][-1, :elen]  # `[T, vocab]`
        # NOTE: show the last utterance in a mini-batch

        topk_ids = np.argsort(probs, axis=1)

        plt.clf()
        n_frames = probs.shape[0]
        times_probs = np.arange(n_frames)

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
        plt.savefig(os.path.join(_save_path, '%s.png' % 'prob'), dvi=500)
        plt.close()

    def decode_ctc(self, eouts, elens, params, idx2token,
                   lm=None, lm_2nd=None, lm_2nd_rev=None,
                   nbest=1, refs_id=None, utt_ids=None, speakers=None):
        """Decoding with CTC scores in the inference stage.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            params (dict):
                recog_beam_width (int): size of beam
                recog_length_penalty (float): length penalty
                recog_lm_weight (float): weight of first path LM score
                recog_lm_second_weight (float): weight of second path LM score
                recog_lm_rev_weight (float): weight of second path backward LM score
            lm: firsh path LM
            lm_2nd: second path LM
            lm_2nd_rev: secoding path backward LM
        Returns:
            probs (FloatTensor): `[B, T, vocab]`
            topk_ids (LongTensor): `[B, T, topk]`
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`

        """
        if params['recog_beam_width'] == 1:
            best_hyps = self.ctc.greedy(eouts, elens)
        else:
            best_hyps = self.ctc.beam_search(eouts, elens, params, idx2token,
                                             lm, lm_2nd, lm_2nd_rev,
                                             nbest, refs_id, utt_ids, speakers)
        return best_hyps

    def ctc_probs(self, eouts, temperature=1.):
        """Return CTC probabilities.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
        Returns:
            probs (FloatTensor): `[B, T, vocab]`

        """
        return torch.softmax(self.ctc.output(eouts) / temperature, dim=-1)

    def ctc_log_probs(self, eouts, temperature=1.):
        """Return log-scale CTC probabilities.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
        Returns:
            log_probs (FloatTensor): `[B, T, vocab]`

        """
        return torch.log_softmax(self.ctc.output(eouts) / temperature, dim=-1)

    def ctc_probs_topk(self, eouts, temperature=1., topk=None):
        """Get CTC top-K probabilities.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            temperature (float): softmax temperature
            topk (int):
        Returns:
            probs (FloatTensor): `[B, T, vocab]`
            topk_ids (LongTensor): `[B, T, topk]`

        """
        probs = torch.softmax(self.ctc.output(eouts) / temperature, dim=-1)
        if topk is None:
            topk = probs.size(-1)
        _, topk_ids = torch.topk(probs, k=topk, dim=-1, largest=True, sorted=True)
        return probs, topk_ids

    def lm_rescoring(self, hyps, lm, lm_weight, reverse=False, tag=''):
        for i in range(len(hyps)):
            ys = hyps[i]['hyp']  # include <sos>
            if reverse:
                ys = ys[::-1]

            ys = [np2tensor(np.fromiter(ys, dtype=np.int64), self.device_id)]
            ys_in = pad_list([y[:-1] for y in ys], -1)  # `[1, L-1]`
            ys_out = pad_list([y[1:] for y in ys], -1)  # `[1, L-1]`

            lmout, lmstate, scores_lm = lm.predict(ys_in, None)
            score_lm = sum([scores_lm[0, t, ys_out[0, t]] for t in range(ys_out.size(1))])
            score_lm /= ys_out.size(1)

            hyps[i]['score'] += score_lm * lm_weight
            hyps[i]['score_lm_' + tag] = score_lm
