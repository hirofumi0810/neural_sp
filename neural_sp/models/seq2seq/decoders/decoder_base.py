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
import torch

from neural_sp.models.base import ModelBase
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list

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

    def _plot_attention(self):
        raise NotImplementedError

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
