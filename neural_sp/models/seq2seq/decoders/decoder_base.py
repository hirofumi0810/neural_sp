#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for decoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch

from neural_sp.models.base import ModelBase

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
