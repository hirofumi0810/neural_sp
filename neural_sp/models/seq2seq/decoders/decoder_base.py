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
import torch.nn.functional as F

from neural_sp.models.base import ModelBase
from neural_sp.models.torch_utils import tensor2np


class DecoderBase(ModelBase):
    """Base class for decoders."""

    def __init__(self):

        super(ModelBase, self).__init__()
        logger = logging.getLogger('training')
        logger.info('Overriding DecoderBase class.')

    @property
    def device_id(self):
        return torch.cuda.device_of(next(self.parameters()).data).idx

    def reset_parameters(self, param_init):
        raise NotImplementedError

    def greedy(self):
        raise NotImplementedError

    def beam_search(self):
        raise NotImplementedError

    def _plot_attention(self):
        raise NotImplementedError

    def decode_ctc(self, eouts, elens, beam_width=1, lm=None, lm_weight=0.0,
                   lm_usage='rescoring'):
        """Decoding by the CTC layer in the inference stage.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            beam_width (int): size of beam
            lm (RNNLM or GatedConvLM or TransformerLM):
            lm_weight (float): language model weight (the vocabulary is the same as CTC)
            lm_usage (str): rescoring or shallow_fusion
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`

        """
        if beam_width == 1:
            best_hyps = self.ctc.greedy(eouts, elens)
        else:
            best_hyps = self.ctc.beam_search(eouts, elens, beam_width,
                                             lm, lm_weight,
                                             lm_usage=lm_usage)
        return best_hyps

    def ctc_log_probs(self, eouts, temperature=1):
        return F.log_softmax(self.ctc.output(eouts) / temperature, dim=-1)

    def ctc_probs_topk(self, eouts, temperature, topk):
        probs = F.softmax(self.ctc.output(eouts) / temperature, dim=-1)
        if topk is None:
            topk = probs.size(-1)
        _, topk_ids = torch.topk(probs.sum(1), k=topk, dim=-1, largest=True, sorted=True)
        return tensor2np(probs), tensor2np(topk_ids)
