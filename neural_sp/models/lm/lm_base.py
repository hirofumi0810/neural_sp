#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for language models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn.functional as F

from neural_sp.models.base import ModelBase
from neural_sp.models.torch_utils import compute_accuracy
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


class LMBase(ModelBase):
    """RNN language model."""

    def __init__(self, args):

        super(ModelBase, self).__init__()
        logger = logging.getLogger('training')
        logger.info('Overriding LMBase class.')

    def reset_parameters(self, param_init):
        raise NotImplementedError

    def forward(self, ys, hidden=None, reporter=None, is_eval=False, n_caches=0,
                ylens=[], predict_last=False):
        """Forward computation.

        Args:
            ys (list): A list of length `[B]`, which contains arrays of size `[L]`
            hidden (tuple or list): (h_n, c_n) or (hxs, cxs)
            reporter ():
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
            n_caches (int):
            ylens (list): not used
            predict_last (bool): used for TransformerLM and GatedConvLM
        Returns:
            loss (FloatTensor): `[1]`
            hidden (tuple or list): (h_n, c_n) or (hxs, cxs)
            reporter ():

        """
        if is_eval:
            self.eval()
            with torch.no_grad():
                loss, hidden, reporter = self._forward(ys, hidden, reporter, n_caches, predict_last)
        else:
            self.train()
            loss, hidden, reporter = self._forward(ys, hidden, reporter)

        return loss, hidden, reporter

    def _forward(self, ys, hidden, reporter, n_caches=0, predict_last=False):
        ys = [np2tensor(y, self.device_id).long() for y in ys]
        ys = pad_list(ys, self.pad)
        ys_in = ys[:, :-1]
        ys_out = ys[:, 1:]

        lmout, hidden = self.decode(self.encode(ys_in), hidden)
        if self.adaptive_softmax is None:
            logits = self.generate(lmout)
        else:
            logits = lmout

        if predict_last:
            ys_out = ys_out[:, -1].unsqueeze(1)
            logits = logits[:, -1].unsqueeze(1)

        # Compute XE sequence loss
        if n_caches > 0 and len(self.cache_ids) > 0:
            assert ys_out.size(1) == 1
            assert ys_out.size(0) == 1
            if self.adaptive_softmax is None:
                probs = F.softmax(logits, dim=-1)
            else:
                probs = self.adaptive_softmax.log_prob(logits).exp()
            cache_probs = probs.new_zeros(probs.size())

            # Truncate cache
            self.cache_ids = self.cache_ids[-n_caches:]  # list of `[B, 1]`
            self.cache_keys = self.cache_keys[-n_caches:]  # list of `[B, 1, n_units]`

            # Compute inner-product over caches
            cache_attn = F.softmax(self.cache_theta * torch.matmul(
                torch.cat(self.cache_keys, dim=1), lmout.transpose(2, 1)).squeeze(2), dim=1)

            # For visualization
            if len(self.cache_ids) == n_caches:
                self.cache_attn += [cache_attn.cpu().numpy()]
                self.cache_attn = self.cache_attn[-n_caches:]

            # Sum all probabilities
            for offset, idx in enumerate(self.cache_ids):
                cache_probs[:, :, idx] += cache_attn[:, offset]
            probs = (1 - self.cache_lambda) * probs + self.cache_lambda * cache_probs
            loss = -torch.log(probs[:, :, ys_out[:, -1]])
        else:
            if self.adaptive_softmax is None:
                loss = F.cross_entropy(logits.view((-1, logits.size(2))),
                                       ys_out.contiguous().view(-1),
                                       ignore_index=self.pad, size_average=True)
            else:
                loss = self.adaptive_softmax(logits.view((-1, logits.size(2))),
                                             ys_out.contiguous().view(-1)).loss

        if n_caches > 0:
            # Register to cache
            self.cache_ids += [ys_out[0, -1].item()]
            self.cache_keys += [lmout]

        # Compute token-level accuracy in teacher-forcing
        if self.adaptive_softmax is None:
            acc = compute_accuracy(logits, ys_out, pad=self.pad)
        else:
            acc = compute_accuracy(self.adaptive_softmax.log_prob(
                logits.view((-1, logits.size(2)))), ys_out, pad=self.pad)

        observation = {'loss.lm': loss.item(),
                       'acc.lm': acc,
                       'ppl.lm': np.exp(loss.item())}

        # Report here
        if reporter is not None:
            is_eval = not self.training
            reporter.add(observation, is_eval)

        return loss, hidden, reporter

    def encode(self, ys):
        """Encode function.

        Args:
            ys (LongTensor): `[B, L]`
        Returns:
            ys (FloatTensor): `[B, L, emb_dim]`

        """
        return self.embed(ys)

    def generate(self, hidden):
        """Generate function.

        Args:
            hidden (FloatTensor): `[B, T, n_units]`
        Returns:
            logits (FloatTensor): `[B, T, vocab]`

        """
        return self.output(hidden)
