# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for language models."""

import logging
import numpy as np
import torch

from neural_sp.models.base import ModelBase
from neural_sp.models.criterion import cross_entropy_lsm
from neural_sp.models.torch_utils import compute_accuracy
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list

logger = logging.getLogger(__name__)


class LMBase(ModelBase):
    """Base class for language models."""

    def __init__(self, args):

        super(ModelBase, self).__init__()
        logger.info(self.__class__.__name__)
        logger.info('Overriding LMBase class.')

    def reset_parameters(self, param_init):
        raise NotImplementedError

    def forward(self, ys, state=None, is_eval=False, n_caches=0,
                ylens=[], predict_last=False):
        """Forward pass.

        Args:
            ys (list): length `B`, each of which contains arrays of size `[L]`
            state (tuple or list):
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
            n_caches (int): number of cached states
            ylens (list): not used
            predict_last (bool): used for TransformerLM and GatedConvLM
        Returns:
            loss (FloatTensor): `[1]`
            new_state (tuple or list):
            observation (dict):

        """
        if is_eval:
            self.eval()
            with torch.no_grad():
                loss, state, observation = self._forward(ys, state, n_caches, predict_last)
        else:
            self.train()
            loss, state, observation = self._forward(ys, state)
        return loss, state, observation

    def _forward(self, ys, state, n_caches=0, predict_last=False):
        ys = [np2tensor(y, self.device) for y in ys]  # <eos> is included
        ys = pad_list(ys, self.pad)
        ys_in, ys_out = ys[:, :-1], ys[:, 1:]

        logits, out, new_state = self.decode(ys_in, state=state, mems=state)
        # NOTE: state=state is used for RNNLM while mems=state is used for TransformerXL.
        # TransformerLM ignores both of them.

        if predict_last:
            ys_out = ys_out[:, -1].unsqueeze(1)
            logits = logits[:, -1].unsqueeze(1)

        # Compute XE sequence loss
        if n_caches > 0 and len(self.cache_ids) > 0:
            assert ys_out.size(1) == 1
            assert ys_out.size(0) == 1
            if self.adaptive_softmax is None:
                probs = torch.softmax(logits, dim=-1)
            else:
                probs = self.adaptive_softmax.log_prob(logits).exp()
            cache_probs = probs.new_zeros(probs.size())

            # Truncate cache
            self.cache_ids = self.cache_ids[-n_caches:]  # list of `[B, 1]`
            self.cache_keys = self.cache_keys[-n_caches:]  # list of `[B, 1, n_units]`

            # Compute inner-product over caches
            cache_attn = torch.softmax(self.cache_theta * torch.matmul(
                torch.cat(self.cache_keys, dim=1), out.transpose(2, 1)).squeeze(2), dim=1)

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
                loss, ppl = cross_entropy_lsm(logits, ys_out.contiguous(),
                                              self.lsm_prob, self.pad, self.training,
                                              normalize_length=True)
            else:
                loss = self.adaptive_softmax(logits.reshape((-1, logits.size(2))),
                                             ys_out.contiguous().view(-1)).loss
                ppl = np.exp(loss.item())

        if n_caches > 0:
            # Register to cache
            self.cache_ids += [ys_out[0, -1].item()]
            self.cache_keys += [out]

        # Compute token-level accuracy in teacher-forcing
        if self.adaptive_softmax is None:
            acc = compute_accuracy(logits, ys_out, pad=self.pad)
        else:
            acc = compute_accuracy(self.adaptive_softmax.log_prob(
                logits.reshape((-1, logits.size(2)))), ys_out, pad=self.pad)

        observation = {'loss.lm': loss.item(), 'acc.lm': acc, 'ppl.lm': ppl}
        return loss, new_state, observation

    def repackage_state(self, state):
        return state

    def reset_length(self, mem_len):
        # for TransformerXL
        self.mem_len = mem_len

    def decode(self, ys, state=None, mems=None, incremental=False):
        raise NotImplementedError

    def predict(self, ys, state=None, mems=None, cache=None, emb_cache=False):
        """Precict function for ASR.

        Args:
            ys (LongTensor): `[B, L]`
            state:
                - RNNLM: dict
                    hxs (FloatTensor): `[n_layers, B, n_units]`
                    cxs (FloatTensor): `[n_layers, B, n_units]`
                - TransformerLM (LongTensor): `[B, L]`
                - TransformerXL (list): length `n_layers + 1`, each of which contains a tensor`[B, L, d_model]`
            mems (list):
            cache (list):
            emb_cache (bool): precompute token embeddings for fast infernece
        Returns:
            lmout (FloatTensor): `[B, L, vocab]`, used for LM integration such as cold fusion
            state:
                - RNNLM: dict
                    hxs (FloatTensor): `[n_layers, B, n_units]`
                    cxs (FloatTensor): `[n_layers, B, n_units]`
                - TransformerLM (LongTensor): `[B, L]`
                - TransformerXL (list): length `n_layers + 1`, each of which contains a tensor`[B, L, d_model]`
            log_probs (FloatTensor): `[B, L, vocab]`

        """
        logits, lmout, new_state = self.decode(ys, state, mems=mems, cache=cache,
                                               incremental=True, emb_cache=emb_cache)
        log_probs = torch.log_softmax(logits, dim=-1)
        return lmout, new_state, log_probs

    def plot_attention(self):
        # raise NotImplementedError
        pass
