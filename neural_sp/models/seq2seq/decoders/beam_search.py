# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utility functions for beam search decoding."""

import numpy as np
import torch

from neural_sp.models.torch_utils import (
    np2tensor,
    pad_list,
    tensor2np,
)


class BeamSearch(object):
    def __init__(self, beam_width, eos, ctc_weight, device, beam_width_bwd=0):

        super(BeamSearch, self).__init__()

        self.beam_width = beam_width
        self.beam_width_bwd = beam_width_bwd
        self.eos = eos
        self.device = device

        self.ctc_weight = ctc_weight

    def remove_complete_hyp(self, hyps_sorted, end_hyps, prune=True, backward=False):
        new_hyps = []
        is_finish = False
        for hyp in hyps_sorted:
            if not backward and len(hyp['hyp']) > 1 and hyp['hyp'][-1] == self.eos:
                end_hyps += [hyp]
            elif backward and len(hyp['hyp_bwd']) > 1 and hyp['hyp_bwd'][-1] == self.eos:
                end_hyps += [hyp]
            else:
                new_hyps += [hyp]
        if len(end_hyps) >= self.beam_width + self.beam_width_bwd:
            if prune:
                end_hyps = end_hyps[:self.beam_width + self.beam_width_bwd]
            is_finish = True
        return new_hyps, end_hyps, is_finish

    def add_ctc_score(self, hyp, topk_ids, ctc_state, total_scores_topk,
                      ctc_prefix_scorer, new_chunk=False, backward=False):
        beam_width = self.beam_width_bwd if backward else self.beam_width
        if ctc_prefix_scorer is None:
            return None, topk_ids.new_zeros(beam_width), total_scores_topk

        ctc_scores, new_ctc_states = ctc_prefix_scorer(hyp, tensor2np(topk_ids[0]), ctc_state,
                                                       new_chunk=new_chunk)
        total_scores_ctc = torch.from_numpy(ctc_scores).to(self.device)
        total_scores_topk += total_scores_ctc * self.ctc_weight
        # Sort again
        total_scores_topk, joint_ids_topk = torch.topk(
            total_scores_topk, k=beam_width, dim=1, largest=True, sorted=True)
        topk_ids = topk_ids[:, joint_ids_topk[0]]
        new_ctc_states = new_ctc_states[joint_ids_topk[0].cpu().numpy()]
        return new_ctc_states, total_scores_ctc, total_scores_topk

    def add_lm_score(self, after_topk=True):
        raise NotImplementedError

    def update_rnnlm_state(self, lm, hyp, y, emb_cache=False):
        """Update RNNLM state for a single utterance.

        Args:
            lm (RNNLM): RNNLM
            hyp (dict): beam candiate
            y (LongTensor): `[1, 1]`
            emb_cache (bool): precompute token embeddings for fast infernece
        Returns:
            lmout (FloatTensor): `[1, 1, lm_n_units]`
            lmstate (dict):
            scores_lm (FloatTensor): `[1, 1, vocab]`

        """
        lmout, lmstate, scores_lm = None, None, None
        if lm is not None:
            lmout, lmstate, scores_lm = lm.predict(y, hyp['lmstate'], emb_cache=emb_cache)
        return lmout, lmstate, scores_lm

    def update_rnnlm_state_batch(self, lm, hyps, y, emb_cache=False):
        """Update RNNLM state in batch-mode.

        Args:
            lm (RNNLM): RNNLM
            hyps (List[dict]): beam candidates
            y (LongTensor): `[B, 1]`
            emb_cache (bool): precompute token embeddings for fast infernece
        Returns:
            lmout (FloatTensor): `[B, 1, lm_n_units]`
            lmstate (dict):
            scores_lm (FloatTensor): `[B, 1, vocab]`

        """
        lmout, lmstate, scores_lm = None, None, None
        if lm is not None:
            if hyps[0]['lmstate'] is not None:
                lm_hxs = torch.cat([beam['lmstate']['hxs'] for beam in hyps], dim=1)
                lm_cxs = torch.cat([beam['lmstate']['cxs'] for beam in hyps], dim=1)
                lmstate = {'hxs': lm_hxs, 'cxs': lm_cxs}
            lmout, lmstate, scores_lm = lm.predict(y, lmstate, emb_cache=emb_cache)
        return lmout, lmstate, scores_lm

    def lm_rescoring(self, hyps, lm, lm_weight, reverse=False, normalize=False,
                     tag=''):
        if lm is None:
            return
        for i in range(len(hyps)):
            ys = hyps[i]['hyp']  # include <sos>
            if reverse:
                ys = ys[::-1]

            ys = [np2tensor(np.fromiter(ys, dtype=np.int64), self.device)]
            ys_in = pad_list([y[:-1] for y in ys], -1)  # `[1, L-1]`
            ys_out = pad_list([y[1:] for y in ys], -1)  # `[1, L-1]`

            if ys_in.size(1) > 0:
                _, _, scores_lm = lm.predict(ys_in, None)
                score_lm = sum([scores_lm[0, t, ys_out[0, t]] for t in range(ys_out.size(1))])
                if normalize:
                    score_lm /= ys_out.size(1)  # normalize by length
            else:
                score_lm = 0

            hyps[i]['score'] += score_lm * lm_weight
            hyps[i]['score_lm_' + tag] = score_lm

    def verify_lm_eval_mode(self, lm, lm_weight):
        if lm is not None:
            assert lm_weight > 0
            lm.eval()
        return lm
