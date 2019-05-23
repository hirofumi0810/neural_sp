#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Beam search (prefix search) decoder in numpy implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F

from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list
from neural_sp.models.torch_utils import tensor2np

LOG_0 = -float("inf")
LOG_1 = 0


class BeamSearchDecoder(object):
    """Beam search decoder.

    Arga:
        blank (int): the index of the blank label

    """

    def __init__(self, blank, space=-1):
        self.blank = blank
        self.space = space  # only for character-level CTC
        self.eos = 2

    def __call__(self, log_probs, xlens, beam_width=1,
                 lm=None, lm_weight=0, length_penalty=0, lm_usage='rescoring'):
        """Performs inference for the given output probabilities.

        Args:
            log_probs (FloatTensor): The output log-scale probabilities
                (e.g. post-softmax) for each time step. `[B, T, vocab]`
            xlens (list): A list of length `[B]`
            beam_width (int): the size of beam
            lm (RNNLM or GatedConvLM or TransformerLM):
            lm_weight (float): language model weight (the vocabulary is the same as CTC)
            length_penalty (float): insertion penalty
            lm_usage (str): rescoring or shallow_fusion
        Returns:
            best_hyps (list): Best path hypothesis. `[B, L]`

        """
        bs, _, vocab = log_probs.size()
        best_hyps = []

        for b in range(bs):
            # Elements in the beam are (prefix, (p_b, p_no_blank))
            # Initialize the beam with the empty sequence, a probability of
            # 1 for ending in blank and zero for ending in non-blank (in log space).
            beam = [{'hyp': [self.eos],
                     'p_b': LOG_1,
                     'p_nb': LOG_0,
                     'clm_score': LOG_1,
                     'clm_hxs': None,
                     'clm_cxs': None}]

            for t in range(xlens[b]):
                new_beam = []

                # Pick up the top-k scores
                log_probs_topk, indices_topk = torch.topk(
                    log_probs[b:b + 1, t], k=min(beam_width, vocab), dim=-1, largest=True, sorted=True)

                for i_beam in range(len(beam)):
                    hyp = beam[i_beam]['hyp']
                    p_b = beam[i_beam]['p_b']
                    p_nb = beam[i_beam]['p_nb']
                    clm_score = beam[i_beam]['clm_score']
                    clm_hxs = beam[i_beam]['clm_hxs']
                    clm_cxs = beam[i_beam]['clm_cxs']

                    # case 1. hyp is not extended
                    new_p_b = np.logaddexp(p_b + log_probs[b, t, self.blank].item(),
                                           p_nb + log_probs[b, t, self.blank].item())
                    if len(hyp) > 1:
                        new_p_nb = p_nb + log_probs[b, t, hyp[-1]].item()
                    else:
                        new_p_nb = LOG_0
                    new_beam.append({'hyp': hyp,
                                     'score': np.logaddexp(new_p_b, new_p_nb) + clm_score * lm_weight,
                                     'p_b': new_p_b,
                                     'p_nb': new_p_nb,
                                     'clm_score': clm_score,
                                     'clm_hxs': clm_hxs[:] if clm_hxs is not None else None,
                                     'clm_cxs': clm_cxs[:] if clm_cxs is not None else None})

                    # case 2. hyp is extended
                    new_p_b = LOG_0
                    for c in tensor2np(indices_topk)[0]:
                        p_t = log_probs[b, t, c].item()

                        if c == self.blank:
                            continue

                        last_token = hyp[-1] if len(hyp) > 1 else None
                        if c == last_token:
                            new_p_nb = p_b + p_t
                            # TODO(hirofumi): apply character LM here
                        else:
                            new_p_nb = np.logaddexp(p_b + p_t, p_nb + p_t)
                            # TODO(hirofumi): apply character LM here
                            if c == self.space:
                                pass
                                # TODO(hirofumi): apply word LM here

                        # Update LM states for shallow fusion
                        if lm_weight > 0 and lm is not None and lm_usage == 'shallow_fusion':
                            clmout, clmstate = lm.decode(
                                lm.encode(log_probs.new_zeros(1, 1).fill_(c).long()), (clm_hxs, clm_cxs))
                            clm_score = F.log_softmax(lm.generate(clmout), dim=-1)[0, 0, c]

                        new_beam.append({'hyp': beam[i_beam]['hyp'] + [c],
                                         'score': np.logaddexp(new_p_b, new_p_nb) + clm_score * lm_weight,
                                         'p_b': new_p_b,
                                         'p_nb': new_p_nb,
                                         'clm_score': clm_score,
                                         'clm_hxs': clm_hxs[:] if clm_hxs is not None else None,
                                         'clm_cxs': clm_cxs[:] if clm_cxs is not None else None})

                # Pruning
                beam = sorted(new_beam, key=lambda x: x['score'], reverse=True)[:beam_width]

            # Rescoing lattice
            if lm_weight > 0 and lm is not None and lm_usage == 'rescoring':
                new_beam = []
                device_id = torch.cuda.device_of(log_probs).idx
                for i_beam in range(len(beam)):
                    ys = [np2tensor(np.fromiter([self.eos] + beam[i_beam]['hyp'],
                                                dtype=np.int64), device_id).long()]
                    ys_pad = pad_list(ys, lm.pad)
                    clmout, _ = lm.decode(lm.encode(ys_pad), None)
                    clm_score = F.log_softmax(lm.generate(clmout), dim=-1).sum()
                    new_beam.append({'hyp': beam[i_beam]['hyp'],
                                     'score': np.logaddexp(beam[i_beam]['p_b'], beam[i_beam]['p_nb']) + clm_score * lm_weight})
                beam = sorted(new_beam, key=lambda x: x['score'], reverse=True)

            best_hyp = beam[0]['hyp'][1:]
            best_hyps.append(np.array(best_hyp))

        return np.array(best_hyps)


class CTCPrefixScore(object):
    """Compute CTC label sequence scores.

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the probablities of multiple labels
    simultaneously

    [Reference]:
        https://github.com/espnet/espnet
    """

    def __init__(self, log_probs, blank, eos):
        """
        Args:
            log_probs ():
            blank (int): index of <blank>
            eos (int): index of <eos>

        """
        self.blank = blank
        self.eos = eos
        self.xlen = len(log_probs)
        self.log_probs = log_probs
        self.logzero = -10000000000.0

    def initial_state(self):
        """Obtain an initial CTC state

        :return: CTC state
        """
        # initial CTC state is made of a frame x 2 tensor that corresponds to
        # r_t^n(<sos>) and r_t^b(<sos>), where 0 and 1 of axis=1 represent
        # superscripts n and b (non-blank and blank), respectively.
        r = np.full((self.xlen, 2), self.logzero, dtype=np.float32)
        r[0, 1] = self.log_probs[0, self.blank]
        for i in range(1, self.xlen):
            r[i, 1] = r[i - 1, 1] + self.log_probs[i, self.blank]
        return r

    def __call__(self, hyp, cs, r_prev):
        """Compute CTC prefix scores for next labels.

        Args:
            hyp (list): prefix label sequence
            cs (np.ndarray): array of next labels. A tensor of size `[beam_width]`
            r_prev (np.ndarray): previous CTC state
        Returns:
            ctc_scores (np.ndarray):
            ctc_states (np.ndarray):
        """
        # initialize CTC states
        ylen = len(hyp) - 1  # ignore sos
        # new CTC states are prepared as a frame x (n or b) x n_labels tensor
        # that corresponds to r_t^n(h) and r_t^b(h).
        r = np.ndarray((self.xlen, 2, len(cs)), dtype=np.float32)
        xs = self.log_probs[:, cs]
        if ylen == 0:
            r[0, 0] = xs[0]
            r[0, 1] = self.logzero
        else:
            r[ylen - 1] = self.logzero

        # prepare forward probabilities for the last label
        r_sum = np.logaddexp(r_prev[:, 0], r_prev[:, 1])  # log(r_t^n(g) + r_t^b(g))
        last = hyp[-1]
        if ylen > 0 and last in cs:
            log_phi = np.ndarray((self.xlen, len(cs)), dtype=np.float32)
            for i in range(len(cs)):
                log_phi[:, i] = r_sum if cs[i] != last else r_prev[:, 1]
        else:
            log_phi = r_sum

        # compute forward probabilities log(r_t^n(h)), log(r_t^b(h)),
        # and log prefix probabilites log(psi)
        start = max(ylen, 1)
        log_psi = r[start - 1, 0]
        for t in range(start, self.xlen):
            # non-blank
            r[t, 0] = np.logaddexp(r[t - 1, 0], log_phi[t - 1]) + xs[t]
            # blank
            r[t, 1] = np.logaddexp(r[t - 1, 0], r[t - 1, 1]) + self.log_probs[t, self.blank]
            log_psi = np.logaddexp(log_psi, log_phi[t - 1] + xs[t])

        # get P(...eos|X) that ends with the prefix itself
        eos_pos = np.where(cs == self.eos)[0]
        if len(eos_pos) > 0:
            log_psi[eos_pos] = r_sum[-1]  # log(r_T^n(g) + r_T^b(g))

        # return the log prefix probability and CTC states, where the label axis
        # of the CTC states is moved to the first axis to slice it easily
        return log_psi, np.rollaxis(r, 2)
