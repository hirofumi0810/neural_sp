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

    def __call__(self, log_probs, xlens, beam_width=1,
                 rnnlm=None, rnnlm_weight=0, length_penalty=0):
        """Performs inference for the given output probabilities.

        Args:
            log_probs (FloatTensor): The output log-scale probabilities
                (e.g. post-softmax) for each time step. `[B, T, vocab]`
            xlens (list): A list of length `[B]`
            beam_width (int): the size of beam
            rnnlm (RNNLM):
            rnnlm_weight (float): language model weight
            length_penalty (float): insertion bonus
        Returns:
            best_hyps (list): Best path hypothesis. `[B, L]`

        """
        bs, _, vocab = log_probs.size()
        best_hyps = []

        for b in range(bs):
            # Elements in the beam are (prefix, (p_blank, p_no_blank))
            # Initialize the beam with the empty sequence, a probability of
            # 1 for ending in blank and zero for ending in non-blank
            # (in log space).
            beam = [{'hyp': [],
                     'p_blank': LOG_1,
                     'p_nonblank': LOG_1,
                     'rnnlm_score': LOG_1,
                     'rnnlm_hxs': None,
                     'rnnlm_cxs': None}]

            for t in range(xlens[b]):
                new_beam = []

                # Pick up the top-k scores
                log_probs_topk, indices_topk = torch.topk(
                    log_probs[:, t, :], k=beam_width, dim=-1, largest=True, sorted=True)

                for c in tensor2np(indices_topk)[b]:
                    p_t = log_probs[b, t, c].item()

                    # The variables p_blank and p_nonblank are respectively the
                    # probabilities for the prefix given that it ends in a
                    # blank and does not end in a blank at this time step.
                    for i_beam in range(len(beam)):
                        prefix = beam[i_beam]['hyp']
                        p_blank = beam[i_beam]['p_blank']
                        p_nonblank = beam[i_beam]['p_nonblank']
                        rnnlm_score = beam[i_beam]['rnnlm_score']
                        rnnlm_hxs = beam[i_beam]['rnnlm_hxs']
                        rnnlm_cxs = beam[i_beam]['rnnlm_cxs']

                        # If we propose a blank the prefix doesn't change.
                        # Only the probability of ending in blank gets updated.
                        if c == self.blank:
                            new_p_blank = np.logaddexp(p_blank + p_t, p_nonblank + p_t)
                            new_beam.append({'hyp': beam[i_beam]['hyp'],
                                             'p_blank': new_p_blank,
                                             'p_nonblank': LOG_0,
                                             'rnnlm_score': rnnlm_score,
                                             'rnnlm_hxs': rnnlm_hxs[:] if rnnlm_hxs is not None else None,
                                             'rnnlm_cxs': rnnlm_cxs[:] if rnnlm_cxs is not None else None})
                            continue

                        # Extend the prefix by the new character c and it to the
                        # beam. Only the probability of not ending in blank gets
                        # updated.
                        prefix_end = prefix[-1] if len(prefix) > 0 else None
                        new_p_blank = LOG_0
                        new_p_nonblank = LOG_0
                        if c != prefix_end:
                            new_p_nonblank = np.logaddexp(p_blank + p_t, p_nonblank + p_t)
                        else:
                            # We don't include the previous probability of not ending
                            # in blank (p_nonblank) if c is repeated at the end. The CTC
                            # algorithm merges characters not separated by a
                            # blank.
                            new_p_nonblank = p_blank + p_t

                        # Update RNNLM states
                        if rnnlm_weight > 0 and rnnlm is not None:
                            y_rnnlm = log_probs.new(1, 1).fill_(c).long()
                            y_rnnlm = rnnlm.embed(y_rnnlm)
                            logits_step_rnnlm, rnnlm_out, rnnlm_state = rnnlm.predict(
                                y_rnnlm, h=(rnnlm_hxs, rnnlm_cxs))
                        else:
                            rnnlm_state = None

                        # # Add RNNLM score
                        if rnnlm_weight > 0 and rnnlm is not None:
                            rnnlm_log_probs = F.log_softmax(logits_step_rnnlm.squeeze(1), dim=1)
                            assert log_probs[:, t, :].size() == rnnlm_log_probs.size()
                            rnnlm_score = rnnlm_log_probs.data[0, c]

                        new_beam.append({'hyp': beam[i_beam]['hyp'] + [c],
                                         'p_blank': new_p_blank,
                                         'p_nonblank': new_p_nonblank,
                                         'rnnlm_score': rnnlm_score,
                                         'rnnlm_hxs': rnnlm_state[0][:] if rnnlm_state is not None else None,
                                         'rnnlm_cxs': rnnlm_state[1][:] if rnnlm_state is not None else None})

                        # If c is repeated at the end we also update the unchanged
                        # prefix. This is the merging case.
                        if c == prefix_end:
                            new_p_nonblank = p_nonblank + p_t
                            new_beam.append({'hyp': beam[i_beam]['hyp'],
                                             'p_blank': new_p_blank,
                                             'p_nonblank': new_p_nonblank,
                                             'rnnlm_score': rnnlm_score,
                                             'rnnlm_hxs': rnnlm_state[0][:] if rnnlm_state is not None else None,
                                             'rnnlm_cxs': rnnlm_state[1][:] if rnnlm_state is not None else None, })

                # Sort and trim the beam before moving on to the
                # next time-step.
                beam = sorted(new_beam,
                              key=lambda x: np.logaddexp(x['p_blank'], x['p_nonblank']) +
                              x['rnnlm_score'] * rnnlm_weight,
                              reverse=True)
                beam = beam[:beam_width]

            best_hyp = beam[0]['hyp']
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
            cs (torch.FloatTensor): array of next labels. A tensor of size `[beam_width]`
            r_prev (): previous CTC state
        Returns:
            ctc_scores ():
            ctc_states ():
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
