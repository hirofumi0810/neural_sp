#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Beam search (prefix search) decoder in numpy implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import torch
import torch.nn.functional as F

from neural_sp.models.torch_utils import var2np

LOG_0 = -float("inf")
LOG_1 = 0


class BeamSearchDecoder(object):
    """Beam search decoder.

    Arga:
        blank_index (int): the index of the blank label

    """

    def __init__(self, blank_index):
        self.blank = blank_index

    def __call__(self, log_probs, x_lens, beam_width=1,
                 rnnlm=None, rnnlm_weight=0., length_penalty=0., space_index=5):
        """Performs inference for the given output probabilities.

        Args:
            log_probs (torch.autograd.Variable): The output log-scale probabilities
                (e.g. post-softmax) for each time step. `[B, T, num_classes]`
            x_lens (list): A list of length `[B]`
            beam_width (int): the size of beam
            rnnlm ():
            rnnlm_weight (float): language model weight
            length_penalty (float): insertion bonus
            space_index (int, optional): the index of the space label This is used for character-level CTC.
        Returns:
            best_hyps (list): Best path hypothesis. `[B, labels_max_seq_len]`
            best_hyps_lens (list): Lengths of best path hypothesis. `[B]`

        """
        assert isinstance(log_probs, torch.autograd.Variable)
        batch_size, _, num_classes = log_probs.size()
        best_hyps = []

        for b in six.moves.range(batch_size):
            # Elements in the beam are (prefix, (p_blank, p_no_blank))
            # Initialize the beam with the empty sequence, a probability of
            # 1 for ending in blank and zero for ending in non-blank
            # (in log space).
            beam = [{'hyp': [],
                     'p_blank': LOG_1,
                     'p_nonblank': LOG_1,
                     'rnnlm_score': LOG_1,
                     'rnnlm_state': None}]

            for t in six.moves.range(x_lens[b]):
                new_beam = []

                # Pick up the top-k scores
                log_probs_topk, indices_topk = torch.topk(
                    log_probs[:, t, :], k=beam_width, dim=-1, largest=True, sorted=True)

                for c in var2np(indices_topk)[b]:
                    p_t = log_probs[b, t, c].item()

                    # The variables p_blank and p_nonblank are respectively the
                    # probabilities for the prefix given that it ends in a
                    # blank and does not end in a blank at this time step.
                    for i_beam in six.moves.range(len(beam)):
                        prefix = beam[i_beam]['hyp']
                        p_blank = beam[i_beam]['p_blank']
                        p_nonblank = beam[i_beam]['p_nonblank']
                        rnnlm_score = beam[i_beam]['rnnlm_score']
                        rnnlm_state = beam[i_beam]['rnnlm_state']

                        # If we propose a blank the prefix doesn't change.
                        # Only the probability of ending in blank gets updated.
                        if c == self.blank:
                            new_p_blank = np.logaddexp(
                                p_blank + p_t, p_nonblank + p_t)
                            new_beam.append({'hyp': beam[i_beam]['hyp'],
                                             'p_blank': new_p_blank,
                                             'p_nonblank': LOG_0,
                                             'rnnlm_score': rnnlm_score,
                                             'rnnlm_state': rnnlm_state})
                            continue

                        # Extend the prefix by the new character c and it to the
                        # beam. Only the probability of not ending in blank gets
                        # updated.
                        prefix_end = prefix[-1] if len(prefix) > 0 else None
                        new_p_blank = LOG_0
                        new_p_nonblank = LOG_0
                        if c != prefix_end:
                            new_p_nonblank = np.logaddexp(
                                p_blank + p_t, p_nonblank + p_t)
                        else:
                            # We don't include the previous probability of not ending
                            # in blank (p_nonblank) if c is repeated at the end. The CTC
                            # algorithm merges characters not separated by a
                            # blank.
                            new_p_nonblank = p_blank + p_t

                        # Update RNNLM states
                        if rnnlm_weight > 0 and rnnlm is not None:
                            y_rnnlm = Variable(log_probs.new(1, 1).fill_(c).long(), volatile=True)
                            y_rnnlm = rnnlm.embed(y_rnnlm)
                            logits_step_rnnlm, rnnlm_out, rnnlm_state = rnnlm.predict(
                                y_rnnlm, h=rnnlm_state)

                        # # Add RNNLM score
                        if rnnlm_weight > 0 and rnnlm is not None:
                            rnnlm_log_probs = F.log_softmax(
                                logits_step_rnnlm.squeeze(1), dim=1)
                            assert log_probs[:, t, :].size(
                            ) == rnnlm_log_probs.size()
                            rnnlm_score = rnnlm_log_probs.data[0, c]

                        new_beam.append({'hyp': beam[i_beam]['hyp'] + [c],
                                         'p_blank': new_p_blank,
                                         'p_nonblank': new_p_nonblank,
                                         'rnnlm_score': rnnlm_score,
                                         'rnnlm_state': rnnlm_state})

                        # If c is repeated at the end we also update the unchanged
                        # prefix. This is the merging case.
                        if c == prefix_end:
                            new_p_nonblank = p_nonblank + p_t
                            new_beam.append({'hyp': beam[i_beam]['hyp'],
                                             'p_blank': new_p_blank,
                                             'p_nonblank': new_p_nonblank,
                                             'rnnlm_score': rnnlm_score,
                                             'rnnlm_state': rnnlm_state})

                # Sort and trim the beam before moving on to the
                # next time-step.
                beam = sorted(new_beam,
                              key=lambda x: np.logaddexp(
                                  x['p_blank'], x['p_nonblank']) + x['rnnlm_score'] * rnnlm_weight,
                              reverse=True)
                beam = beam[:beam_width]

            best_hyp = beam[0]['hyp']
            best_hyps.append(np.array(best_hyp))

        return np.array(best_hyps)
