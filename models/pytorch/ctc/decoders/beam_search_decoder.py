#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Beam search (prefix search) decoder in numpy implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import defaultdict

NEG_INF = -float("inf")
LOG_0 = NEG_INF
LOG_1 = 0


def _make_new_beam():
    def fn(): return (NEG_INF, NEG_INF)
    return defaultdict(fn)


def _logsumexp(*args):
    """Stable log sum exp."""
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = np.max(args)
    lsp = np.log(np.sum(np.exp(a - a_max)
                        for a in args))
    return a_max + lsp


# def _logsumexp_chainer(a, xp, axis=None):
#     vmax = xp.amax(a, axis=axis, keepdims=True)
#     vmax += xp.log(xp.sum(xp.exp(a - vmax),
#                           axis=axis, keepdims=True, dtype=a.dtype))
#     return xp.squeeze(vmax, axis=axis)


class BeamSearchDecoder(object):
    """Beam search decoder.
    Arga:
        blank_index (int): the index of the blank label
        space_index (int, optional): the index of the space label
    """

    def __init__(self, blank_index, space_index=-1):
        self._blank = blank_index
        self._space = space_index

    def __call__(self, log_probs, inputs_seq_len, beam_width=1,
                 alpha=0., beta=0.):
        """Performs inference for the given output probabilities.
        Args:
            log_probs (np.ndarray): The output log-scale probabilities
                (e.g. post-softmax) for each time step.
                A tensor of size `[B, T, num_classes]`
            inputs_seq_len (np.ndarray): A tensor of size `[B]`
            beam_width (int): the size of beam
            alpha (float): language model weight
            beta (float): insertion bonus
        Returns:
            best_hyps (np.ndarray): Best path hypothesis (the output label sequence)
        """
        batch_size, _, num_classes = log_probs.shape
        best_hyps = []

        ##############################
        # Loop pver batch
        ##############################
        for i_batch in range(batch_size):
            # Elements in the beam are (prefix, (p_blank, p_no_blank))
            # Initialize the beam with the empty sequence, a probability of
            # 1 for ending in blank and zero for ending in non-blank
            # (in log space).
            beam = [(tuple(), (LOG_1, LOG_0))]
            time = inputs_seq_len[i_batch]

            ##############################
            # Loop over time
            ##############################
            for t in range(time):

                # A default dictionary to store the next step candidates.
                next_beam = _make_new_beam()

                ##############################
                # Loop over vocab
                ##############################
                for c in range(num_classes):
                    p_t = log_probs[i_batch, t, c]

                    # The variables p_b and p_nb are respectively the
                    # probabilities for the prefix given that it ends in a
                    # blank and does not end in a blank at this time step.
                    for prefix, (p_b, p_nb) in beam:  # Loop over beam

                        # If we propose a blank the prefix doesn't change.
                        # Only the probability of ending in blank gets updated.
                        if c == self._blank:
                            new_p_b, new_p_nb = next_beam[prefix]
                            new_p_b = _logsumexp(
                                new_p_b, p_b + p_t, p_nb + p_t)
                            next_beam[prefix] = (new_p_b, new_p_nb)
                            continue

                        # Extend the prefix by the new character c and it to the
                        # beam. Only the probability of not ending in blank gets
                        # updated.
                        prefix_end = prefix[-1] if prefix else None
                        new_prefix = prefix + (c,)
                        new_p_b, new_p_nb = next_beam[new_prefix]
                        if c != prefix_end:
                            new_p_nb = _logsumexp(
                                new_p_nb, p_b + p_t, p_nb + p_t)
                        else:
                            # We don't include the previous probability of not ending
                            # in blank (p_nb) if c is repeated at the end. The CTC
                            # algorithm merges characters not separated by a
                            # blank.
                            new_p_nb = _logsumexp(new_p_nb, p_b + p_t)

                        next_beam[new_prefix] = (new_p_b, new_p_nb)

                        # TODO: add LM score here

                        # If c is repeated at the end we also update the unchanged
                        # prefix. This is the merging case.
                        if c == prefix_end:
                            new_p_b, new_p_nb = next_beam[prefix]
                            new_p_nb = _logsumexp(new_p_nb, p_nb + p_t)
                            next_beam[prefix] = (new_p_b, new_p_nb)

                # Sort and trim the beam before moving on to the
                # next time-step.
                beam = sorted(next_beam.items(),
                              key=lambda x: _logsumexp(*x[1]),
                              reverse=True)
                beam = beam[:beam_width]

            best_hyp = beam[0][0]
            best_hyps.append(np.array(list(best_hyp)))

        return np.array(best_hyps)
