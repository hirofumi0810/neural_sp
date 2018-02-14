#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Beam search (prefix search) decoder in numpy implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

LOG_0 = -float("inf")
LOG_1 = 0


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
            best_hyps (np.ndarray): Best path hypothesis.
                A tensor of size `[B, labels_max_seq_len]`
        """
        batch_size, _, num_classes = log_probs.shape
        best_hyps = []

        for i_batch in range(batch_size):
            # Initialize the beam with the empty sequence, a probability of
            # 1 for ending in blank and zero for ending in non-blank
            # (in log space).
            beam_prefix = [[-1]]
            beam_p_b = [LOG_1]
            beam_p_nb = [LOG_0]

            for t in range(inputs_seq_len[i_batch]):
                new_beam_prefix = []
                new_beam_p = []
                new_beam_p_b = []
                new_beam_p_nb = []

                for prefix, p_b, p_nb in zip(beam_prefix, beam_p_b, beam_p_nb):
                    # The variables p_b and p_nb are respectively the
                    # probabilities for the prefix given that it ends in a
                    # blank and does not end in a blank at this time step.

                    for c in range(num_classes):
                        p_t = log_probs[i_batch, t, c]

                        # If we propose a blank the prefix doesn't change.
                        # Only the probability of ending in blank gets updated.
                        if c == self._blank:
                            new_prefix = prefix
                            new_p_b = np.logaddexp(p_b + p_t, p_nb + p_t)
                            new_p_nb = p_nb
                            new_p = np.logaddexp(new_p_b, p_nb)
                            new_beam_prefix.append(new_prefix)
                            new_beam_p.append(new_p)
                            new_beam_p_b.append(new_p_b)
                            new_beam_p_nb.append(new_p_nb)
                            continue

                        # Extend the prefix by the new character c and it to the
                        # beam. Only the probability of not ending in blank gets
                        # updated.
                        prefix_end = prefix[-1]
                        new_prefix = prefix + [c]
                        new_p_b = p_b
                        if c != prefix_end:
                            new_p_nb = np.logaddexp(p_b + p_t, p_nb + p_t)
                        else:
                            # We don't include the previous probability of not ending
                            # in blank (p_nb) if c is repeated at the end. The CTC
                            # algorithm merges characters not separated by a
                            # blank.
                            new_p_nb = p_t + p_b
                            # NOTE: 間にblank挟まってる場合

                        new_p = np.logaddexp(p_b, new_p_nb)
                        new_beam_prefix.append(new_prefix)
                        new_beam_p.append(new_p)
                        new_beam_p_b.append(new_p_b)
                        new_beam_p_nb.append(new_p_nb)

                        # TODO: add LM score here

                        # If c is repeated at the end we also update the unchanged
                        # prefix. This is the merging case.
                        if c == prefix_end:
                            new_p_b = p_b
                            new_p_nb = p_t + p_b
                            # NOTE: 間にblank挟まってない場合

                            new_p = np.logaddexp(p_b, new_p_nb)
                            new_beam_prefix.append(prefix)
                            new_beam_p.append(new_p)
                            new_beam_p_b.append(new_p_b)
                            new_beam_p_nb.append(new_p_nb)

                # Sort and trim the beam before moving on to the
                # next time-step.
                perm_indcies = np.argsort(new_beam_p)[::-1]
                # NOTE: np.argsort is in the descending order

                beam_prefix = np.array(new_beam_prefix)[
                    perm_indcies][:beam_width].tolist()
                beam_p_b = np.array(new_beam_p_b)[
                    perm_indcies][:beam_width].tolist()
                beam_p_nb = np.array(new_beam_p_nb)[
                    perm_indcies][:beam_width].tolist()

            best_hyp = beam_prefix[0][1:]
            # NOTE: remove the first -1
            best_hyps.append(np.array(best_hyp))

        return np.array(best_hyps)
