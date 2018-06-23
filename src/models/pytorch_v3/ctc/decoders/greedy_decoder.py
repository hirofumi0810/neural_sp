#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Greedy (best pass) decoder in numpy implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from itertools import groupby


class GreedyDecoder(object):

    def __init__(self, blank_index):
        self._blank = blank_index

    def __call__(self, logits, x_lens):
        """
        Args:
            logits (np.ndarray): A tensor of size `[B, T, num_classes]`
            x_lens (np.ndarray): A tensor of size `[B]`
        Returns:
            best_hyps (np.ndarray): Best path hypothesis.
                A tensor of size `[B, labels_max_seq_len]`
        """
        batch_size = logits.shape[0]
        best_hyps = []

        # Pickup argmax class
        for b in range(batch_size):
            indices = []
            time = x_lens[b]
            for t in range(time):
                argmax = np.argmax(logits[b, t], axis=0)
                indices.append(argmax)

            # Step 1. Collapse repeated labels
            collapsed_indices = [x[0] for x in groupby(indices)]

            # Step 2. Remove all blank labels
            best_hyp = [x for x in filter(
                lambda x: x != self._blank, collapsed_indices)]
            best_hyps.append(np.array(best_hyp))

        return np.array(best_hyps)
