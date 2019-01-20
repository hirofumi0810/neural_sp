#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Greedy (best pass) decoder in numpy implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import groupby
import numpy as np


class GreedyDecoder(object):

    def __init__(self, blank):
        self.blank = blank

    def __call__(self, log_probs, xlens):
        """

        Args:
            log_probs (torch.FloatTensor): `[B, T, vocab]`
            xlens (np.ndarray): `[B]`
        Returns:
            best_hyps (np.ndarray): Best path hypothesis. `[B, labels_max_seq_len]`

        """
        bs = log_probs.size(0)
        best_hyps = []

        # Pickup argmax class
        for b in range(bs):
            indices = []
            time = xlens[b]
            for t in range(time):
                argmax = np.argmax(log_probs[b, t], axis=0).item()
                indices.append(argmax)

            # Step 1. Collapse repeated labels
            collapsed_indices = [x[0] for x in groupby(indices)]

            # Step 2. Remove all blank labels
            best_hyp = [x for x in filter(
                lambda x: x != self.blank, collapsed_indices)]
            best_hyps.append(np.array(best_hyp))

        return np.array(best_hyps)
