#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""CTC decoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from chainer import Variable
from chainer import functions as F


class GreedyDecoder(object):

    """The collapsing function which maps frame-wise indices into label
    sequence. There are 2 steps.
        1. Remove the successive labels except blank labels.
        2. Remove all blank labels.
    Args:
        indices ():
    Returns:
        labels ():
    """

    def __init__(self, blank_index):
        self.blank_index = blank_index

    def __call__(self, logits):
        """Best path (greedy) decoder.
        Args:
            logits (list): list of tensors of size `[T, num_classes]`
        Returns:
            labels (list): list of sequences of class indices,
        """
        batch_size = len(logits)
        max_time, num_classes = logits[0].shape
        framewise_indices = F.argmax(logits[0], axis=1).data

        seq_indices_batch = []
        for i_batch in range(batch_size):
            seq_indices = []
            pre_index = None
            for t in range(max_time):
                if t != 0 and framewise_indices[t] == pre_index:
                    continue
                elif framewise_indices[t] == self.blank_index:
                    pre_index = None
                    continue
                else:
                    pre_index = framewise_indices[t]
                seq_indices.append(framewise_indices[t])

            # Remove blank labels
            # while self.blank_index in seq_indices:
            #     seq_indices.remove(self.blank_index)

            seq_indices_batch.append(seq_indices)

        return seq_indices_batch
