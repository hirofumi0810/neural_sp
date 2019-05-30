#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Sequence summayr network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random


class SpecAugment(object):
    """SpecAugment calss.

    Args:
        W (int): parameter for time warping
        F (int): parameter for frequency masking
        T (int): parameter for time masking
        n_freq_masks (int): number of frequency masks
        n_time_masks (int): number of time masks
        p (float): parameter for upperbound of the time mask

    """

    def __init__(self,
                 W=40,
                 F=27,
                 T=70,
                 n_freq_masks=2,
                 n_time_masks=2,
                 p=0.2):

        super(SpecAugment, self).__init__()

        self.W = W
        self.F = F
        self.T = T
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p

    @staticmethod
    def librispeech_basic(self):
        raise NotImplementedError

    @staticmethod
    def librispeech_double(self):
        raise NotImplementedError

    @staticmethod
    def switchboard_mild(self):
        raise NotImplementedError

    @staticmethod
    def switchboard_strong(self):
        raise NotImplementedError

    def __call__(self, xs):
        """
        Args:
            xs (FloatTensor): `[B, T, F]`
        Returns:
            xs (FloatTensor): `[B, T, F]`

        """
        # xs = self.time_warp(xs)
        xs = self.freq_mask(xs)
        xs = self.time_mask(xs)
        return xs

    def time_warp(xs, W=40):
        raise NotImplementedError

    def freq_mask(self, xs, replace_with_zero=False):
        n_bins = xs.size(-1)

        for i in range(0, self.n_freq_masks):
            f = np.random.uniform(low=0, high=self.F)
            f = int(f)
            f_0 = random.randint(0, n_bins - f)
            xs[:, :, f_0:f_0 + f] = 0

        return xs

    def time_mask(self, xs, replace_with_zero=False):
        n_frames = xs.size(1)

        for i in range(self.n_time_masks):
            t = np.random.uniform(low=0, high=self.T)
            t = min(int(t), int(n_frames * self.p))
            t0 = random.randint(0, n_frames - t)
            xs[:, t0:t0 + t] = 0

        return xs
