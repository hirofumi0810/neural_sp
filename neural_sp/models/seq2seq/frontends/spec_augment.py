#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""SpecAugment data augmentation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class SpecAugment(object):
    """SpecAugment class.

    Args:
        F (int): parameter for frequency masking
        T (int): parameter for time masking
        n_freq_masks (int): number of frequency masks
        n_time_masks (int): number of time masks
        W (int): parameter for time warping
        p (float): parameter for upperbound of the time mask
        adaptive_number_ratio (float): adaptive multiplicity ratio for time masking
        adaptive_size_ratio (float): adaptive size ratio for time masking
        max_n_time_masks (int): maximum number of time masking

    """

    def __init__(self, F, T, n_freq_masks, n_time_masks, p=1.0, W=40,
                 adaptive_number_ratio=0, adaptive_size_ratio=0,
                 max_n_time_masks=20):

        super(SpecAugment, self).__init__()

        self.W = W
        self.F = F
        self.T = T
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p

        # adaptive SpecAugment
        self.adaptive_number_ratio = adaptive_number_ratio
        self.adaptive_size_ratio = adaptive_size_ratio
        self.max_n_time_masks = max_n_time_masks

        self._freq_mask = None
        self._time_mask = None

    def librispeech_basic(self):
        self.W = 80
        self.F = 27
        self.T = 100
        self.n_freq_masks = 1
        self.n_time_masks = 1
        self.p = 1.0

    def librispeech_double(self):
        self.W = 80
        self.F = 27
        self.T = 100
        self.n_freq_masks = 2
        self.n_time_masks = 2
        self.p = 1.0

    def switchboard_mild(self):
        self.W = 40
        self.F = 15
        self.T = 70
        self.n_freq_masks = 2
        self.n_time_masks = 2
        self.p = 0.2

    def switchboard_strong(self):
        self.W = 40
        self.F = 27
        self.T = 70
        self.n_freq_masks = 2
        self.n_time_masks = 2
        self.p = 0.2

    @property
    def freq_mask(self):
        return self._freq_mask

    @property
    def time_mask(self):
        return self._time_mask

    def __call__(self, xs):
        """
        Args:
            xs (FloatTensor): `[B, T, F]`
        Returns:
            xs (FloatTensor): `[B, T, F]`

        """
        # xs = self.time_warp(xs)
        xs = self.mask_freq(xs)
        xs = self.mask_time(xs)
        return xs

    def time_warp(xs, W=40):
        raise NotImplementedError

    def mask_freq(self, xs, replace_with_zero=False):
        n_bins = xs.size(-1)
        for i in range(0, self.n_freq_masks):
            f = int(np.random.uniform(low=0, high=self.F))
            f_0 = int(np.random.uniform(low=0, high=n_bins - f))
            xs[:, :, f_0:f_0 + f] = 0
            assert f_0 <= f_0 + f
            self._freq_mask = (f_0, f_0 + f)
        return xs

    def mask_time(self, xs, replace_with_zero=False):
        n_frames = xs.size(1)
        if self.adaptive_number_ratio > 0:
            n_masks = int(n_frames * self.adaptive_number_ratio)
            n_masks = min(n_masks, self.max_n_time_masks)
        else:
            n_masks = self.n_time_masks
        if self.adaptive_size_ratio > 0:
            T = self.adaptive_size_ratio * n_frames
        else:
            T = self.T
        for i in range(n_masks):
            t = int(np.random.uniform(low=0, high=T))
            t = min(t, int(n_frames * self.p))
            t_0 = int(np.random.uniform(low=0, high=n_frames - t))
            xs[:, t_0:t_0 + t] = 0
            assert t_0 <= t_0 + t
            self._time_mask = (t_0, t_0 + t)
        return xs
