#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for all dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Base(object):

    def __init__(self, *args, **kwargs):
        self.epoch = 0
        self.iteration = 0
        self.is_new_epoch = False

        # Read the vocabulary file
        vocab_count = 0
        with open(kwargs['vocab_file_path'], 'r') as f:
            for line in f:
                vocab_count += 1
        self.num_classes = vocab_count

        if 'vocab_file_path_sub' in kwargs.keys():
            vocab_count_sub = 0
            with open(kwargs['vocab_file_path_sub'], 'r') as f:
                for line in f:
                    vocab_count_sub += 1
            self.num_classes_sub = vocab_count_sub

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        feature = self._load_npy([self.df['input_path'][index]])
        transcript = self.df['transcript'][index]
        return (feature, transcript)

    def __iter__(self):
        """Returns self."""
        return self

    @property
    def sos_index(self):
        return self.num_classes

    @property
    def sos_index_sub(self):
        return self.num_classes_sub

    @property
    def eos_index(self):
        return self.num_classes + 1

    @property
    def eos_index_sub(self):
        return self.num_classes_sub + 1

    @property
    def ctc_padded_value(self):
        return None if self.is_test else -1

    @property
    def att_padded_value(self):
        return None if self.is_test else self.sos_index

    @property
    def epoch_detail(self):
        # Floating point version of epoch
        return self.iteration / len(self)

    def __next__(self):
        raise NotImplementedError

    def next(self, batch_size=None):
        # For python2
        return self.__next__(batch_size)

    def reset(self):
        """Reset data counter."""
        self.rest = set(range(0, len(self), 1))

    def _load_npy(self, paths):
        """Load npy files."""
        return np.array(list(map(lambda path: np.load(path), paths)))

    def split_per_device(self, x, num_gpus):
        if num_gpus > 1:
            return np.array_split(x, num_gpus, axis=0)
        else:
            return x[np.newaxis]
