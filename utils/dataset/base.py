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

        # NOTE: these are needed when using attention-based models
        self.num_vocab = 0
        if 'vocab_file_path' in kwargs.keys():
            # Read the vocabulary file
            with open(kwargs['vocab_file_path'], 'r') as f:
                for line in f:
                    line = line.strip().split()
                    if line != '':
                        self.num_vocab += 1

        self.num_vocab_sub = 0
        if 'vocab_file_path_sub' in kwargs.keys():
            # Read the vocabulary file
            with open(kwargs['vocab_file_path_sub'], 'r') as f:
                for line in f:
                    line = line.strip().split()
                    if line != '':
                        self.num_vocab_sub += 1

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        return (self.input_list[index], self.label_list[index])

    def __iter__(self):
        """Returns self."""
        return self

    @property
    def sos_index(self):
        return self.num_vocab

    @property
    def sos_index_sub(self):
        return self.num_vocab_sub

    @property
    def eos_index(self):
        return self.num_vocab + 1

    @property
    def eos_index_sub(self):
        return self.num_vocab_sub + 1

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
