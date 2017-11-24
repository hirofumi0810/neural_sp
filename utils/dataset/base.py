#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for all dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from struct import unpack


class Base(object):

    def __init__(self, *args, **kwargs):
        self.epoch = 0
        self.iteration = 0
        self.is_new_epoch = False
        self.offset = 0

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
        feature = self.load_npy(self.df['input_path'][index])
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
        # Pad by -1
        return None if self.is_test else -1

    @property
    def att_padded_value(self):
        # Pad by <SOS>
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
        """Reset data counter and offset."""
        self.rest = set(range(0, len(self), 1))
        self.offset = 0

    def split_per_device(self, x, num_gpus):
        if num_gpus > 1:
            return np.array_split(x, num_gpus, axis=0)
        else:
            return x[np.newaxis]

    def load_npy(self, path):
        """Load npy files.
        Args:
            path (string):
        Returns:
            input_data (np.ndarray): A tensor of size (frame_num, feature_dim)
        """
        return np.load(path)

    def load_htk(htk_path):
        """Load each HTK file.
        Args:
            htk_path (string): path to a HTK file
        Returns:
            input_data (np.ndarray): A tensor of size (frame_num, feature_dim)
        """
        # print('...Reading: %s' % htk_path)
        with open(htk_path, "rb") as f:
            # Read header
            spam = f.read(12)
            frame_num, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)

            # for debug
            # print(frame_num)  # frame num
            # print(sampPeriod)  # 10ms
            # print(sampSize)  # feature dim * 4 (byte)
            # print(parmKind)

            # Read data
            feature_dim = int(sampSize / 4)
            f.seek(12, 0)
            input_data = np.fromfile(f, 'f')
            input_data = input_data.reshape(-1, feature_dim)
            input_data.byteswap(True)

        return input_data
