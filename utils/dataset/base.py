#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for all dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Base(object):

    def __init__(self, *args, **kwargs):

        self.epoch = 0
        self.iteration = 0
        self.is_new_epoch = False

        self.map_dict = {}
        if 'map_file_path' in kwargs.keys():
            # Read the mapping file
            with open(kwargs['map_file_path'], 'r') as f:
                for line in f:
                    line = line.strip().split()
                    self.map_dict[line[0]] = int(line[1])

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        return (self.input_list[index], self.label_list[index])

    def __iter__(self):
        """Returns self."""
        return self

    @property
    def sos_index(self):
        return self.map_dict['<']

    @property
    def eos_index(self):
        return self.map_dict['>']

    def next(self, batch_size=None):
        # For python2
        return self.__next__(batch_size)

    def reset(self):
        """Reset data counter. This is useful when you'd like to evaluate
        overall data during training.
        """
        self.rest = set(range(0, len(self), 1))

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration / len(self)

    def __next__(self):
        raise NotImplementedError
