#! /usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Idx2word(object):
    """Convert from index to word.
    Args:
        map_file_path (string): path to the mapping file
    """

    def __init__(self, map_file_path):
        # Read the mapping file
        self.map_dict = {}
        with open(map_file_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                self.map_dict[int(line[1])] = line[0]

    def __call__(self, index_list, padded_value=-1):
        """
        Args:
            index_list (np.ndarray): list of word indices.
                Batch size 1 is expected.
            padded_value (int): the value used for padding
        Returns:
            word_list (list): list of words
        """
        # Remove padded values
        assert type(
            index_list) == np.ndarray, 'index_list should be np.ndarray.'
        index_list = np.delete(index_list, np.where(index_list == -1), axis=0)

        # Convert from indices to the corresponding words
        word_list = list(map(lambda x: self.map_dict[x], index_list))

        return word_list


class Word2idx(object):

    def __init__(self, map_file_path):
        pass

    def __call__(self):
        pass
