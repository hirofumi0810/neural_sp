#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Word2idx(object):

    def __init__(self, vocab_file_path):
        pass

    def __call__(self):
        pass


class Idx2word(object):
    """Convert from index to word.
    Args:
        vocab_file_path (string): path to the vocabulary file
    """

    def __init__(self, vocab_file_path):
        # Read the vocabulary file
        self.map_dict = {}
        vocab_count = 0
        with open(vocab_file_path, 'r') as f:
            for line in f:
                word = line.strip()
                self.map_dict[vocab_count] = word
                vocab_count += 1

        # Add <SOS> & <EOS>
        self.map_dict[vocab_count] = '<'
        self.map_dict[vocab_count + 1] = '>'

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
