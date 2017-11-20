#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Word2idx(object):
    """Convert from word to index.
    Args:
        vocab_file_path (string): path to the vocablary file
        space_mark (string, optional): the space mark to divide a sequence into words
    """

    def __init__(self, vocab_file_path, space_mark='_'):
        self.space_mark = space_mark

        # Read the vocablary file
        self.map_dict = {}
        vocab_count = 0
        with open(vocab_file_path, 'r') as f:
            for line in f:
                word = line.strip()
                self.map_dict[word] = vocab_count
                vocab_count += 1

        # Add <SOS> & <EOS>
        self.map_dict['<'] = vocab_count
        self.map_dict['>'] = vocab_count + 1

    def __call__(self, str_word):
        """Convert from word to index.
        Args:
            str_word (string): a sequence of words
        Returns:
            index_list (np.ndarray): word indices
        """
        word_list = str_word.split(self.space_mark)
        index_list = []

        # Convert from word to index
        for word in word_list:
            if word in self.map_dict.keys():
                index_list.append(self.map_dict[word])
            else:
                # Replace with <UNK>
                index_list.append(self.map_dict['OOV'])

        return np.array(index_list)


class Idx2word(object):
    """Convert from index to word.
    Args:
        vocab_file_path (string): path to the vocabulary file
        space_mark (string, optional): the space mark to divide a sequence into words
    """

    def __init__(self, vocab_file_path, space_mark='_'):
        self.space_mark = space_mark

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

    def __call__(self, index_list):
        """
        Args:
            index_list (np.ndarray): list of word indices.
        Returns:
            str_word (string): a sequence of words
        """
        # Convert from indices to the corresponding words
        word_list = list(map(lambda x: self.map_dict[x], index_list))
        str_word = self.space_mark.join(word_list)

        return str_word
