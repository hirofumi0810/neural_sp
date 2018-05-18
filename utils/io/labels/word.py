#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import codecs


class Word2idx(object):
    """Convert word to index.
    Args:
        vocab_file_path (string): path to the vocablary file
        space_mark (string, optional): the space mark to divide a sequence into words
    """

    def __init__(self, vocab_file_path, space_mark='_'):
        self.space_mark = space_mark

        # Load the vocablary file
        self.map_dict = {}
        vocab_count = 0
        with codecs.open(vocab_file_path, 'r', 'utf-8') as f:
            for line in f:
                w = line.strip()
                self.map_dict[w] = vocab_count
                vocab_count += 1

        # Add <EOS>
        self.map_dict['>'] = vocab_count

    def __call__(self, str_word):
        """Convert word into index.
        Args:
            str_word (string): a sequence of words
        Returns:
            indices (np.ndarray): word indices
        """
        word_list = str_word.split(self.space_mark)
        indices = []

        # Convert word strings into the corresponding indices
        for w in word_list:
            if w in self.map_dict.keys():
                indices.append(self.map_dict[w])
            else:
                # Replace with <UNK>
                indices.append(self.map_dict['OOV'])

        return np.array(indices)


class Idx2word(object):
    """Convert index into word.
    Args:
        vocab_file_path (string): path to the vocabulary file
        space_mark (string, optional): the space mark to divide a sequence into words
        return_list (bool, optional): if True, return list of words
    """

    def __init__(self, vocab_file_path, space_mark='_', return_list=False):
        self.space_mark = space_mark
        self.return_list = return_list

        # Load the vocabulary file
        self.map_dict = {}
        vocab_count = 0
        with codecs.open(vocab_file_path, 'r', 'utf-8') as f:
            for line in f:
                w = line.strip()
                self.map_dict[vocab_count] = w
                vocab_count += 1

        # Add <EOS>
        self.map_dict[vocab_count] = '>'

    def __call__(self, indices):
        """
        Args:
            indices (np.ndarray): list of word indices.
        Returns:
            str_word (string): a sequence of words
                or
            word_list (list): list of words
        """
        # Convert word indices into the corresponding strings
        word_list = list(map(lambda w: self.map_dict[w], indices))

        if self.return_list:
            return word_list

        str_word = self.space_mark.join(word_list)

        return str_word


class Char2word(object):
    """Convert character indices into the word index.
    Args:
        vocab_file_path_word (string): path to the vocabulary file of words
        vocab_file_path_char (string): path to the vocabulary file of characters
    """

    def __init__(self, vocab_file_path_word, vocab_file_path_char):

        # Load the vocabulary file (word)
        self.map_dict_w = {}
        vocab_count_w = 0
        with codecs.open(vocab_file_path_word, 'r', 'utf-8') as f:
            for line in f:
                w = line.strip()
                # string -> index
                self.map_dict_w[w] = vocab_count_w
                vocab_count_w += 1

        # Load the vocabulary file
        self.map_dict_c = {}
        vocab_count_c = 0
        with codecs.open(vocab_file_path_char, 'r', 'utf-8') as f:
            for line in f:
                c = line.strip()
                # index -> string
                self.map_dict_c[vocab_count_c] = c
                vocab_count_c += 1

        # Add <EOS>
        self.map_dict_w['>'] = vocab_count_w
        self.map_dict_c[vocab_count_c] = '>'

    def __call__(self, char_indices):
        """
        Args:
            char_indices (np.ndarray): list of character indices.
        Returns:
            str_word (string): a sequence of words
                or
            word_list (list): list of words
        """
        # Convert character indices into the corresponding character strings
        str_single_word = ''.join(
            list(map(lambda c: self.map_dict_c[c], char_indices)))

        # Convert a word string into the corresponding index
        if str_single_word in self.map_dict_w.keys():
            word_index = self.map_dict_w[str_single_word]
        else:
            word_index = self.map_dict_w['OOV']
        return word_index


class Word2char(object):
    """Convert a word index into character indices.
    Args:
        vocab_file_path_word (string): path to the vocabulary file of words
        vocab_file_path_char (string): path to the vocabulary file of characters
    """

    def __init__(self, vocab_file_path_word, vocab_file_path_char):

        # Load the vocabulary file (word)
        self.map_dict_w = {}
        vocab_count_w = 0
        with codecs.open(vocab_file_path_word, 'r', 'utf-8') as f:
            for line in f:
                w = line.strip()
                # index -> string
                self.map_dict_w[vocab_count_w] = w
                vocab_count_w += 1

        # Load the vocabulary file
        self.map_dict_c = {}
        vocab_count_c = 0
        with codecs.open(vocab_file_path_char, 'r', 'utf-8') as f:
            for line in f:
                c = line.strip()
                # string -> index
                self.map_dict_c[c] = vocab_count_c
                vocab_count_c += 1

        # Add <EOS>
        self.map_dict_w[vocab_count_w] = '>'
        self.map_dict_c['>'] = vocab_count_c

    def __call__(self, word_indices):
        """
        Args:
            indices (np.ndarray): list of word indices.
        Returns:
            str_word (string): a sequence of words
                or
            word_list (list): list of words
        """
        # Convert word indices into the the corresponding character strings
        word_list = list(map(lambda w: self.map_dict_w[w], word_indices))
        str_char = self.space_mark.join(word_list)

        # Convert character strings into the corresponding indices
        char_indices = list(map(lambda c: self.map_dict_c[c], list(str_char)))
        return char_indices
