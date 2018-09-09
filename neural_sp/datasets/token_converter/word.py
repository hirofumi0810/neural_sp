#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs


class Word2idx(object):
    """Convert word string into tokenid.

    Args:
        dict_path (str): path to the dictionary file

    """

    def __init__(self, dict_path):
        # Load the dictionary file
        self.word2idx = {}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                w, id = line.strip().encode('utf_8').split(' ')
                self.word2idx[w] = int(id)

    def __call__(self, text):
        """Convert word string into tokenid.

        Args:
            text (str):
        Returns:
            tokenids (list): word tokenids

        """
        word_list = text.split(' ')
        tokenids = []
        for w in word_list:
            if w in self.word2idx.keys():
                tokenids.append(self.word2idx[w])
            else:
                # Replace with <unk>
                tokenids.append(self.word2idx['<unk>'])
        return tokenids


class Idx2word(object):
    """Convert tokenid into word string.

    Args:
        dict_path (str): path to the dictionary file

    """

    def __init__(self, dict_path):
        # Load the dictionary file
        self.idx2word = {}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                w, id = line.strip().encode('utf_8').split(' ')
                self.idx2word[int(id)] = w

    def __call__(self, tokenids, return_list=False):
        """Convert tokenid into word string.

        Args:
            tokenids (np.ndarray or list): list of word tokenids
            return_list (bool): if True, return list of words
        Returns:
            text (str):
                or
            word_list (list): list of words

        """
        # Convert word tokenids into the corresponding strings
        word_list = list(map(lambda w: self.idx2word[w], tokenids))
        if return_list:
            return word_list
        return ' '.join(word_list)


class Char2word(object):
    """Convert character-level tokenid into the word-level tokenid.

    Args:
        dict_path_word (str): path to the dictionary file of words
        dict_path_char (str): path to the dictionary file of characters

    """

    def __init__(self, dict_path_word, dict_path_char):
        # Load the dictionary file (word)
        self.word2idx = {}
        with codecs.open(dict_path_word, 'r', 'utf-8') as f:
            for line in f:
                w, id = line.strip().encode('utf_8').split(' ')
                # string -> tokenid
                self.word2idx[w] = int(id)

        # Load the dictionary file
        self.idx2char = {}
        with codecs.open(dict_path_char, 'r', 'utf-8') as f:
            for line in f:
                c, id = line.strip().encode('utf_8').split(' ')
                # tokenid -> string
                self.idx2char[int(id)] = c

    def __call__(self, char_indices):
        """Convert character-level tokenid into the word-level tokenid.

        Args:
            char_indices (np.ndarray): list of character tokenid
        Returns:
            word_index (int): a tokenid of the corresponding word

        """
        # Convert character tokenid into the corresponding character strings
        str_single_word = ''.join(list(map(lambda i: self.idx2char[i], char_indices)))

        # Convert word strings into the corresponding word tokenid
        if str_single_word in self.word2idx.keys():
            word_index = self.word2idx[str_single_word]
        else:
            word_index = self.word2idx['<unk>']
        return word_index


class Word2char(object):
    """Convert a word-level tokenid into character-level tokenid.

    Args:
        dict_path_word (str): path to the dictionary file of words
        dict_path_char (str): path to the dictionary file of characters

    """

    def __init__(self, dict_path_word, dict_path_char):
        # Load the dictionary file (word)
        self.idx2word = {}
        with codecs.open(dict_path_word, 'r', 'utf-8') as f:
            for line in f:
                w, id = line.strip().encode('utf_8').split(' ')
                # tokenid -> string
                self.idx2word[int(id)] = w

        # Load the dictionary file
        self.char2idx = {}
        with codecs.open(dict_path_char, 'r', 'utf-8') as f:
            for line in f:
                c, id = line.strip().encode('utf_8').split(' ')
                # string -> tokenid
                self.char2idx[c] = int(id)

    def __call__(self, word_index):
        """Convert a word-level tokenid into character-level tokenid.

        Args:
            word_index (int): tokenid of a single word
        Returns:
            char_indices (list): tokenids of the corresponding characters

        """
        # Convert word tokenid into the the corresponding character strings
        str_single_word = self.idx2word[word_index]

        # Convert character strings into the corresponding character tokenids
        char_indices = list(map(lambda c: self.char2idx[c], list(str_single_word)))
        return char_indices
