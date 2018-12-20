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
    """Class for converting word sequence into indices.

    Args:
        dict_path (str): path to a dictionary file

    """

    def __init__(self, dict_path):
        # Load a dictionary file
        self.word2idx = {}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                w, id = line.strip().encode('utf_8').split(' ')
                self.word2idx[w] = int(id)

    def __call__(self, text):
        """Convert word sequence into indices.

        Args:
            text (str): word sequence
        Returns:
            token_ids (list): word indices

        """
        word_list = text.split(' ')
        token_ids = []
        for w in word_list:
            if w in self.word2idx.keys():
                token_ids.append(self.word2idx[w])
            else:
                # Replace with <unk>
                token_ids.append(self.word2idx['<unk>'])
        return token_ids


class Idx2word(object):
    """Class for converting indices into word sequence.

    Args:
        dict_path (str): path to a dictionary file

    """

    def __init__(self, dict_path):
        # Load a dictionary file
        self.idx2word = {}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                w, id = line.strip().encode('utf_8').split(' ')
                self.idx2word[int(id)] = w

    def __call__(self, token_ids, return_list=False):
        """Convert indices into word sequence.

        Args:
            token_ids (np.ndarray or list): word indices
            return_list (bool): if True, return list of words
        Returns:
            text (str): word sequence
                or
            word_list (list): list of words

        """
        word_list = list(map(lambda w: self.idx2word[w], token_ids))
        if return_list:
            return word_list
        return ' '.join(word_list)


class Char2word(object):
    """Class for converting character indices into the signle word index.

    Args:
        dict_path_word (str): path to a word dictionary file
        dict_path_char (str): path to a character dictionary file

    """

    def __init__(self, dict_path_word, dict_path_char):
        # Load a word dictionary file
        self.word2idx = {}
        with codecs.open(dict_path_word, 'r', 'utf-8') as f:
            for line in f:
                w, id = line.strip().encode('utf_8').split(' ')
                self.word2idx[w] = int(id)

        # Load a character dictionary file
        self.idx2char = {}
        with codecs.open(dict_path_char, 'r', 'utf-8') as f:
            for line in f:
                c, id = line.strip().encode('utf_8').split(' ')
                self.idx2char[int(id)] = c

    def __call__(self, char_ids):
        """Convert character indices into the single word index.

        Args:
            char_ids (np.ndarray or list): character indices corresponding to a single word
        Returns:
            word_id (int): a single word index

        """
        # char ids -> text
        str_single_word = ''.join(list(map(lambda i: self.idx2char[i], char_ids)))

        # text -> word id
        if str_single_word in self.word2idx.keys():
            word_id = self.word2idx[str_single_word]
        else:
            word_id = self.word2idx['<unk>']
        return word_id


class Word2char(object):
    """Class for converting a word index into the character indices.

    Args:
        dict_path_word (str): path to a dictionary file of words
        dict_path_char (str): path to a dictionary file of characters

    """

    def __init__(self, dict_path_word, dict_path_char):
        # Load a word dictionary file
        self.idx2word = {}
        with codecs.open(dict_path_word, 'r', 'utf-8') as f:
            for line in f:
                w, id = line.strip().encode('utf_8').split(' ')
                self.idx2word[int(id)] = w

        # Load a character dictionary file
        self.char2idx = {}
        with codecs.open(dict_path_char, 'r', 'utf-8') as f:
            for line in f:
                c, id = line.strip().encode('utf_8').split(' ')
                self.char2idx[c] = int(id)

    def __call__(self, word_id):
        """Convert a word index into character indices.

        Args:
            word_id (int): a single word index
        Returns:
            char_indices (list): character indices

        """
        # word id -> text
        str_single_word = self.idx2word[word_id]

        # text -> char ids
        char_indices = list(map(lambda c: self.char2idx[c], list(str_single_word)))
        return char_indices
