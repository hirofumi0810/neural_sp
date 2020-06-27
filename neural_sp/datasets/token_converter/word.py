#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Word-level token <-> index converter."""

import codecs


class Word2idx(object):
    """Class for converting word sequence into indices.

    Args:
        dict_path (str): path to a dictionary file
        word_char_mix (bool):

    """

    def __init__(self, dict_path, word_char_mix=False):
        self.word_char_mix = word_char_mix

        # Load a dictionary file
        self.token2idx = {'<blank>': 0}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                w, idx = line.strip().split(' ')
                self.token2idx[w] = int(idx)
        self.vocab = len(self.token2idx.keys())

    def __call__(self, text):
        """Convert word sequence into indices.

        Args:
            text (str): word sequence
        Returns:
            token_ids (list): word indices

        """
        token_ids = []
        words = text.split(' ')
        for w in words:
            if w in self.token2idx.keys():
                token_ids.append(self.token2idx[w])
            else:
                # Replace with <unk>
                if self.word_char_mix:
                    for c in list(w):
                        if c in self.token2idx.keys():
                            token_ids.append(self.token2idx[c])
                        else:
                            token_ids.append(self.token2idx['<unk>'])
                else:
                    token_ids.append(self.token2idx['<unk>'])
        return token_ids


class Idx2word(object):
    """Class for converting indices into word sequence.

    Args:
        dict_path (str): path to a dictionary file

    """

    def __init__(self, dict_path):
        # Load a dictionary file
        self.idx2token = {0: '<blank>'}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                w, idx = line.strip().split(' ')
                self.idx2token[int(idx)] = w
        self.vocab = len(self.idx2token.keys())
        # for synchronous bidirectional attention
        self.idx2token[self.vocab] = '<l2r>'
        self.idx2token[self.vocab + 1] = '<r2l>'
        self.idx2token[self.vocab + 2] = '<null>'

    def __call__(self, token_ids, return_list=False):
        """Convert indices into word sequence.

        Args:
            token_ids (np.ndarray or list): word indices
            return_list (bool): if True, return list of words
        Returns:
            text (str): word sequence
                or
            words (list): list of words

        """
        words = list(map(lambda w: self.idx2token[w], token_ids))
        if return_list:
            return words
        return ' '.join(words)


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
                w, idx = line.strip().split(' ')
                self.word2idx[w] = int(idx)

        # Load a character dictionary file
        self.idx2char = {}
        with codecs.open(dict_path_char, 'r', 'utf-8') as f:
            for line in f:
                c, idx = line.strip().split(' ')
                self.idx2char[int(idx)] = c

    def __call__(self, char_ids):
        """Convert character indices into the single word index.

        Args:
            char_ids (np.ndarray or list): character indices corresponding to a single word
        Returns:
            word_idx (int): a single word index

        """
        # char ids -> text
        single_word = ''.join(list(map(lambda i: self.idx2char[i], char_ids)))

        # text -> word idx
        if single_word in self.word2idx.keys():
            word_idx = self.word2idx[single_word]
        else:
            word_idx = self.word2idx['<unk>']
        return word_idx


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
                w, idx = line.strip().split(' ')
                self.idx2word[int(idx)] = w

        # Load a character dictionary file
        self.char2idx = {}
        with codecs.open(dict_path_char, 'r', 'utf-8') as f:
            for line in f:
                c, idx = line.strip().split(' ')
                self.char2idx[c] = int(idx)

    def __call__(self, word_idx):
        """Convert a word index into character indices.

        Args:
            word_idx (int): a single word index
        Returns:
            char_indices (list): character indices

        """
        # word idx -> text
        single_word = self.idx2word[word_idx]

        # text -> char ids
        char_indices = list(map(lambda c: self.char2idx[c], list(single_word)))
        return char_indices
