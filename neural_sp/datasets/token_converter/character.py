#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Character-level token <-> index converter."""

import codecs
import os


class Char2idx(object):
    """Class for converting character sequence into indices.

    Args:
        dict_path (str): path to a dictionary file
        remove_list (list): characters to ignore

    """

    def __init__(self, dict_path, nlsyms=False, remove_space=False, remove_list=[]):
        self.remove_space = remove_space
        self.remove_list = remove_list

        # Load a dictionary file
        self.token2idx = {'<blank>': 0}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                c, idx = line.strip().split(' ')
                if c in remove_list:
                    continue
                self.token2idx[c] = int(idx)
        self.vocab = len(self.token2idx.keys())

        self.nlsyms_list = []
        if nlsyms and os.path.isfile(nlsyms):
            with codecs.open(nlsyms, 'r', 'utf-8') as f:
                for line in f:
                    self.nlsyms_list.append(line.strip())

    def __call__(self, text):
        """Convert character sequence into indices.

        Args:
            text (str): character sequence
        Returns:
            token_ids (list): character indices

        """
        token_ids = []
        words = text.replace(' ', '<space>').split('<space>')
        for i, w in enumerate(words):
            if w in self.nlsyms_list:
                token_ids.append(self.token2idx[w])
            else:
                for c in list(w):
                    if c in self.token2idx.keys():
                        token_ids.append(self.token2idx[c])
                    else:
                        # Replace with <unk>
                        token_ids.append(self.token2idx['<unk>'])
                        # NOTE: OOV handling is prepared for Japanese and Chinese

            if not self.remove_space:
                if i < len(words) - 1:
                    token_ids.append(self.token2idx['<space>'])
        return token_ids


class Idx2char(object):
    """Class for converting indices into character sequence.

    Args:
        dict_path (str): path to a dictionary file
        remove_list (list): characters to ignore

    """

    def __init__(self, dict_path, remove_list=[]):
        self.remove_list = remove_list

        # Load a dictionary file
        self.idx2token = {0: '<blank>'}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                c, idx = line.strip().split(' ')
                if c in remove_list:
                    continue
                self.idx2token[int(idx)] = c
        self.vocab = len(self.idx2token.keys())
        # for synchronous bidirectional attention
        self.idx2token[self.vocab] = '<l2r>'
        self.idx2token[self.vocab + 1] = '<r2l>'
        self.idx2token[self.vocab + 2] = '<null>'

    def __call__(self, token_ids, return_list=False):
        """Convert indices into character sequence.

        Args:
            token_ids (np.ndarray or list): character indices
            return_list (bool): if True, return list of characters
        Returns:
            text (str): character sequence
                or
            characters (list): list of characters

        """
        characters = list(map(lambda c: self.idx2token[c], token_ids))
        if return_list:
            return characters
        return ''.join(characters).replace('<space>', ' ')
