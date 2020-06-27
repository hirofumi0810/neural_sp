#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Phone-level token <-> index converter."""

import codecs


class Phone2idx(object):
    """Class for converting phone sequence to indices.

    Args:
        dict_path (str): path to a dictionary file
        remove_list (list): phones to ingore

    """

    def __init__(self, dict_path, remove_list=[]):
        # Load a dictionary file
        self.token2idx = {'<blank>': 0}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                p, idx = line.strip().split(' ')
                if p in remove_list:
                    continue
                self.token2idx[p] = int(idx)
        self.vocab = len(self.token2idx.keys())

    def __call__(self, text):
        """Convert phone sequence to indices.

        Args:
            text (str): phone sequence divided by spaces
        Returns:
            token_ids (list): phone indices

        """
        phones = text.split(' ')
        token_ids = list(map(lambda p: self.token2idx[p], phones))
        return token_ids


class Idx2phone(object):
    """Class for converting indices to phone sequence.

    Args:
        dict_path (str): path to a dictionary file
        remove_list (list): phones to ingore

    """

    def __init__(self, dict_path, remove_list=[]):
        # Load a dictionary file
        self.idx2token = {0: '<blank>'}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                p, idx = line.strip().split(' ')
                if p in remove_list:
                    continue
                self.idx2token[int(idx)] = p
        self.vocab = len(self.idx2token.keys())
        # for synchronous bidirectional attention
        self.idx2token[self.vocab] = '<l2r>'
        self.idx2token[self.vocab + 1] = '<r2l>'
        self.idx2token[self.vocab + 2] = '<null>'

    def __call__(self, token_ids, return_list=False):
        """Convert indices to phone sequence.

        Args:
            token_ids (list): phone indices
            return_list (bool): if True, return list of phones
        Returns:
            text (str): phone sequence divided by spaces
                or
            phones (list): list of phones

        """
        phones = list(map(lambda i: self.idx2token[i], token_ids))
        if return_list:
            return phones
        return ' '.join(phones)
