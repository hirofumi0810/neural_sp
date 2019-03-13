#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs


class Phone2idx(object):
    """Class for converting phone sequence to indices.

    Args:
        dict_path (str): path to a vocabulary file
        remove_list (list): phones to ingore

    """

    def __init__(self, dict_path, remove_list=[]):
        # Load a vocabulary file
        self.token2idx = {}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                p, idx = line.strip().split(' ')
                if p in remove_list:
                    continue
                self.token2idx[p] = int(idx)

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
        dict_path (str): path to a vocabulary file
        remove_list (list): phones to ingore

    """

    def __init__(self, dict_path, remove_list=[]):
        # Load a vocabulary file
        self.idx2token = {0: '<blank>'}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                p, idx = line.strip().split(' ')
                if p in remove_list:
                    continue
                self.idx2token[int(idx)] = p

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
