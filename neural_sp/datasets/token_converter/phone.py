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
        self.token2id = {}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                p, id = line.strip().encode('utf_8').split(' ')
                if p in remove_list:
                    continue
                self.token2id[p] = int(id)

    def __call__(self, text):
        """Convert phone sequence to indices.

        Args:
            text (str): phone sequence divided by spaces
        Returns:
            token_ids (list): phone indices

        """
        phone_list = text.split(' ')
        token_ids = list(map(lambda p: self.token2id[p], phone_list))
        return token_ids


class Idx2phone(object):
    """Class for converting indices to phone sequence.

    Args:
        dict_path (str): path to a vocabulary file
        remove_list (list): phones to ingore

    """

    def __init__(self, dict_path, remove_list=[]):
        # Load a vocabulary file
        self.id2token = {}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                p, id = line.strip().encode('utf_8').split(' ')
                if p in remove_list:
                    continue
                self.id2token[int(id)] = p

    def __call__(self, token_ids, return_list=False):
        """Convert indices to phone sequence.

        Args:
            token_ids (list): phone indices
            return_list (bool): if True, return list of phones
        Returns:
            text (str): phone sequence divided by spaces
                or
            phone_list (list): list of phones

        """
        phone_list = list(map(lambda i: self.id2token[i], token_ids))
        if return_list:
            return phone_list
        return ' '.join(phone_list)
