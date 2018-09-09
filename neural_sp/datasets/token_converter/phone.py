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
    """Convert from phone to index.

    Args:
        dict_path (str): path to the vocabulary file
        remove_list (list): phones to neglect

    """

    def __init__(self, dict_path, remove_list=[]):
        # Load the vocabulary file
        self.token2id = {}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                p, id = line.strip().encode('utf_8').split(' ')
                if p in remove_list:
                    continue
                self.token2id[p] = int(id)

    def __call__(self, text):
        """

        Args:
            text (str): string of space-divided phones
        Returns:
            indices (list): phone indices

        """
        # Convert phone strings into the corresponding indices
        phone_list = text.split(' ')
        indices = list(map(lambda p: self.token2id[p], phone_list))
        return indices


class Idx2phone(object):
    """Convert from index to phone.

    Args:
        dict_path (str): path to the vocabulary file
        remove_list (list): phones to neglect

    """

    def __init__(self, dict_path, remove_list=[]):
        # Load the vocabulary file
        self.id2token = {}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                p, id = line.strip().encode('utf_8').split(' ')
                if p in remove_list:
                    continue
                self.id2token[int(id)] = p

    def __call__(self, indices, return_list=False):
        """

        Args:
            indices (list): phone indices
            return_list (bool): if True, return list of phones
        Returns:
            text (str): a sequence of phones

        """
        # Convert phone indices to the corresponding strings
        phone_list = list(map(lambda i: self.id2token[i], indices))
        if return_list:
            return phone_list
        text = ' '.join(phone_list)
        return text
