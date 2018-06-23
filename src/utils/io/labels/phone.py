#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs


class Phone2idx(object):
    """Convert from phone to index.
    Args:
        vocab_file_path (string): path to the vocabulary file
        remove_list (list): phones to neglect
    """

    def __init__(self, vocab_file_path, remove_list=[]):
        # Load the vocabulary file
        self.map_dict = {}
        vocab_count = 0
        with codecs.open(vocab_file_path, 'r', 'utf-8') as f:
            for line in f:
                p = line.strip()
                if p in remove_list:
                    continue
                self.map_dict[p] = vocab_count
                vocab_count += 1

        # Add <EOS>
        self.map_dict['>'] = vocab_count

    def __call__(self, str_phone):
        """
        Args:
            str_phone (string): string of space-divided phones
        Returns:
            indices (list): phone indices
        """
        # Convert phone strings into the corresponding indices
        phone_list = str_phone.split('_')
        indices = list(map(lambda x: self.map_dict[x], phone_list))
        return indices


class Idx2phone(object):
    """Convert from index to phone.
    Args:
        vocab_file_path (string): path to the vocabulary file
        remove_list (list): phones to neglect
    """

    def __init__(self, vocab_file_path, remove_list=[]):
        # Load the vocabulary file
        self.map_dict = {}
        vocab_count = 0
        with codecs.open(vocab_file_path, 'r', 'utf-8') as f:
            for line in f:
                p = line.strip()
                if p in remove_list:
                    continue
                self.map_dict[vocab_count] = p
                vocab_count += 1

        # Add <EOS>
        self.map_dict[vocab_count] = '>'

    def __call__(self, indices, return_list=False):
        """
        Args:
            indices (list): phone indices
            return_list (bool): if True, return list of phones
        Returns:
            str_phone (string): a sequence of phones
        """
        # Convert phone indices to the corresponding strings
        phone_list = list(map(lambda x: self.map_dict[x], indices))
        if return_list:
            return phone_list
        str_phone = '_'.join(phone_list)
        return str_phone
