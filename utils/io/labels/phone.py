#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Phone2idx(object):
    """Convert from phone to index.
    Args:
        vocab_file_path (string): path to the vocabulary file
        remove_list (list, optional): phones to neglect
    """

    def __init__(self, vocab_file_path, remove_list=[]):
        # Read the vocabulary file
        self.map_dict = {}
        vocab_count = 0
        with open(vocab_file_path, 'r') as f:
            for line in f:
                phone = line.strip()
                if phone in remove_list:
                    continue
                self.map_dict[phone] = vocab_count
                vocab_count += 1

        # Add <EOS>
        self.map_dict['>'] = vocab_count

    def __call__(self, str_phone):
        """
        Args:
            str_phone (string): string of space-divided phones
        Returns:
            index_list (np.ndarray): phone indices
        """
        # Convert from phone to the corresponding indices
        phone_list = str_phone.split(' ')
        index_list = list(map(lambda x: self.map_dict[x], phone_list))

        return np.array(index_list)


class Idx2phone(object):
    """Convert from index to phone.
    Args:
        vocab_file_path (string): path to the vocabulary file
        remove_list (list, optional): phones to neglect
    """

    def __init__(self, vocab_file_path, remove_list=[]):
        # Read the vocabulary file
        self.map_dict = {}
        vocab_count = 0
        with open(vocab_file_path, 'r') as f:
            for line in f:
                phone = line.strip()
                if phone in remove_list:
                    continue
                self.map_dict[vocab_count] = phone
                vocab_count += 1

        # Add <EOS>
        self.map_dict[vocab_count] = '>'

    def __call__(self, index_list):
        """
        Args:
            index_list (list): phone indices
        Returns:
            str_phone (string): a sequence of phones
        """
        # Convert from indices to the corresponding phones
        phone_list = list(map(lambda x: self.map_dict[x], index_list))
        str_phone = ' '.join(phone_list)

        return str_phone
