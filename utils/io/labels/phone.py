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
    """

    def __init__(self, vocab_file_path):
        # Read the vocabulary file
        self.map_dict = {}
        vocab_count = 0
        with open(vocab_file_path, 'r') as f:
            for line in f:
                phone = line.strip()
                self.map_dict[phone] = vocab_count
                vocab_count += 1

        # Add <SOS> & <EOS>
        self.map_dict['<'] = vocab_count
        self.map_dict['>'] = vocab_count + 1

    def __call__(self, phone_list):
        """
        Args:
            phone_list (list): list of phones (string)
        Returns:
            phone_list (list): phone indices
        """
        # Convert from phone to index
        for i in range(len(phone_list)):
            phone_list[i] = self.map_dict[phone_list[i]]
        return np.array(phone_list)


class Idx2phone(object):
    """Convert from index to phone.
    Args:
        vocab_file_path (string): path to the vocabulary file
    """

    def __init__(self, vocab_file_path):
        # Read the vocabulary file
        self.map_dict = {}
        vocab_count = 0
        with open(vocab_file_path, 'r') as f:
            for line in f:
                phone = line.strip()
                self.map_dict[vocab_count] = phone
                vocab_count += 1

        # Add <SOS> & <EOS>
        self.map_dict[vocab_count] = '<'
        self.map_dict[vocab_count + 1] = '>'

    def __call__(self, index_list, padded_value=-1):
        """
        Args:
            index_list (list): phone indices
            padded_value (int): the value used for padding
        Returns:
            str_phone (string): a sequence of phones
        """
        # Remove padded values
        assert type(
            index_list) == np.ndarray, 'index_list should be np.ndarray.'
        index_list = np.delete(index_list, np.where(index_list == -1), axis=0)

        # Convert from indices to the corresponding phones
        phone_list = list(map(lambda x: self.map_dict[x], index_list))
        str_phone = ' '.join(phone_list)

        return str_phone
