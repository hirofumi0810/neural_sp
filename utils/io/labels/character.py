#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Char2idx(object):
    """Convert from character to index.
    Args:
        vocab_file_path (string): path to the vocabulary file
        double_letter (bool, optional): if True, group repeated letters
    """

    def __init__(self, vocab_file_path, double_letter=False):
        self.double_letter = double_letter

        # Read the vocabulary file
        self.map_dict = {}
        vocab_count = 0
        with open(vocab_file_path, 'r') as f:
            for line in f:
                char = line.strip()
                self.map_dict[char] = vocab_count
                vocab_count += 1

        # Add <SOS> & <EOS>
        self.map_dict['<'] = vocab_count
        self.map_dict['>'] = vocab_count + 1

    def __call__(self, str_char):
        """
        Args:
            str_char (string): string of characters
        Returns:
            char_list (list): character indices
        """
        char_list = list(str_char)

        # Convert from character to index
        if self.double_letter:
            skip_flag = False
            for i in range(len(char_list) - 1):
                if skip_flag:
                    char_list[i] = ''
                    skip_flag = False
                    continue

                if char_list[i] + char_list[i + 1] in self.map_dict.keys():
                    char_list[i] = self.map_dict[char_list[i] +
                                                 char_list[i + 1]]
                    skip_flag = True
                else:
                    char_list[i] = self.map_dict[char_list[i]]

            # Final character
            if skip_flag:
                char_list[-1] = ''
            else:
                char_list[-1] = self.map_dict[char_list[-1]]

            # Remove skipped characters
            while '' in char_list:
                char_list.remove('')
        else:
            for i in range(len(char_list)):
                char_list[i] = self.map_dict[char_list[i]]

        return char_list


class Idx2char(object):
    """Convert from index to character.
    Args:
        vocab_file_path (string): path to the vocabulary file
        capital_divide (bool, optional): set True when using capital-divided
            character sequences
        space_mark (string): the space mark to divide a sequence into words
    """

    def __init__(self, vocab_file_path, capital_divide=False, space_mark=' '):
        self.capital_divide = capital_divide
        self.space_mark = space_mark

        # Read the vocabulary file
        self.map_dict = {}
        vocab_count = 0
        with open(vocab_file_path, 'r') as f:
            for line in f:
                char = line.strip()
                self.map_dict[vocab_count] = char
                vocab_count += 1

        # Add <SOS> & <EOS>
        self.map_dict[vocab_count] = '<'
        self.map_dict[vocab_count + 1] = '>'

    def __call__(self, index_list, padded_value=-1):
        """
        Args:
            index_list (np.ndarray): list of character indices.
                Batch size 1 is expected.
            padded_value (int): the value used for padding
        Returns:
            str_char (string): a sequence of characters
        """
        # Remove padded values
        assert type(
            index_list) == np.ndarray, 'index_list should be np.ndarray.'
        index_list = np.delete(index_list, np.where(
            index_list == padded_value), axis=0)

        # Convert from indices to the corresponding characters
        char_list = list(map(lambda x: self.map_dict[x], index_list))

        if self.capital_divide:
            char_list_tmp = []
            for i in range(len(char_list)):
                if i != 0 and 'A' <= char_list[i] <= 'Z':
                    char_list_tmp += [self.space_mark, char_list[i].lower()]
                else:
                    char_list_tmp += [char_list[i].lower()]
            str_char = ''.join(char_list_tmp)
        else:
            str_char = ''.join(char_list)

        return str_char
        # TODO: change to batch version
