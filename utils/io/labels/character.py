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
        space_mark (string, optional): the space mark to divide a sequence into words
        capital_divide (bool, optional): if True, words will be divided by
            capital letters. This is used for English.
        double_letter (bool, optional): if True, group repeated letters.
            This is used for Japanese.
        remove_list (list, optional): characters to neglect
    """

    def __init__(self, vocab_file_path, space_mark='_', capital_divide=False,
                 double_letter=False, remove_list=[]):
        self.space_mark = space_mark
        self.capital_divide = capital_divide
        self.double_letter = double_letter
        self.remove_list = remove_list

        # Read the vocabulary file
        self.map_dict = {}
        vocab_count = 0
        with open(vocab_file_path, 'r') as f:
            for line in f:
                char = line.strip()
                if char in remove_list:
                    continue
                self.map_dict[char] = vocab_count
                vocab_count += 1

        # Add <EOS>
        self.map_dict['>'] = vocab_count

    def __call__(self, str_char):
        """
        Args:
            str_char (string): a sequence of characters
        Returns:
            index_list (list): character indices
        """
        index_list = []

        # Convert from character to index
        if self.capital_divide:
            for word in str_char.split(self.space_mark):
                # Replace the first character with the capital letter
                index_list.append(self.map_dict[word[0].upper()])

                # Check double-letters
                skip_flag = False
                for i in range(1, len(word) - 1, 1):
                    if skip_flag:
                        skip_flag = False
                        continue

                    if not skip_flag and word[i:i + 2] in self.map_dict.keys():
                        index_list.append(self.map_dict[word[i:i + 2]])
                        skip_flag = True
                    else:
                        index_list.append(self.map_dict[word[i]])

                # Final character
                if not skip_flag:
                    index_list.append(self.map_dict[word[-1]])

        elif self.double_letter:
            skip_flag = False
            for i in range(len(str_char) - 1):
                if skip_flag:
                    skip_flag = False
                    continue

                if not skip_flag and str_char[i:i + 2] in self.map_dict.keys():
                    index_list.append(self.map_dict[str_char[i:i + 2]])
                    skip_flag = True
                else:
                    index_list.append(self.map_dict[str_char[i]])

            # Final character
            if not skip_flag:
                index_list.append(self.map_dict[str_char[-1]])

        else:
            index_list = list(map(lambda x: self.map_dict[x], list(str_char)))

        return np.array(index_list)


class Idx2char(object):
    """Convert from index to character.
    Args:
        vocab_file_path (string): path to the vocabulary file
        space_mark (string, optional): the space mark to divide a sequence into words
        capital_divide (bool, optional): if True, words will be divided by
            capital letters. This is used for English.
        remove_list (list, optional): characters to neglect
        return_list (bool, optional): if True, return list of characters
    """

    def __init__(self, vocab_file_path, space_mark='_', capital_divide=False,
                 remove_list=[], return_list=False):
        self.space_mark = space_mark
        self.capital_divide = capital_divide
        self.remove_list = remove_list
        self.return_list = return_list

        # Read the vocabulary file
        self.map_dict = {}
        vocab_count = 0
        with open(vocab_file_path, 'r') as f:
            for line in f:
                char = line.strip()
                if char in remove_list:
                    continue
                self.map_dict[vocab_count] = char
                vocab_count += 1

        # Add <EOS>
        self.map_dict[vocab_count] = '>'

    def __call__(self, index_list):
        """
        Args:
            index_list (list): list of character indices.
        Returns:
            str_char (string): a sequence of characters
        """
        _char_list = list(map(lambda x: self.map_dict[x], index_list))
        char_list = []

        # Convert from indices to the corresponding characters
        if self.capital_divide:
            for i in range(len(_char_list)):
                if i != 0 and 'A' <= _char_list[i] <= 'Z':
                    char_list += [self.space_mark, _char_list[i].lower()]
                else:
                    char_list += [_char_list[i].lower()]

            if self.return_list:
                return char_list

            str_char = ''.join(char_list)
        else:
            if self.return_list:
                return _char_list

            str_char = ''.join(_char_list)

        return str_char
