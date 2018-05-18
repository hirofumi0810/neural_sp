#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import codecs


class Char2idx(object):
    """Convert character into index.
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

        # Load the vocabulary file
        self.map_dict = {}
        vocab_count = 0
        with codecs.open(vocab_file_path, 'r', 'utf-8') as f:
            for line in f:
                c = line.strip()
                if c in remove_list:
                    continue
                self.map_dict[c] = vocab_count
                vocab_count += 1

        # Add <EOS>
        self.map_dict['>'] = vocab_count

    def __call__(self, str_char):
        """
        Args:
            str_char (string): a sequence of characters
        Returns:
            indices (list): character indices
        """
        indices = []

        # Convert character strings into the corresponding indices
        if self.capital_divide:
            for w in str_char.split(self.space_mark):
                # Replace the first character with the capital letter
                indices.append(self.map_dict[w[0].upper()])

                # Check double-letters
                skip_flag = False
                for i in range(1, len(w) - 1, 1):
                    if skip_flag:
                        skip_flag = False
                        continue

                    if not skip_flag and w[i:i + 2] in self.map_dict.keys():
                        indices.append(self.map_dict[w[i:i + 2]])
                        skip_flag = True
                    else:
                        indices.append(self.map_dict[w[i]])

                # Final character
                if not skip_flag:
                    indices.append(self.map_dict[w[-1]])
        # NOTE: capital_divide is prepared for English

        elif self.double_letter:
            skip_flag = False
            for i in range(len(str_char) - 1):
                if skip_flag:
                    skip_flag = False
                    continue

                if not skip_flag and str_char[i:i + 2] in self.map_dict.keys():
                    indices.append(self.map_dict[str_char[i:i + 2]])
                    skip_flag = True
                elif str_char[i] in self.map_dict.keys():
                    indices.append(self.map_dict[str_char[i]])
                else:
                    indices.append(self.map_dict['OOV'])

            # Final character
            if not skip_flag:
                if str_char[-1] in self.map_dict.keys():
                    indices.append(self.map_dict[str_char[-1]])
                else:
                    indices.append(self.map_dict['OOV'])
        # NOTE: double_letter is prepared for Japanese

        else:
            for c in list(str_char):
                if c in self.map_dict.keys():
                    indices.append(self.map_dict[c])
                else:
                    indices.append(self.map_dict['OOV'])
        # NOTE: OOV handling is prepared for Japanese and Chinese

        return np.array(indices)


class Idx2char(object):
    """Convert index into character.
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

        # Load the vocabulary file
        self.map_dict = {}
        vocab_count = 0
        with codecs.open(vocab_file_path, 'r', 'utf-8') as f:
            for line in f:
                c = line.strip()
                if c in remove_list:
                    continue
                self.map_dict[vocab_count] = c
                vocab_count += 1

        # Add <EOS>
        self.map_dict[vocab_count] = '>'

    def __call__(self, indices):
        """
        Args:
            indices (list): list of character indices.
        Returns:
            str_char (string): a sequence of characters
                or
            char_list (list): list of characters
        """
        _char_list = list(map(lambda c: self.map_dict[c], indices))
        char_list = []

        # Convert character indices into the corresponding strings
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
