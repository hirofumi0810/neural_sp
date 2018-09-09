#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import six


class Char2idx(object):
    """Convert character into index.

    Args:
        dict_path (str): path to the vocabulary file
        capital_divide (bool): if True, words will be divided by
            capital letters. This is used for English.
        double_letter (bool): if True, group repeated letters.
            This is used for Japanese.
        remove_list (list): characters to neglect

    """

    def __init__(self, dict_path, capital_divide=False,
                 double_letter=False, remove_list=[]):
        self.capital_divide = capital_divide
        self.double_letter = double_letter
        self.remove_list = remove_list

        # Load the vocabulary file
        self.token2id = {}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                c, id = line.strip().encode('utf_8').split(' ')
                if c in remove_list:
                    continue
                self.token2id[c] = int(id)

    def __call__(self, text):
        """

        Args:
            text (str): a sequence of characters.
                This doesn't include <space> (separated by whitespaces)
        Returns:
            indices (list): character indices

        """
        indices = []

        # Convert character strings into the corresponding indices
        if self.capital_divide:
            for w in text.split(' '):
                # Replace the first character with the capital letter
                indices.append(self.token2id[w[0].upper()])
                indices.append(self.token2id['<space>'])

                # Check double-letters
                skip_flag = False
                for i in six.moves.range(1, len(w) - 1, 1):
                    if skip_flag:
                        skip_flag = False
                        continue

                    if not skip_flag and w[i:i + 2] in self.token2id.keys():
                        indices.append(self.token2id[w[i:i + 2]])
                        skip_flag = True
                    else:
                        indices.append(self.token2id[w[i]])
                    indices.append(self.token2id['<space>'])

                # Final character
                if not skip_flag:
                    indices.append(self.token2id[w[-1]])
        # NOTE: capital_divide is prepared only for English

        elif self.double_letter:
            skip_flag = False
            for i in six.moves.range(len(text) - 1):
                if skip_flag:
                    skip_flag = False
                    continue

                if not skip_flag and text[i:i + 2] in self.token2id.keys():
                    indices.append(self.token2id[text[i:i + 2]])
                    skip_flag = True
                elif text[i] in self.token2id.keys():
                    indices.append(self.token2id[text[i]])
                elif text[i] == ' ':
                    indices.append(self.token2id['<space>'])
                else:
                    indices.append(self.token2id['<unk>'])

            # Final character
            if not skip_flag:
                if text[-1] in self.token2id.keys():
                    indices.append(self.token2id[text[-1]])
                elif text[-1] == ' ':
                    indices.append(self.token2id['<space>'])
                else:
                    indices.append(self.token2id['<unk>'])
        # NOTE: double_letter is prepared for Japanese

        else:
            # NOTE: OOV handling is prepared for Japanese and Chinese
            for c in list(text):
                if c in self.token2id.keys():
                    indices.append(self.token2id[c])
                elif c == ' ':
                    indices.append(self.token2id['<space>'])
                else:
                    indices.append(self.token2id['<unk>'])

        return indices


class Idx2char(object):
    """Convert index into character.

    Args:
        dict_path (str): path to the vocabulary file
        capital_divide (bool): if True, words will be divided by
            capital letters. This is used for English.
        remove_list (list): characters to neglect

    """

    def __init__(self, dict_path, capital_divide=False, remove_list=[]):
        self.capital_divide = capital_divide
        self.remove_list = remove_list

        # Load the vocabulary file
        self.id2token = {}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                c, id = line.strip().encode('utf_8').split(' ')
                if c in remove_list:
                    continue
                self.id2token[int(id)] = c

    def __call__(self, indices, return_list=False):
        """

        Args:
            indices (list): list of character indices
            return_list (bool): if True, return list of characters
        Returns:
            text (str): a sequence of characters
                or
            char_list (list): list of characters

        """
        char_list = list(map(lambda c: self.id2token[c], indices))

        # Convert character indices into the corresponding strings
        if self.capital_divide:
            char_list_cd = []
            for i in six.moves.range(len(char_list)):
                if i != 0 and 'A' <= char_list[i] <= 'Z':
                    char_list_cd += ['<space>', char_list[i].lower()]
                else:
                    char_list_cd += [char_list[i].lower()]

            if return_list:
                return char_list_cd
            text = ' '.join(char_list_cd)
        else:
            if return_list:
                return char_list
            text = ' '.join(char_list)

        return text
