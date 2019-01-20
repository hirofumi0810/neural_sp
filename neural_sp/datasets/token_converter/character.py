#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs


class Char2id(object):
    """Class for converting character sequence into indices.

    Args:
        dict_path (str): path to a vocabulary file
        remove_list (list): characters to ignore

    """

    def __init__(self, dict_path, nlsyms=None, remove_space=False, remove_list=[]):
        self.remove_space = remove_space
        self.remove_list = remove_list

        # Load a vocabulary file
        self.token2id = {}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                c, id = line.strip().encode('utf_8').split(' ')
                if c in remove_list:
                    continue
                self.token2id[c] = int(id)

    def __call__(self, text):
        """Convert character sequence into indices.

        Args:
            text (str): character sequence
        Returns:
            token_ids (list): character indices

        """
        token_ids = []
        text = text.replace(' ', '<space>')
        words = text.split('<space>')
        for i,  w in enumerate(words):
            if w in self.nlsyms:
                token_ids.append(self.token2id[w])
            else:
                for c in list(w):
                    if c in self.token2id.keys():
                        token_ids.append(self.token2id[c])
                    else:
                        # Replace with <unk>
                        token_ids.append(self.token2id['<unk>'])
                        # NOTE: OOV handling is prepared for Japanese and Chinese

            if not self.remove_space:
                if i < len(words) - 1:
                    token_ids.append(self.token2id['<space>'])
        return token_ids


class Id2char(object):
    """Class for converting indices into character sequence.

    Args:
        dict_path (str): path to a vocabulary file
        remove_list (list): characters to ignore

    """

    def __init__(self, dict_path,  remove_list=[]):
        self.remove_list = remove_list

        # Load a vocabulary file
        self.id2token = {0: '<blank>'}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                c, id = line.strip().encode('utf_8').split(' ')
                if c in remove_list:
                    continue
                self.id2token[int(id)] = c

    def __call__(self, token_ids, return_list=False):
        """Convert indices into character sequence.

        Args:
            token_ids (np.ndarray or list): character indices
            return_list (bool): if True, return list of characters
        Returns:
            text (str): character sequence
                or
            char_list (list): list of characters

        """
        char_list = list(map(lambda c: self.id2token[c], token_ids))
        if return_list:
            return char_list
        return ''.join(char_list).replace('<space>', ' ')
