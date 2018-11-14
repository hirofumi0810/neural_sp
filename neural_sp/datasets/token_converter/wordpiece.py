#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import sentencepiece as spm


class Wp2idx(object):
    """Convert word-piece string into tokenid.

    Args:
        dict_path (str): path to the dictionary file

    """

    def __init__(self, dict_path, wp_model):
        # Load the dictionary file
        self.wp2idx = {}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                wp, id = line.strip().encode('utf_8').split(' ')
                self.wp2idx[wp] = int(id)

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(wp_model)

    def __call__(self, text):
        """Convert word-piece string into tokenid.

        Args:
            text (str):
        Returns:
            tokenids (list): word-piece tokenids

        """
        wp_list = self.sp.EncodeAsPieces(text)
        tokenids = []
        for wp in wp_list:
            if wp in self.wp2idx.keys():
                tokenids.append(self.wp2idx[wp])
            else:
                # Replace with <unk>
                tokenids.append(self.wp2idx['<unk>'])
        return tokenids


class Idx2wp(object):
    """Convert tokenid into word-piece string.

    Args:
        dict_path (str): path to the dictionary file

    """

    def __init__(self, dict_path, wp_model):
        # Load the dictionary file
        self.idx2wp = {}
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                wp, id = line.strip().encode('utf_8').split(' ')
                self.idx2wp[int(id)] = wp

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(wp_model)

    def __call__(self, tokenids, return_list=False):
        """Convert tokenid into word-piece string.

        Args:
            tokenids (np.ndarray or list): list of word-piece tokenids
            return_list (bool): if True, return list of words
        Returns:
            text (str):
                or
            wp_list (list): list of words

        """
        # Convert word-piece tokenids into the corresponding strings
        wp_list = list(map(lambda wp: self.idx2wp[wp], tokenids))
        if return_list:
            return wp_list
        return self.sp.DecodePieces(wp_list)
