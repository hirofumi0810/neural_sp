# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Wordpiece-level token <-> index converter."""

import codecs
import sentencepiece as spm


class Wp2idx(object):
    """Class for converting word-piece sequence into indices.

    Args:
        dict_path (str): path to a dictionary file
        wp_model ():

    """

    def __init__(self, dict_path, wp_model):
        # Load a dictionary file
        self.token2idx = {'<blank>': 0}
        with codecs.open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                wp, idx = line.strip().split(' ')
                self.token2idx[wp] = int(idx)
        self.vocab = len(self.token2idx.keys())

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(wp_model)

    def __call__(self, text):
        """Convert word-piece sequence into indices.

        Args:
            text (str): word sequence
        Returns:
            token_ids (list): word-piece indices

        """
        # Remove space before the first special symbol
        wps = self.sp.EncodeAsPieces(text)
        if wps[0] == '▁' and wps[1][0] == '<':
            wps = wps[1:]

        token_ids = []
        for wp in wps:
            if wp in self.token2idx.keys():
                token_ids.append(self.token2idx[wp])
            else:
                # Replace with <unk>
                token_ids.append(self.token2idx['<unk>'])
        return token_ids


class Idx2wp(object):
    """Class for converting indices into word-piece sequence.

    Args:
        dict_path (str): path to a dictionary file
        wp_model ():

    """

    def __init__(self, dict_path, wp_model):
        # Load a dictionary file
        self.idx2token = {0: '<blank>'}
        with codecs.open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                wp, idx = line.strip().split(' ')
                self.idx2token[int(idx)] = wp
        self.vocab = len(self.idx2token.keys())
        # for synchronous bidirectional attention
        self.idx2token[self.vocab] = '<l2r>'
        self.idx2token[self.vocab + 1] = '<r2l>'
        self.idx2token[self.vocab + 2] = '<null>'

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(wp_model)

    def __call__(self, token_ids, return_list=False):
        """Convert indices into word-piece sequence.

        Args:
            token_ids (np.ndarray or list): word-piece indices
            return_list (bool): if True, return list of words
        Returns:
            text (str): word sequence
                or
            wordpieces (list): list of words

        """
        if len(token_ids) == 0:
            return ''
        wordpieces = list(map(lambda wp: self.idx2token[wp], token_ids))
        if return_list:
            return wordpieces
        return self.sp.DecodePieces(wordpieces)

    def is_word_boundary(self):
        raise NotImplementedError
