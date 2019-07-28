#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for loading dataset for language model.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pandas as pd
import random

from neural_sp.datasets.asr import count_vocab_size
from neural_sp.datasets.token_converter.character import Char2idx
from neural_sp.datasets.token_converter.character import Idx2char
from neural_sp.datasets.token_converter.phone import Idx2phone
from neural_sp.datasets.token_converter.phone import Phone2idx
from neural_sp.datasets.token_converter.word import Idx2word
from neural_sp.datasets.token_converter.word import Word2idx
from neural_sp.datasets.token_converter.wordpiece import Idx2wp
from neural_sp.datasets.token_converter.wordpiece import Wp2idx

random.seed(1)
np.random.seed(1)


class Dataset(object):

    def __init__(self, tsv_path, dict_path,
                 unit, batch_size, nlsyms=False, n_epochs=None,
                 is_test=False, min_n_tokens=1, bptt=2,
                 shuffle=False, backward=False, serialize=False,
                 wp_model=None, corpus=''):
        """A class for loading dataset.

        Args:
            tsv_path (str): path to the dataset tsv file
            dict_path (str): path to the dictionary
            unit (str): word or wp or char or phone or word_char
            batch_size (int): size of mini-batch
            nlsyms (str): path to the non-linguistic symbols file
            n_epochs (int): max epoch. None means infinite loop
            is_test (bool):
            min_n_tokens (int): exclude utterances shorter than this value
            bptt (int): BPTT length
            shuffle (bool): shuffle utterances.
                This is disabled when sort_by_input_length is True.
            backward (bool): flip all text in the corpus
            serialize (bool): serialize text according to contexts in dialogue
            wp_model (): path to the word-piece model for sentencepiece
            corpus (str): name of corpus


        """
        super(Dataset, self).__init__()

        self.epoch = 0
        self.iteration = 0
        self.offset = 0

        self.set = os.path.basename(tsv_path).split('.')[0]
        self.is_test = is_test
        self.unit = unit
        self.batch_size = batch_size
        self.bptt = bptt
        self.sos = 2
        self.eos = 2
        self.max_epoch = n_epochs
        self.shuffle = shuffle
        self.vocab = count_vocab_size(dict_path)
        assert bptt >= 2

        self.idx2token = []
        self.token2idx = []

        # Set index converter
        if unit in ['word', 'word_char']:
            self.idx2token += [Idx2word(dict_path)]
            self.token2idx += [Word2idx(dict_path, word_char_mix=(unit == 'word_char'))]
        elif unit == 'wp':
            self.idx2token += [Idx2wp(dict_path, wp_model)]
            self.token2idx += [Wp2idx(dict_path, wp_model)]
        elif unit == 'char':
            self.idx2token += [Idx2char(dict_path)]
            self.token2idx += [Char2idx(dict_path, nlsyms=nlsyms)]
        elif 'phone' in unit:
            self.idx2token += [Idx2phone(dict_path)]
            self.token2idx += [Phone2idx(dict_path)]
        else:
            raise ValueError(unit)

        # Load dataset tsv file
        self.df = pd.read_csv(tsv_path, encoding='utf-8', delimiter='\t')
        self.df = self.df.loc[:, ['utt_id', 'speaker', 'feat_path',
                                  'xlen', 'xdim', 'text', 'token_id', 'ylen', 'ydim']]

        # Remove inappropriate utterances
        if is_test:
            print('Original utterance num: %d' % len(self.df))
            n_utts = len(self.df)
            self.df = self.df[self.df.apply(lambda x: x['ylen'] > 0, axis=1)]
            print('Removed %d empty utterances' % (n_utts - len(self.df)))
        else:
            print('Original utterance num: %d' % len(self.df))
            n_utts = len(self.df)
            self.df = self.df[self.df.apply(lambda x: x['ylen'] >= min_n_tokens, axis=1)]
            print('Removed %d utterances (threshold)' % (n_utts - len(self.df)))

        # Sort tsv records
        if shuffle:
            self.df = self.df.reindex(np.random.permutation(self.df.index))
        elif serialize:
            assert corpus == 'swbd'
            self.df['session'] = self.df['speaker'].apply(lambda x: str(x).split('-')[0])
            self.df['onset'] = self.df['utt_id'].apply(lambda x: int(x.split('_')[-1].split('-')[0]))
            self.df = self.df.sort_values(by=['session', 'onset'], ascending=True)
        else:
            self.df = self.df.sort_values(by='utt_id', ascending=True)

        # Concatenate into a single sentence
        concat_ids = []
        indices = list(self.df.index)
        if backward:
            indices = indices[::-1]
        for i in indices:
            assert self.df['token_id'][i] != ''
            concat_ids += [self.eos] + list(map(int, self.df['token_id'][i].split()))
        concat_ids += [self.eos]
        # NOTE: <sos> and <eos> have the same index

        # Reshape
        n_utts = len(concat_ids)
        concat_ids = concat_ids[:n_utts // batch_size * batch_size]
        print('Removed %d tokens / %d tokens' % (n_utts - len(concat_ids), n_utts))
        self.concat_ids = np.array(concat_ids).reshape((batch_size, -1))

    def __len__(self):
        return len(self.concat_ids.reshape((-1,)))

    @property
    def epoch_detail(self):
        """Percentage of the current epoch."""
        return float(self.offset * self.batch_size) / len(self)

    def reset(self):
        """Reset data counter and offset."""
        self.df_indices = list(self.df.index)
        self.offset = 0

    def next(self, batch_size=None, bptt=None):
        """Generate each mini-batch.

        Args:
            batch_size (int): size of mini-batch
            bptt (int): BPTT length
        Returns:
            batch (dict):
                ys (list): target labels in the main task of size `[B, L]`
                utt_ids (list): file names of input data of size `[B]`
            is_new_epoch (bool): flag for the end of the current epoch

        """
        is_new_epoch = False

        if batch_size is None:
            batch_size = self.batch_size
        elif self.concat_ids.shape[0] != batch_size:
            self.concat_ids = self.concat_ids.reshape((batch_size, -1))
            # NOTE: only for the first iteration during evaluation

        if bptt is None:
            bptt = self.bptt

        if self.max_epoch is not None and self.epoch >= self.max_epoch:
            raise StopIteration
            # NOTE: max_epoch == None means infinite loop

        ys = self.concat_ids[:, self.offset:self.offset + bptt]
        self.offset += bptt - 1
        # NOTE: the last token in ys must be feeded as inputs in the next mini-batch

        # Last mini-batch
        if (self.offset + 1) * batch_size >= len(self):
            self.offset = 0
            is_new_epoch = True
            self.epoch += 1

        return ys, is_new_epoch
