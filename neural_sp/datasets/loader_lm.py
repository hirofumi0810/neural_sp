#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for loading dataset for the RNNLM.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import random
import os

from neural_sp.datasets.base import Base
from neural_sp.datasets.token_converter.character import Char2id
from neural_sp.datasets.token_converter.character import Id2char
from neural_sp.datasets.token_converter.phone import Id2phone
from neural_sp.datasets.token_converter.phone import Phone2id
from neural_sp.datasets.token_converter.word import Id2word
from neural_sp.datasets.token_converter.word import Word2id
from neural_sp.datasets.token_converter.wordpiece import Id2wp
from neural_sp.datasets.token_converter.wordpiece import Wp2id

random.seed(1)
np.random.seed(1)


class Dataset(Base):

    def __init__(self, csv_path, dict_path,
                 unit, batch_size, nepochs=None,
                 is_test=False, bptt=-1, shuffle=False, wp_model=None):
        """A class for loading dataset.

        Args:
            csv_path (str):
            dict_path (str):
            unit (str): word or wp or char or phone or word_char
            batch_size (int): the size of mini-batch
            bptt (int):
            nepochs (int): the max epoch. None means infinite loop.
            shuffle (bool): if True, shuffle utterances.
                This is disabled when sort_by_input_length is True.
            wp_model ():

        """
        super(Dataset, self).__init__()

        self.set = os.path.basename(csv_path).split('.')[0]
        self.is_test = is_test
        self.unit = unit
        self.batch_size = batch_size
        self.bptt = bptt
        self.sos = 2
        self.eos = 2
        self.max_epoch = nepochs
        self.shuffle = shuffle
        self.vocab = self.count_vocab_size(dict_path)

        # Set index converter
        if unit in ['word', 'word_char']:
            self.id2word = Id2word(dict_path)
            self.word2id = Word2id(dict_path, word_char_mix=(unit == 'word_char'))
        elif unit == 'wp':
            self.id2wp = Id2wp(dict_path, wp_model)
            self.wp2id = Wp2id(dict_path, wp_model)
        elif unit == 'char':
            self.id2char = Id2char(dict_path)
            self.char2id = Char2id(dict_path)
        else:
            raise ValueError(unit)

        # Load dataset csv file
        df = pd.read_csv(csv_path, encoding='utf-8', delimiter=',')
        df = df.loc[:, ['utt_id', 'feat_path', 'x_len', 'x_dim', 'text', 'token_id', 'y_len', 'y_dim']]

        # Sort csv records
        if shuffle:
            self.df = df.reindex(np.random.permutation(df.index))
        else:
            self.df = df.sort_values(by='utt_id', ascending=True)

        # Concatenate into a single sentence
        concat_ids = []
        for i in list(self.df.index):
            assert self.df['token_id'][i] != ''
            concat_ids += [self.eos] + list(map(int, self.df['token_id'][i].split()))
        concat_ids += [self.eos]
        # NOTE: <sos> and <eos> have the same index

        # Reshape
        nutt = len(concat_ids)
        concat_ids = concat_ids[:nutt // batch_size * batch_size]
        print('Removed %d tokens / %d tokens' % (nutt - len(concat_ids), nutt))
        self.concat_ids = np.array(concat_ids).reshape((batch_size, -1))

    def __len__(self):
        return len(self.concat_ids.reshape((-1,)))

    @property
    def epoch_detail(self):
        # Floating point version of epoch
        return self.epoch + (self.offset / len(self.concat_ids))

    def __next__(self, batch_size=None):
        """Generate each mini-batch.

        Args:
            batch_size (int): the size of mini-batch
        Returns:
            batch (dict):
                ys (list): target labels in the main task of size `[B, L]`
                ylens (list):
                utt_ids (list): file names of input data of size `[B]`
            is_new_epoch (bool): If true, 1 epoch is finished

        """
        is_new_epoch = False

        if batch_size is None:
            batch_size = self.batch_size

        if self.max_epoch is not None and self.epoch >= self.max_epoch:
            raise StopIteration()
        # NOTE: max_epoch == None means infinite loop

        ys = self.concat_ids[:, self.offset:self.offset + self.bptt]
        self.offset += self.bptt

        # Last mini-batch
        if (self.offset + 1) * self.batch_size >= len(self.concat_ids.reshape((-1,))):
            self.offset = 0
            is_new_epoch = True
            self.epoch += 1

            # TODO(hirofumi): add randomness
            # if self.shuffle:
            #     # Shuffle csv records
            #     self.df = self.df.reindex(np.random.permutation(self.df.index))
            #
            #     # Concatenate into a single sentence
            #     concat_ids = [self.eos]
            #     for i in list(self.df.index):
            #         assert self.df['token_id'][i] != ''
            #         concat_ids += list(map(int, self.df['token_id'][i].split())) + [self.eos]
            #
            #     # Truncate
            #     concat_ids = concat_ids[:len(concat_ids) // batch_size * batch_size]
            #     self.concat_ids = np.array(concat_ids).reshape((batch_size, -1))

        return ys, is_new_epoch
