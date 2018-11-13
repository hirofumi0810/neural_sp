#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for loading dataset for the RNNLM.
   In this class, all data will be loaded at each step.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import os
import pandas as pd
import random

from neural_sp.datasets.base import Base
from neural_sp.datasets.token_converter.character import Char2idx
from neural_sp.datasets.token_converter.character import Idx2char
from neural_sp.datasets.token_converter.word import Idx2word
from neural_sp.datasets.token_converter.word import Word2idx

random.seed(1)
np.random.seed(1)

logger = logging.getLogger('training')


class Dataset(Base):

    def __init__(self, csv_path, dict_path, label_type, batch_size, bptt, eos,
                 max_epoch=None, is_test=False, shuffle=False):
        """A class for loading dataset.

        Args:
            csv_path (str):
            dict_path (str):
            label_type (str): word or wp or char or phone
            batch_size (int): the size of mini-batch
            bptt (int):
            eos (int):
            max_epoch (int): the max epoch. None means infinite loop.
            is_test (bool):
            shuffle (bool): if True, shuffle utterances.
                This is disabled when sort_by_input_length is True.

        """
        super(Dataset, self).__init__()

        self.set = os.path.basename(csv_path).split('.')[0]
        self.is_test = is_test
        self.label_type = label_type
        self.batch_size = batch_size
        self.bptt = bptt
        self.eos = eos
        self.max_epoch = max_epoch
        self.shuffle = shuffle
        self.num_classes = self.count_vocab_size(dict_path)

        # Set index converter
        if label_type in ['word', 'wp']:
            self.idx2word = Idx2word(dict_path)
            self.word2idx = Word2idx(dict_path)
        elif label_type == 'char':
            self.idx2char = Idx2char(dict_path)
            self.char2idx = Char2idx(dict_path)
        else:
            raise ValueError(label_type)

        # Load dataset csv file
        df = pd.read_csv(csv_path, encoding='utf-8')
        df = df.loc[:, ['utt_id', 'feat_path', 'x_len', 'x_dim', 'text', 'token_id', 'y_len', 'y_dim']]

        # Sort csv records
        if shuffle:
            self.df = df.reindex(np.random.permutation(df.index))
        else:
            self.df = df.sort_values(by='utt_id', ascending=True)

        # Concatenate into a single sentence
        concat_ids = [eos]
        for i in list(self.df.index):
            assert self.df['token_id'][i] != ''
            concat_ids += list(map(int, self.df['token_id'][i].split())) + [eos]

        # Truncate
        concat_ids = concat_ids[:len(concat_ids) // batch_size * batch_size]
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
            ys (list):
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
