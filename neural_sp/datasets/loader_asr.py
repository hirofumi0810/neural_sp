#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for loading dataset for the CTC and attention-based model.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import kaldi_io
import logging
import numpy as np
import os
import pandas as pd

from neural_sp.datasets.base import Base
from neural_sp.datasets.token_converter.character import Char2idx
from neural_sp.datasets.token_converter.character import Idx2char
from neural_sp.datasets.token_converter.phone import Idx2phone
from neural_sp.datasets.token_converter.phone import Phone2idx
from neural_sp.datasets.token_converter.word import Idx2word
from neural_sp.datasets.token_converter.word import Word2idx
from neural_sp.datasets.token_converter.wordpiece import Idx2wp
from neural_sp.datasets.token_converter.wordpiece import Wp2idx

np.random.seed(1)

logger = logging.getLogger('training')


class Dataset(Base):

    def __init__(self, csv_path, dict_path,
                 unit, batch_size, nepochs=None,
                 is_test=False,  min_nframes=40, max_nframes=2000,
                 shuffle=False, sort_by_input_length=False,
                 short2long=False, sort_stop_epoch=None,
                 nques=None, dynamic_batching=False,
                 ctc=False, subsample_factor=1,
                 skip_speech=False, wp_model=None,
                 csv_path_sub1=None, dict_path_sub1=None, unit_sub1=None,
                 ctc_sub1=False, subsample_factor_sub1=1,
                  csv_path_sub2=None, dict_path_sub2=None, unit_sub2=None,
                  ctc_sub2=False, subsample_factor_sub2=1):
        """A class for loading dataset.

        Args:
            csv_path (str):
            dict_path (str):
            unit (str): word or wp or char or phone or word_char
            batch_size (int): the size of mini-batch
            nepochs (int): the max epoch. None means infinite loop.
            is_test (bool):
            min_nframes (int): Exclude utteraces shorter than this value
            max_nframes (int): Exclude utteraces longer than this value
            shuffle (bool): if True, shuffle utterances.
                This is disabled when sort_by_input_length is True.
            sort_by_input_length (bool): if True, sort all utterances in the ascending order
            short2long (bool): if True, sort utteraces in the descending order
            sort_stop_epoch (int): After sort_stop_epoch, training will revert
                back to a random order
            nques (int): the number of elements to enqueue
            dynamic_batching (bool): if True, batch size will be chainged
                dynamically in training
            ctc (bool):
            subsample_factor (int):
            skip_speech (bool): skip loading speech features
            wp_model ():

        """
        super(Dataset, self).__init__()

        self.set = os.path.basename(csv_path).split('.')[0]
        self.is_test = is_test
        self.unit = unit
        self.unit_sub1 = unit_sub1
        self.batch_size = batch_size
        self.max_epoch = nepochs
        self.shuffle = shuffle
        self.sort_by_input_length = sort_by_input_length
        self.sort_stop_epoch = sort_stop_epoch
        self.nques = nques
        self.dynamic_batching = dynamic_batching
        self.skip_speech = skip_speech
        self.vocab = self.count_vocab_size(dict_path)

        # Set index converter
        if unit in ['word', 'word_char']:
            self.idx2word = Idx2word(dict_path)
            self.word2idx = Word2idx(dict_path, word_char_mix=(unit == 'word_char'))
        elif unit == 'wp':
            self.idx2wp = Idx2wp(dict_path, wp_model)
            self.wp2idx = Wp2idx(dict_path, wp_model)
        elif unit == 'char':
            self.idx2char = Idx2char(dict_path)
            self.char2idx = Char2idx(dict_path)
        elif 'phone' in unit:
            self.idx2phone = Idx2phone(dict_path)
            self.phone2idx = Phone2idx(dict_path)
        else:
            raise ValueError(unit)

        if dict_path_sub1 is not None:
            self.vocab_sub1 = self.count_vocab_size(dict_path_sub1)

            # Set index converter
            if unit_sub1 is not None:
                if unit_sub1 == 'wp':
                    self.idx2word = Idx2word(dict_path_sub1)
                    self.word2idx = Word2idx(dict_path_sub1)
                elif unit_sub1 == 'char':
                    self.idx2char = Idx2char(dict_path_sub1)
                    self.char2idx = Char2idx(dict_path_sub1)
                elif 'phone' in unit_sub1:
                    self.idx2phone = Idx2phone(dict_path_sub1)
                    self.phone2idx = Phone2idx(dict_path_sub1)
                else:
                    raise ValueError(unit_sub1)
        else:
            self.vocab_sub1 = -1

        if dict_path_sub2 is not None:
            self.vocab_sub2 = self.count_vocab_size(dict_path_sub2)

            # Set index converter
            if unit_sub2 is not None:
                if unit_sub2 == 'wp':
                    self.idx2word = Idx2word(dict_path_sub2)
                    self.word2idx = Word2idx(dict_path_sub2)
                elif unit_sub2 == 'char':
                    self.idx2char = Idx2char(dict_path_sub2)
                    self.char2idx = Char2idx(dict_path_sub2)
                elif 'phone' in unit_sub2:
                    self.idx2phone = Idx2phone(dict_path_sub2)
                    self.phone2idx = Phone2idx(dict_path_sub2)
                else:
                    raise ValueError(unit_sub2)
        else:
            self.vocab_sub2 = -1

        # Load dataset csv file
        df = pd.read_csv(csv_path, encoding='utf-8', delimiter=',')
        df = df.loc[:, ['utt_id', 'feat_path', 'x_len', 'x_dim', 'text', 'token_id', 'y_len', 'y_dim']]
        if csv_path_sub1 is not None:
            df_sub1 = pd.read_csv(csv_path_sub1, encoding='utf-8', delimiter=',')
            df_sub1 = df_sub1.loc[:, ['utt_id', 'feat_path', 'x_len', 'x_dim', 'text', 'token_id', 'y_len', 'y_dim']]
        else:
            df_sub1 = None
        if csv_path_sub2 is not None:
            df_sub2 = pd.read_csv(csv_path_sub2, encoding='utf-8', delimiter=',')
            df_sub2 = df_sub2.loc[:, ['utt_id', 'feat_path', 'x_len', 'x_dim', 'text', 'token_id', 'y_len', 'y_dim']]
        else:
            df_sub2 = None

        # Remove inappropriate utteraces
        if not self.is_test:
            logger.info('Original utterance num: %d' % len(df))
            num_utt_org = len(df)

            # Remove by threshold
            df = df[df.apply(lambda x: min_nframes <= x['x_len'] <= max_nframes, axis=1)]
            logger.info('Removed %d utterances (threshold)' % (num_utt_org - len(df)))

            # Rempve for CTC loss calculatioon
            if ctc and subsample_factor > 1:
                logger.info('Checking utterances for CTC...')
                logger.info('Original utterance num: %d' % len(df))
                num_utt_org = len(df)
                df = df[df.apply(lambda x: x['y_len'] <= x['x_len'] // subsample_factor, axis=1)]
                logger.info('Removed %d utterances (for CTC)' % (num_utt_org - len(df)))

            if df_sub1 is not None:
                logger.info('Original utterance num (sub1): %d' % len(df_sub1))
                num_utt_org = len(df_sub1)

                # Remove by threshold
                df_sub1 = df_sub1[df_sub1.apply(lambda x: min_nframes <= x['x_len'] <= max_nframes, axis=1)]
                logger.info('Removed %d utterances (threshold, sub1)' % (num_utt_org - len(df_sub1)))

                # Rempve for CTC loss calculatioon
                if ctc_sub1 and subsample_factor_sub1 > 1:
                    logger.info('Checking utterances for CTC...')
                    logger.info('Original utterance num (sub1): %d' % len(df_sub1))
                    num_utt_org = len(df_sub1)
                    df_sub1 = df_sub1[df_sub1.apply(lambda x: x['y_len'] <= x['x_len'] //
                                                    subsample_factor_sub1, axis=1)]
                    logger.info('Removed %d utterances (for CTC, sub1)' % (num_utt_org - len(df_sub1)))

                # Make up the number
                if len(df) != len(df_sub1):
                    df = df.drop(df.index.difference(df_sub1.index))
                    df_sub1 = df_sub1.drop(df_sub1.index.difference(df.index))

                # TODO(hirofumi): add df_sub2

        # Sort csv records
        if sort_by_input_length:
            df = df.sort_values(by='x_len', ascending=short2long)
        else:
            if shuffle:
                df = df.reindex(np.random.permutation(df.index))
            else:
                df = df.sort_values(by='utt_id', ascending=True)

        self.df = df
        self.df_sub1 = df_sub1
        self.df_sub2 = df_sub2
        self.rest = set(list(df.index))
        self.input_dim = kaldi_io.read_mat(self.df['feat_path'][0]).shape[-1]

    def make_batch(self, utt_indices):
        """Create mini-batch per step.

        Args:
            utt_indices (np.ndarray):
        Returns:
            batch (dict):
                xs (list): input data of size `[B, T, input_dim]`
                x_lens (list):
                ys (list): target labels in the main task of size `[B, L]`
                y_lens (list):
                ys_sub1 (list): target labels in the 1st sub task of size `[B, L_sub1]`
                y_lens_sub1 (list):
                ys_sub2 (list): target labels in the 2nd sub task of size `[B, L_sub2]`
                y_lens_sub2 (list):
                utt_ids (list): file names of input data of size `[B]`

        """
        # input
        if not self.skip_speech:
            xs = [kaldi_io.read_mat(self.df['feat_path'][i]) for i in utt_indices]
            x_lens = [self.df['x_len'][i] for i in utt_indices]
        else:
            xs, x_lens = [], []

        # output
        if self.is_test:
            ys = [self.df['text'][i].encode('utf-8') for i in utt_indices]
        else:
            ys = [list(map(int, self.df['token_id'][i].split())) for i in utt_indices]
        y_lens = [self.df['y_len'][i] for i in utt_indices]
        text = [self.df['text'][i].encode('utf-8') for i in utt_indices]

        if self.df_sub1 is not None:
            if self.is_test:
                ys_sub1 = [self.df_sub1['text'][i].encode('utf-8') for i in utt_indices]
            else:
                ys_sub1 = [list(map(int, self.df_sub1['token_id'][i].split())) for i in utt_indices]
            y_lens_sub1 = [self.df_sub1['y_len'][i] for i in utt_indices]
        else:
            ys_sub1, y_lens_sub1 = [], []

        if self.df_sub2 is not None:
            if self.is_test:
                ys_sub2 = [self.df_sub2['text'][i].encode('utf-8') for i in utt_indices]
            else:
                ys_sub2 = [list(map(int, self.df_sub2['token_id'][i].split())) for i in utt_indices]
            y_lens_sub2 = [self.df_sub2['y_len'][i] for i in utt_indices]
        else:
            ys_sub2, y_lens_sub2 = [], []

        utt_ids = [self.df['utt_id'][i].encode('utf-8') for i in utt_indices]

        return {'xs': xs, 'x_lens': x_lens,
                'ys': ys, 'y_lens': y_lens,
                'ys_sub1': ys_sub1, 'y_lens_sub1': y_lens_sub1,
                'ys_sub2': ys_sub2, 'y_lens_sub2': y_lens_sub2,
                'utt_ids':  utt_ids, 'text': text,
                'feat_path': [self.df['feat_path'][i] for i in utt_indices]}
