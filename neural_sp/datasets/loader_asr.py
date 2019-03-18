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
from utils import kaldi_io

np.random.seed(1)


class Dataset(Base):

    def __init__(self, tsv_path, dict_path,
                 unit, batch_size, n_epochs=None,
                 is_test=False, min_n_frames=40, max_n_frames=2000,
                 shuffle=False, sort_by_input_length=False,
                 short2long=False, sort_stop_epoch=None,
                 n_ques=None, dynamic_batching=False,
                 ctc=False, subsample_factor=1,
                 wp_model=False, concat_n_utterances=1, prev_n_tokens=0,
                 tsv_path_sub1=False, dict_path_sub1=False, unit_sub1=False,
                 wp_model_sub1=False,
                 ctc_sub1=False, subsample_factor_sub1=1,
                 wp_model_sub2=False,
                 tsv_path_sub2=False, dict_path_sub2=False, unit_sub2=False,
                 ctc_sub2=False, subsample_factor_sub2=1,
                 wp_model_sub3=False,
                 tsv_path_sub3=False, dict_path_sub3=False, unit_sub3=False,
                 ctc_sub3=False, subsample_factor_sub3=1):
        """A class for loading dataset.

        Args:
            tsv_path (str):
            dict_path (str):
            unit (str): word or wp or char or phone or word_char
            batch_size (int): the size of mini-batch
            n_epochs (int): the max epoch. None means infinite loop.
            is_test (bool):
            min_n_frames (int): Exclude utteraces shorter than this value
            max_n_frames (int): Exclude utteraces longer than this value
            shuffle (bool): if True, shuffle utterances.
                This is disabled when sort_by_input_length is True.
            sort_by_input_length (bool): if True, sort all utterances in the ascending order
            short2long (bool): if True, sort utteraces in the descending order
            sort_stop_epoch (int): After sort_stop_epoch, training will revert
                back to a random order
            n_ques (int): the number of elements to enqueue
            dynamic_batching (bool): if True, batch size will be chainged
                dynamically in training
            ctc (bool):
            subsample_factor (int):
            wp_model ():
            concat_n_utterances (int):
            prev_n_tokens (int):

        """
        super(Dataset, self).__init__()

        self.set = os.path.basename(tsv_path).split('.')[0]
        self.is_test = is_test
        self.unit = unit
        self.unit_sub1 = unit_sub1
        self.batch_size = batch_size
        self.max_epoch = n_epochs
        self.shuffle = shuffle
        self.sort_by_input_length = sort_by_input_length
        self.sort_stop_epoch = sort_stop_epoch
        self.n_ques = n_ques
        self.dynamic_batching = dynamic_batching
        self.concat_n_utterances = concat_n_utterances
        self.prev_n_tokens = prev_n_tokens
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

        if dict_path_sub1:
            self.vocab_sub1 = self.count_vocab_size(dict_path_sub1)

            # Set index converter
            if unit_sub1:
                if unit_sub1 == 'wp':
                    self.idx2wp_sub1 = Idx2wp(dict_path_sub1, wp_model_sub1)
                    self.wp2idx_sub1 = Wp2idx(dict_path_sub1, wp_model_sub1)
                elif unit_sub1 == 'char':
                    self.idx2char_sub1 = Idx2char(dict_path_sub1)
                    self.char2idx_sub1 = Char2idx(dict_path_sub1)
                elif 'phone' in unit_sub1:
                    self.idx2phone_sub1 = Idx2phone(dict_path_sub1)
                    self.phone2idx_sub1 = Phone2idx(dict_path_sub1)
                else:
                    raise ValueError(unit_sub1)
        else:
            self.vocab_sub1 = -1

        if dict_path_sub2:
            self.vocab_sub2 = self.count_vocab_size(dict_path_sub2)

            # Set index converter
            if unit_sub2:
                if unit_sub2 == 'wp':
                    self.idx2wp_sub2 = Idx2wp(dict_path_sub2, wp_model_sub2)
                    self.wp2idx_sub2 = Wp2idx(dict_path_sub2, wp_model_sub2)
                elif unit_sub2 == 'char':
                    self.idx2char_sub2 = Idx2char(dict_path_sub2)
                    self.char2idx_sub2 = Char2idx(dict_path_sub2)
                elif 'phone' in unit_sub2:
                    self.idx2phone_sub2 = Idx2phone(dict_path_sub2)
                    self.phone2idx_sub2 = Phone2idx(dict_path_sub2)
                else:
                    raise ValueError(unit_sub2)
        else:
            self.vocab_sub2 = -1

        if dict_path_sub3:
            self.vocab_sub3 = self.count_vocab_size(dict_path_sub3)

            # Set index converter
            if unit_sub3:
                if unit_sub3 == 'wp':
                    self.idx2wp_sub3 = Idx2wp(dict_path_sub3, wp_model_sub3)
                    self.wp2idx_sub3 = Wp2idx(dict_path_sub3, wp_model_sub3)
                elif unit_sub3 == 'char':
                    self.idx2char_sub3 = Idx2char(dict_path_sub3)
                    self.char2idx_sub3 = Char2idx(dict_path_sub3)
                elif 'phone' in unit_sub3:
                    self.idx2phone_sub3 = Idx2phone(dict_path_sub3)
                    self.phone2idx_sub3 = Phone2idx(dict_path_sub3)
                else:
                    raise ValueError(unit_sub3)
        else:
            self.vocab_sub3 = -1

        # Load dataset csv file
        self.df = pd.read_csv(tsv_path, encoding='utf-8', delimiter='\t')
        self.df = self.df.loc[:, ['utt_id', 'speaker', 'feat_path',
                                  'xlen', 'xdim', 'text', 'token_id', 'ylen', 'ydim']]
        for i in range(1, 4):
            if locals()['tsv_path_sub' + str(i)]:
                df_sub = pd.read_csv(locals()['tsv_path_sub' + str(i)], encoding='utf-8', delimiter='\t')
                df_sub = df_sub.loc[:, ['utt_id', 'speaker', 'feat_path',
                                        'xlen', 'xdim', 'text', 'token_id', 'ylen', 'ydim']]
                setattr(self, 'df_sub' + str(i), df_sub)
            else:
                setattr(self, 'df_sub' + str(i), None)

        if concat_n_utterances > 1:
            max_n_frames = 10000
            min_n_frames = 1

        # Remove inappropriate utteraces
        if self.is_test:
            print('Original utterance num: %d' % len(self.df))
            n_utts = len(self.df)
            self.df = self.df[self.df.apply(lambda x: x['ylen'] > 0, axis=1)]
            print('Removed %d empty utterances' % (n_utts - len(self.df)))
        else:
            print('Original utterance num: %d' % len(self.df))
            n_utts = len(self.df)
            self.df = self.df[self.df.apply(lambda x: min_n_frames <= x['xlen'] <= max_n_frames, axis=1)]
            self.df = self.df[self.df.apply(lambda x: x['ylen'] > 0, axis=1)]
            print('Removed %d utterances (threshold)' % (n_utts - len(self.df)))

            if ctc and subsample_factor > 1:
                n_utts = len(self.df)
                self.df = self.df[self.df.apply(lambda x: x['ylen'] <= x['xlen'] // subsample_factor, axis=1)]
                print('Removed %d utterances (for CTC)' % (n_utts - len(self.df)))

            for i in range(1, 4):
                df_sub = getattr(self, 'df_sub' + str(i))
                ctc_sub = locals()['ctc_sub' + str(i)]
                subsample_factor_sub = locals()['subsample_factor_sub' + str(i)]
                if df_sub is not None:
                    if ctc_sub and subsample_factor_sub > 1:
                        df_sub = df_sub[df_sub.apply(lambda x: x['ylen'] <= x['xlen'] // subsample_factor_sub, axis=1)]

                    if len(self.df) != len(df_sub):
                        n_utts = len(self.df)
                        self.df = self.df.drop(self.df.index.difference(df_sub.index))
                        print('Removed %d utterances (for CTC, sub%d)' % (n_utts - len(self.df), i))
                        for j in range(1, i + 1):
                            setattr(self, 'df_sub' + str(j), getattr(self, 'df_sub' + str(j)
                                                                     ).drop(getattr(self, 'df_sub' + str(j)).index.difference(self.df.index)))

        # Sort csv records
        if sort_by_input_length:
            self.df = self.df.sort_values(by='xlen', ascending=short2long)
        else:
            if shuffle:
                self.df = self.df.reindex(np.random.permutation(self.df.index))
            else:
                self.df = self.df.sort_values(by='utt_id', ascending=True)

        self.rest = set(list(self.df.index))
        self.input_dim = kaldi_io.read_mat(self.df['feat_path'][0]).shape[-1]

    def make_batch(self, utt_indices):
        """Create mini-batch per step.

        Args:
            utt_indices (np.ndarray):
        Returns:
            batch (dict):
                xs (list): input data of size `[B, T, input_dim]`
                xlens (list):
                ys (list): target labels in the main task of size `[B, L]`
                ylens (list):
                ys_sub1 (list): target labels in the 1st auxiliary task of size `[B, L_sub1]`
                ylens_sub1 (list):
                ys_sub2 (list): target labels in the 2nd auxiliary task of size `[B, L_sub2]`
                ylens_sub2 (list):
                ys_sub3 (list): target labels in the 3rd auxiliary task of size `[B, L_sub3]`
                ylens_sub3 (list):
                utt_ids (list): file names of input data of size `[B]`

        """
        # input
        xs = [kaldi_io.read_mat(self.df['feat_path'][i]) for i in utt_indices]
        xlens = [self.df['xlen'][i] for i in utt_indices]

        # output
        if self.is_test:
            ys = [self.df['text'][i] for i in utt_indices]
        else:
            ys = [list(map(int, str(self.df['token_id'][i]).split())) for i in utt_indices]
        ylens = [self.df['ylen'][i] for i in utt_indices]
        text = [self.df['text'][i] for i in utt_indices]

        if self.df_sub1 is not None:
            if self.is_test:
                ys_sub1 = [self.df_sub1['text'][i] for i in utt_indices]
            else:
                ys_sub1 = [list(map(int, self.df_sub1['token_id'][i].split())) for i in utt_indices]
            ylens_sub1 = [self.df_sub1['ylen'][i] for i in utt_indices]
        else:
            ys_sub1, ylens_sub1 = [], []

        if self.df_sub2 is not None:
            if self.is_test:
                ys_sub2 = [self.df_sub2['text'][i] for i in utt_indices]
            else:
                ys_sub2 = [list(map(int, self.df_sub2['token_id'][i].split())) for i in utt_indices]
            ylens_sub2 = [self.df_sub2['ylen'][i] for i in utt_indices]
        else:
            ys_sub2, ylens_sub2 = [], []

        if self.df_sub3 is not None:
            if self.is_test:
                ys_sub3 = [self.df_sub3['text'][i] for i in utt_indices]
            else:
                ys_sub3 = [list(map(int, self.df_sub3['token_id'][i].split())) for i in utt_indices]
            ylens_sub3 = [self.df_sub3['ylen'][i] for i in utt_indices]
        else:
            ys_sub3, ylens_sub3 = [], []

        utt_ids = [self.df['utt_id'][i] for i in utt_indices]
        speakers = [self.df['speaker'][i] for i in utt_indices]

        batch_dict = {'xs': xs, 'xlens': xlens,
                      'ys': ys, 'ylens': ylens,
                      'ys_sub1': ys_sub1, 'ylens_sub1': ylens_sub1,
                      'ys_sub2': ys_sub2, 'ylens_sub2': ylens_sub2,
                      'ys_sub3': ys_sub3, 'ylens_sub3': ylens_sub3,
                      'utt_ids':  utt_ids, 'speakers': speakers,
                      'text': text,
                      'feat_path': [self.df['feat_path'][i] for i in utt_indices]}

        return batch_dict
