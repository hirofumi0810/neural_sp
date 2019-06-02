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
import kaldiio

from neural_sp.datasets.base import Base
# from neural_sp.datasets.parallel import multiprocess
from neural_sp.datasets.token_converter.character import Char2idx
from neural_sp.datasets.token_converter.character import Idx2char
from neural_sp.datasets.token_converter.phone import Idx2phone
from neural_sp.datasets.token_converter.phone import Phone2idx
from neural_sp.datasets.token_converter.word import Idx2word
from neural_sp.datasets.token_converter.word import Word2idx
from neural_sp.datasets.token_converter.wordpiece import Idx2wp
from neural_sp.datasets.token_converter.wordpiece import Wp2idx

np.random.seed(1)


class Dataset(Base):

    def __init__(self, tsv_path, dict_path,
                 unit, batch_size, nlsyms=False, n_epochs=None,
                 is_test=False, min_n_frames=40, max_n_frames=2000,
                 shuffle=False, sort_by_input_length=False,
                 short2long=False, sort_stop_epoch=None,
                 n_ques=None, dynamic_batching=False,
                 ctc=False, subsample_factor=1,
                 wp_model=False, corpus='',
                 tsv_path_sub1=False, dict_path_sub1=False, unit_sub1=False,
                 wp_model_sub1=False,
                 ctc_sub1=False, subsample_factor_sub1=1,
                 wp_model_sub2=False,
                 tsv_path_sub2=False, dict_path_sub2=False, unit_sub2=False,
                 ctc_sub2=False, subsample_factor_sub2=1,
                 contextualize=False, skip_thought=False):
        """A class for loading dataset.

        Args:
            tsv_path (str): path to the dataset tsv file
            dict_path (str): path to the dictionary
            unit (str): word or wp or char or phone or word_char
            batch_size (int): size of mini-batch
            nlsyms (str): path to the non-linguistic symbols file
            n_epochs (int): max epoch. None means infinite loop.
            is_test (bool):
            min_n_frames (int): exclude utterances shorter than this value
            max_n_frames (int): exclude utterances longer than this value
            shuffle (bool): shuffle utterances.
                This is disabled when sort_by_input_length is True.
            sort_by_input_length (bool): sort all utterances in the ascending order
            short2long (bool): sort utterances in the descending order
            sort_stop_epoch (int): After sort_stop_epoch, training will revert
                back to a random order
            n_ques (int): number of elements to enqueue
            dynamic_batching (bool): change batch size dynamically in training
            ctc (bool):
            subsample_factor (int):
            wp_model (): path to the word-piece model for sentencepiece
            corpus (str): name of corpus
            contextualize (bool):
            skip_thought (bool):

        """
        super(Dataset, self).__init__()

        self.set = os.path.basename(tsv_path).split('.')[0]
        self.is_test = is_test
        self.unit = unit
        self.unit_sub1 = unit_sub1
        self.batch_size = batch_size
        self.max_epoch = n_epochs
        self.shuffle = shuffle
        self.sort_stop_epoch = sort_stop_epoch
        self.sort_by_input_length = sort_by_input_length
        self.n_ques = n_ques
        self.dynamic_batching = dynamic_batching
        self.corpus = corpus
        self.contextualize = contextualize
        self.skip_thought = skip_thought

        self.vocab = self.count_vocab_size(dict_path)
        self.eos = 2
        self.pad = 3
        # NOTE: reserved in advance

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

        for i in range(1, 3):
            dict_path_sub = locals()['dict_path_sub' + str(i)]
            wp_model_sub = locals()['wp_model_sub' + str(i)]
            unit_sub = locals()['unit_sub' + str(i)]
            if dict_path_sub:
                setattr(self, 'vocab_sub' + str(i), self.count_vocab_size(dict_path_sub))

                # Set index converter
                if unit_sub:
                    if unit_sub == 'wp':
                        self.idx2token += [Idx2wp(dict_path_sub, wp_model_sub)]
                        self.token2idx += [Wp2idx(dict_path_sub, wp_model_sub)]
                    elif unit_sub == 'char':
                        self.idx2token += [Idx2char(dict_path_sub)]
                        self.token2idx += [Char2idx(dict_path_sub, nlsyms=nlsyms)]
                    elif 'phone' in unit_sub:
                        self.idx2token += [Idx2phone(dict_path_sub)]
                        self.token2idx += [Phone2idx(dict_path_sub)]
                    else:
                        raise ValueError(unit_sub)
            else:
                setattr(self, 'vocab_sub' + str(i), -1)

        # Load dataset tsv file
        self.df = pd.read_csv(tsv_path, encoding='utf-8', delimiter='\t')
        self.df = self.df.loc[:, ['utt_id', 'speaker', 'feat_path',
                                  'xlen', 'xdim', 'text', 'token_id', 'ylen', 'ydim']]
        for i in range(1, 3):
            if locals()['tsv_path_sub' + str(i)]:
                df_sub = pd.read_csv(locals()['tsv_path_sub' + str(i)], encoding='utf-8', delimiter='\t')
                df_sub = df_sub.loc[:, ['utt_id', 'speaker', 'feat_path',
                                        'xlen', 'xdim', 'text', 'token_id', 'ylen', 'ydim']]
                setattr(self, 'df_sub' + str(i), df_sub)
            else:
                setattr(self, 'df_sub' + str(i), None)
        self.input_dim = kaldiio.load_mat(self.df['feat_path'][0]).shape[-1]

        if corpus == 'swbd':
            self.df['session'] = self.df['speaker'].apply(lambda x: str(x).split('-')[0])
            # self.df['session'] = self.df['speaker'].apply(lambda x: str(x))
        else:
            self.df['session'] = self.df['speaker'].apply(lambda x: str(x))

        if contextualize or skip_thought:
            max_n_frames = 10000
            min_n_frames = 100

            # Sort by onset
            self.df = self.df.assign(prev_utt='')
            if corpus == 'swbd':
                self.df['onset'] = self.df['utt_id'].apply(lambda x: int(x.split('_')[-1].split('-')[0]))
            elif corpus == 'csj':
                self.df['onset'] = self.df['utt_id'].apply(lambda x: int(x.split('_')[1]))
            elif corpus == 'wsj':
                self.df['onset'] = self.df['utt_id'].apply(lambda x: x)
            else:
                raise NotImplementedError
            self.df = self.df.sort_values(by=['session', 'onset'], ascending=True)

            # Extract previous utterances
            if not skip_thought and not is_test:
                self.df = self.df.assign(line_no=list(range(len(self.df))))
                groups = self.df.groupby('session').groups  # dict
                self.df['prev_utt'] = self.df.apply(
                    lambda x: [self.df.loc[i, 'line_no']
                               for i in groups[x['session']] if self.df.loc[i, 'onset'] < x['onset']], axis=1)
                self.df['n_prev_utt'] = self.df.apply(lambda x: len(x['prev_utt']), axis=1)

        elif is_test and corpus == 'swbd':
            # Sort by onset
            self.df['onset'] = self.df['utt_id'].apply(lambda x: int(x.split('_')[-1].split('-')[0]))
            self.df = self.df.sort_values(by=['session', 'onset'], ascending=True)

        # Remove inappropriate utterances
        if is_test:
            print('Original utterance num: %d' % len(self.df))
            n_utts = len(self.df)
            self.df = self.df[self.df.apply(lambda x: x['ylen'] > 0, axis=1)]
            print('Removed %d empty utterances' % (n_utts - len(self.df)))
        else:
            print('Original utterance num: %d' % len(self.df))
            n_utts = len(self.df)
            self.df = self.df[self.df.apply(lambda x: min_n_frames <= x[
                                            'xlen'] <= max_n_frames, axis=1)]
            self.df = self.df[self.df.apply(lambda x: x['ylen'] > 0, axis=1)]
            print('Removed %d utterances (threshold)' % (n_utts - len(self.df)))

            if ctc and subsample_factor > 1:
                n_utts = len(self.df)
                self.df = self.df[self.df.apply(lambda x: x['ylen'] <= (x['xlen'] // subsample_factor), axis=1)]
                print('Removed %d utterances (for CTC)' % (n_utts - len(self.df)))

            for i in range(1, 3):
                df_sub = getattr(self, 'df_sub' + str(i))
                ctc_sub = locals()['ctc_sub' + str(i)]
                subsample_factor_sub = locals()['subsample_factor_sub' + str(i)]
                if df_sub is not None:
                    if ctc_sub and subsample_factor_sub > 1:
                        df_sub = df_sub[df_sub.apply(
                            lambda x: x['ylen'] <= (x['xlen'] // subsample_factor_sub), axis=1)]

                    if len(self.df) != len(df_sub):
                        n_utts = len(self.df)
                        self.df = self.df.drop(self.df.index.difference(df_sub.index))
                        print('Removed %d utterances (for CTC, sub%d)' % (n_utts - len(self.df), i))
                        for j in range(1, i + 1):
                            setattr(self, 'df_sub' + str(j),
                                    getattr(self, 'df_sub' + str(j)).drop(getattr(self, 'df_sub' + str(j)).index.difference(self.df.index)))

        # Sort tsv records
        if not is_test:
            if contextualize:
                self.df = self.df.sort_values(by='n_prev_utt', ascending=short2long)
            elif sort_by_input_length:
                self.df = self.df.sort_values(by='xlen', ascending=short2long)
            elif shuffle:
                self.df = self.df.reindex(np.random.permutation(self.df.index))

        self.rest = set(list(self.df.index))

    def make_batch(self, df_indices):
        """Create mini-batch per step.

        Args:
            df_indices (np.ndarray):
        Returns:
            batch (dict):
                xs (list): input data of size `[T, input_dim]`
                xlens (list): lengths of each element in xs
                ys (list): reference labels in the main task of size `[L]`
                ys_sub1 (list): reference labels in the 1st auxiliary task of size `[L_sub1]`
                ys_sub2 (list): reference labels in the 2nd auxiliary task of size `[L_sub2]`
                utt_ids (list): name of each utterance
                speakers (list): name of each speaker
                sessions (list): name of each session

        """
        # inputs
        if self.skip_thought:
            xs = []
        else:
            xs = [kaldiio.load_mat(self.df['feat_path'][i]) for i in df_indices]
            # xs = multiprocess(kaldiio.load_mat, self.df['feat_path'][df_indices], core=4)

        # outputs
        if self.is_test:
            ys = [self.token2idx[0](self.df['text'][i]) for i in df_indices]
        else:
            ys = [list(map(int, str(self.df['token_id'][i]).split())) for i in df_indices]

        ys_hist = [[] for _ in range(len(df_indices))]
        if self.contextualize:
            for j, i in enumerate(df_indices):
                for idx in self.df['prev_utt'][i]:
                    ys_hist[j].append(list(map(int, str(self.df['token_id'][idx]).split())))

        ys_prev, ys_next = [], []
        text_prev, text_next = [], []
        if self.skip_thought:
            for i in df_indices:
                if i - 1 in self.df.index and self.df['session'][i - 1] == self.df['session'][i]:
                    ys_prev += [list(map(int, str(self.df['token_id'][i - 1]).split()))]
                    text_prev += [self.df['text'][i - 1]]
                else:
                    ys_prev += [[self.pad]]
                    text_prev += ['']
                if i + 1 in self.df.index and self.df['session'][i + 1] == self.df['session'][i]:
                    ys_next += [list(map(int, str(self.df['token_id'][i + 1]).split()))]
                    text_next += [self.df['text'][i + 1]]
                else:
                    ys_next += [[self.pad]]
                    text_next += ['']

        ys_sub1 = []
        if self.df_sub1 is not None:
            ys_sub1 = [list(map(int, str(self.df_sub1['token_id'][i]).split())) for i in df_indices]
        elif self.vocab_sub1 > 0 and not self.is_test:
            ys_sub1 = [self.token2idx[1](self.df['text'][i]) for i in df_indices]

        ys_sub2 = []
        if self.df_sub2 is not None:
            ys_sub2 = [list(map(int, str(self.df_sub2['token_id'][i]).split())) for i in df_indices]
        elif self.vocab_sub2 > 0 and not self.is_test:
            ys_sub2 = [self.token2idx[2](self.df['text'][i]) for i in df_indices]

        batch_dict = {
            'xs': xs,
            'xlens': [self.df['xlen'][i] for i in df_indices],
            'ys': ys,
            'ys_hist': ys_hist,
            'ys_sub1': ys_sub1,
            'ys_sub2': ys_sub2,
            'utt_ids': [self.df['utt_id'][i] for i in df_indices],
            'speakers': [self.df['speaker'][i] for i in df_indices],
            'sessions': [self.df['session'][i] for i in df_indices],
            'text': [self.df['text'][i] for i in df_indices],
            'feat_path': [self.df['feat_path'][i] for i in df_indices],  # for plot
            'ys_prev': ys_prev,
            'text_prev': text_prev,
            'ys_next': ys_next,
            'text_next': text_next,
        }

        return batch_dict
