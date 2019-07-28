#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for loading dataset for ASR.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import copy
import kaldiio
import numpy as np
import os
import pandas as pd
import random

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


def count_vocab_size(dict_path):
    vocab_count = 1  # for <blank>
    with codecs.open(dict_path, 'r', 'utf-8') as f:
        for line in f:
            if line.strip() != '':
                vocab_count += 1
    return vocab_count


class Dataset(object):

    def __init__(self, tsv_path, dict_path,
                 unit, batch_size, nlsyms=False, n_epochs=None,
                 is_test=False, min_n_frames=40, max_n_frames=2000,
                 shuffle_bucket=False, sort_by='utt_id',
                 short2long=False, sort_stop_epoch=None, dynamic_batching=False,
                 ctc=False, subsample_factor=1, wp_model=False, corpus='',
                 tsv_path_sub1=False, dict_path_sub1=False, unit_sub1=False,
                 wp_model_sub1=False, ctc_sub1=False, subsample_factor_sub1=1,
                 tsv_path_sub2=False, dict_path_sub2=False, unit_sub2=False,
                 wp_model_sub2=False, ctc_sub2=False, subsample_factor_sub2=1,
                 discourse_aware=False, skip_thought=False):
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
            shuffle_bucket (bool): gather the similar length of utterances and shuffle them
            sort_by (str): sort all utterances in the ascending order
                input: sort by input length
                output: sort by output length
                shuffle: shuffle all utterances
            short2long (bool): sort utterances in the descending order
            sort_stop_epoch (int): After sort_stop_epoch, training will revert
                back to a random order
            dynamic_batching (bool): change batch size dynamically in training
            ctc (bool):
            subsample_factor (int):
            wp_model (): path to the word-piece model for sentencepiece
            corpus (str): name of corpus
            discourse_aware (bool):
            skip_thought (bool):

        """
        super(Dataset, self).__init__()

        self.epoch = 0
        self.iteration = 0
        self.offset = 0

        self.set = os.path.basename(tsv_path).split('.')[0]
        self.is_test = is_test
        self.unit = unit
        self.unit_sub1 = unit_sub1
        self.batch_size = batch_size
        self.max_epoch = n_epochs
        self.shuffle_bucket = shuffle_bucket
        if shuffle_bucket:
            assert sort_by in ['input', 'output']
        self.sort_stop_epoch = sort_stop_epoch
        self.sort_by = sort_by
        assert sort_by in ['input', 'output', 'shuffle', 'utt_id']
        self.dynamic_batching = dynamic_batching
        self.corpus = corpus
        self.discourse_aware = discourse_aware
        self.skip_thought = skip_thought

        self.vocab = count_vocab_size(dict_path)
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
                setattr(self, 'vocab_sub' + str(i), count_vocab_size(dict_path_sub))

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
        df = pd.read_csv(tsv_path, encoding='utf-8', delimiter='\t')
        df = df.loc[:, ['utt_id', 'speaker', 'feat_path',
                                  'xlen', 'xdim', 'text', 'token_id', 'ylen', 'ydim']]
        for i in range(1, 3):
            if locals()['tsv_path_sub' + str(i)]:
                df_sub = pd.read_csv(locals()['tsv_path_sub' + str(i)], encoding='utf-8', delimiter='\t')
                df_sub = df_sub.loc[:, ['utt_id', 'speaker', 'feat_path',
                                        'xlen', 'xdim', 'text', 'token_id', 'ylen', 'ydim']]
                setattr(self, 'df_sub' + str(i), df_sub)
            else:
                setattr(self, 'df_sub' + str(i), None)
        self.input_dim = kaldiio.load_mat(df['feat_path'][0]).shape[-1]

        if corpus == 'swbd':
            df['session'] = df['speaker'].apply(lambda x: str(x).split('-')[0])
        else:
            df['session'] = df['speaker'].apply(lambda x: str(x))

        if discourse_aware or skip_thought:
            max_n_frames = 10000
            min_n_frames = 100

            # Sort by onset
            df = df.assign(prev_utt='')
            if corpus == 'swbd':
                df['onset'] = df['utt_id'].apply(lambda x: int(x.split('_')[-1].split('-')[0]))
            elif corpus == 'csj':
                df['onset'] = df['utt_id'].apply(lambda x: int(x.split('_')[1]))
            elif corpus == 'wsj':
                df['onset'] = df['utt_id'].apply(lambda x: x)
            else:
                raise NotImplementedError
            df = df.sort_values(by=['session', 'onset'], ascending=True)

            # Extract previous utterances
            if not skip_thought:
                # df = df.assign(line_no=list(range(len(df))))
                groups = df.groupby('session').groups
                df['n_session_utt'] = df.apply(
                    lambda x: len([i for i in groups[x['session']]]), axis=1)

                # df['prev_utt'] = df.apply(
                #     lambda x: [df.loc[i, 'line_no']
                #                for i in groups[x['session']] if df.loc[i, 'onset'] < x['onset']], axis=1)
                # df['n_prev_utt'] = df.apply(lambda x: len(x['prev_utt']), axis=1)

        elif is_test and corpus == 'swbd':
            # Sort by onset
            df['onset'] = df['utt_id'].apply(lambda x: int(x.split('_')[-1].split('-')[0]))
            df = df.sort_values(by=['session', 'onset'], ascending=True)

        # Remove inappropriate utterances
        if is_test:
            print('Original utterance num: %d' % len(df))
            n_utts = len(df)
            df = df[df.apply(lambda x: x['ylen'] > 0, axis=1)]
            print('Removed %d empty utterances' % (n_utts - len(df)))
        else:
            print('Original utterance num: %d' % len(df))
            n_utts = len(df)
            df = df[df.apply(lambda x: min_n_frames <= x[
                'xlen'] <= max_n_frames, axis=1)]
            df = df[df.apply(lambda x: x['ylen'] > 0, axis=1)]
            print('Removed %d utterances (threshold)' % (n_utts - len(df)))

            if ctc and subsample_factor > 1:
                n_utts = len(df)
                df = df[df.apply(lambda x: x['ylen'] <= (x['xlen'] // subsample_factor), axis=1)]
                print('Removed %d utterances (for CTC)' % (n_utts - len(df)))

            for i in range(1, 3):
                df_sub = getattr(self, 'df_sub' + str(i))
                ctc_sub = locals()['ctc_sub' + str(i)]
                subsample_factor_sub = locals()['subsample_factor_sub' + str(i)]
                if df_sub is not None:
                    if ctc_sub and subsample_factor_sub > 1:
                        df_sub = df_sub[df_sub.apply(
                            lambda x: x['ylen'] <= (x['xlen'] // subsample_factor_sub), axis=1)]

                    if len(df) != len(df_sub):
                        n_utts = len(df)
                        df = df.drop(df.index.difference(df_sub.index))
                        print('Removed %d utterances (for CTC, sub%d)' % (n_utts - len(df), i))
                        for j in range(1, i + 1):
                            setattr(self, 'df_sub' + str(j),
                                    getattr(self, 'df_sub' + str(j)).drop(getattr(self, 'df_sub' + str(j)).index.difference(df.index)))

            # Re-indexing
            for i in range(1, 3):
                if getattr(self, 'df_sub' + str(i)) is not None:
                    setattr(self, 'df_sub' + str(i), getattr(self, 'df_sub' + str(i)).reset_index())

        # Sort tsv records
        if not is_test:
            if discourse_aware:
                self.utt_offset = 0
                self.n_utt_session_dict = {}
                self.session_offset_dict = {}
                for session_id, ids in sorted(df.groupby('session').groups.items(), key=lambda x: len(x[1])):
                    n_utt = len(ids)
                    # key: n_utt, value: session_id
                    if n_utt not in self.n_utt_session_dict.keys():
                        self.n_utt_session_dict[n_utt] = []
                    self.n_utt_session_dict[n_utt].append(session_id)

                    # key: session_id, value: id for the first utterance in each session
                    self.session_offset_dict[session_id] = ids[0]

                self.n_utt_session_dict_epoch = copy.deepcopy(self.n_utt_session_dict)
                # if discourse_aware == 'state_carry_over':
                #     df = df.sort_values(by=['n_session_utt', 'utt_id'], ascending=short2long)
                # else:
                #     df = df.sort_values(by=['n_prev_utt'], ascending=short2long)
            elif sort_by == 'input':
                df = df.sort_values(by=['xlen'], ascending=short2long)
            elif sort_by == 'output':
                df = df.sort_values(by=['ylen'], ascending=short2long)
            elif sort_by == 'shuffle':
                df = df.reindex(np.random.permutation(self.df.index))

        for i in range(1, 3):
            if getattr(self, 'df_sub' + str(i)) is not None:
                setattr(self, 'df_sub' + str(i),
                        getattr(self, 'df_sub' + str(i)).reindex(df.index).reset_index())

        # Re-indexing
        self.df = df.reset_index()
        self.df_indices = list(self.df.index)

    def __len__(self):
        return len(self.df)

    @property
    def epoch_detail(self):
        """Percentage of the current epoch."""
        return 1 - (len(self.df_indices) / len(self))

    def reset(self):
        """Reset data counter and offset."""
        self.df_indices = list(self.df.index)
        self.offset = 0

    def next(self, batch_size=None):
        """Generate each mini-batch.

        Args:
            batch_size (int): size of mini-batch
        Returns:
            batch (dict):
            is_new_epoch (bool): flag for the end of the current epoch

        """
        if batch_size is None:
            batch_size = self.batch_size

        if self.max_epoch is not None and self.epoch >= self.max_epoch:
            raise StopIteration
            # NOTE: max_epoch == None means infinite loop

        df_indices, is_new_epoch = self.sample_index(batch_size)
        batch = self.make_batch(df_indices)

        if is_new_epoch:
            # shuffle the whole data
            if self.epoch == self.sort_stop_epoch:
                self.sort_by = 'shuffle'
                self.df = self.df.reindex(np.random.permutation(self.df.index))
                for i in range(1, 3):
                    if getattr(self, 'df_sub' + str(i)) is not None:
                        setattr(self, 'df_sub' + str(i),
                                getattr(self, 'df_sub' + str(i)).reindex(self.df.index).reset_index())

                # Re-indexing
                self.df = self.df.reset_index()

            self.reset()
            self.epoch += 1

        return batch, is_new_epoch

    def sample_index(self, batch_size):
        """Sample data indices of mini-batch.

        Args:
            batch_size (int): size of mini-batch
        Returns:
            df_indices (np.ndarray): indices for dataframe
            is_new_epoch (bool): flag for the end of the current epoch

        """
        is_new_epoch = False

        if self.discourse_aware:
            n_utt = min(self.n_utt_session_dict_epoch.keys())
            assert self.utt_offset < n_utt
            df_indices = [self.df[self.session_offset_dict[session_id] + self.utt_offset:self.session_offset_dict[session_id] + self.utt_offset + 1].index[0]
                          for session_id in self.n_utt_session_dict_epoch[n_utt][:batch_size]]

            self.utt_offset += 1
            if self.utt_offset == n_utt:
                if len(self.n_utt_session_dict_epoch[n_utt][batch_size:]) > 0:
                    self.n_utt_session_dict_epoch[n_utt] = self.n_utt_session_dict_epoch[n_utt][batch_size:]
                else:
                    self.n_utt_session_dict_epoch.pop(n_utt)
                self.utt_offset = 0

                # reset for the new epoch
                if len(self.n_utt_session_dict_epoch.keys()) == 0:
                    self.n_utt_session_dict_epoch = copy.deepcopy(self.n_utt_session_dict)
                    is_new_epoch = True

        else:
            if len(self.df_indices) > batch_size:
                if self.shuffle_bucket:
                    # Sample offset randomly
                    offset = random.sample(self.df_indices, 1)[0]
                    df_indices_offset = self.df_indices.index(offset)
                else:
                    offset = self.offset

                # Change batch size dynamically
                if self.sort_by is not None:
                    min_xlen = self.df[offset:offset + 1]['xlen'].values[0]
                    min_ylen = self.df[offset:offset + 1]['ylen'].values[0]
                    batch_size = self.set_batch_size(batch_size, min_xlen, min_ylen)

                if self.shuffle_bucket:
                    if len(self.df_indices[df_indices_offset:]) < batch_size:
                        df_indices = self.df_indices[df_indices_offset:][:]
                    else:
                        df_indices = self.df_indices[df_indices_offset:df_indices_offset + batch_size][:]
                else:
                    df_indices = list(self.df[offset:offset + batch_size].index)
                    self.offset += len(df_indices)
            else:
                # Last mini-batch
                df_indices = self.df_indices[:]
                is_new_epoch = True

                # Change batch size dynamically
                if self.sort_by is not None:
                    min_xlen = self.df[df_indices[0]:df_indices[0] + 1]['xlen'].values[0]
                    min_ylen = self.df[df_indices[0]:df_indices[0] + 1]['ylen'].values[0]
                    batch_size = self.set_batch_size(batch_size, min_xlen, min_ylen)
                    # Remove the rest
                    df_indices = df_indices[:batch_size]

            # Shuffle uttrances in mini-batch
            df_indices = random.sample(df_indices, len(df_indices))

            for i in df_indices:
                self.df_indices.remove(i)

        return df_indices, is_new_epoch

    def make_batch(self, df_indices):
        """Create mini-batch per step.

        Args:
            df_indices (np.ndarray): indices for dataframe
        Returns:
            batch (dict):
                xs (list): input data of size `[T, input_dim]`
                xlens (list): lengths of xs
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

        # outputs
        if self.is_test:
            ys = [self.token2idx[0](self.df['text'][i]) for i in df_indices]
        else:
            ys = [list(map(int, str(self.df['token_id'][i]).split())) for i in df_indices]

        ys_hist = [[] for _ in range(len(df_indices))]
        if self.discourse_aware:
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
                    ys_prev += [[]]
                    text_prev += ['']  # first utterance
                if i + 1 in self.df.index and self.df['session'][i + 1] == self.df['session'][i]:
                    ys_next += [list(map(int, str(self.df['token_id'][i + 1]).split()))]
                    text_next += [self.df['text'][i + 1]]
                else:
                    ys_next += [[]]  # last utterance
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

    def set_batch_size(self, batch_size, min_xlen, min_ylen):
        if not self.dynamic_batching:
            return batch_size

        if min_xlen <= 800:
            pass
        elif min_xlen <= 1600 or 80 < min_ylen <= 100:
            batch_size = int(batch_size / 2)
        else:
            batch_size = int(batch_size / 4)

        if batch_size < 1:
            batch_size = 1

        return batch_size
