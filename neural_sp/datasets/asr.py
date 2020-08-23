#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for loading dataset for ASR.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

import codecs
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
    with codecs.open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() != '':
                vocab_count += 1
    return vocab_count


class Dataset(object):

    def __init__(self, tsv_path, dict_path,
                 unit, batch_size, n_epochs=1e10,
                 is_test=False, min_n_frames=40, max_n_frames=2000,
                 shuffle_bucket=False, sort_by='utt_id',
                 short2long=False, sort_stop_epoch=1000, dynamic_batching=False,
                 corpus='',
                 tsv_path_sub1=False, tsv_path_sub2=False,
                 dict_path_sub1=False, dict_path_sub2=False, nlsyms=False,
                 unit_sub1=False, unit_sub2=False,
                 wp_model=False, wp_model_sub1=False, wp_model_sub2=False,
                 ctc=False, ctc_sub1=False, ctc_sub2=False,
                 subsample_factor=1, subsample_factor_sub1=1, subsample_factor_sub2=1,
                 discourse_aware=False, first_n_utterances=-1):
        """A class for loading dataset.

        Args:
            tsv_path (str): path to the dataset tsv file
            dict_path (str): path to the dictionary
            unit (str): word/wp/char/phone/word_char
            batch_size (int): size of mini-batch
            nlsyms (str): path to the non-linguistic symbols file
            n_epochs (int): total epochs for training.
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
            first_n_utterances (int): evaluate the first N utterances

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
        self.n_epochs = n_epochs
        self.shuffle_bucket = shuffle_bucket
        if shuffle_bucket:
            assert sort_by in ['input', 'output']
        self.sort_stop_epoch = sort_stop_epoch
        self.sort_by = sort_by
        assert sort_by in ['input', 'output', 'shuffle', 'utt_id']
        self.dynamic_batching = dynamic_batching
        self.corpus = corpus
        self.discourse_aware = discourse_aware
        if discourse_aware:
            assert not is_test

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
        elif unit in ['char']:
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

        # Remove inappropriate utterances
        if is_test or discourse_aware:
            print('Original utterance num: %d' % len(df))
            n_utts = len(df)
            df = df[df.apply(lambda x: x['ylen'] > 0, axis=1)]
            print('Removed %d empty utterances' % (n_utts - len(df)))
            if first_n_utterances > 0:
                n_utts = len(df)
                df = df[df.apply(lambda x: x['ylen'] > 0, axis=1)]
                df = df.truncate(before=0, after=first_n_utterances - 1)
                print('Select first %d utterances' % len(df))
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

        if corpus == 'swbd':
            # 1. serialize
            # df['session'] = df['speaker'].apply(lambda x: str(x).split('-')[0])
            # 2. not serialize
            df['session'] = df['speaker'].apply(lambda x: str(x))
        else:
            df['session'] = df['speaker'].apply(lambda x: str(x))

        # Sort tsv records
        if discourse_aware:
            # Sort by onset (start time)
            df = df.assign(prev_utt='')
            df = df.assign(line_no=list(range(len(df))))
            if corpus == 'swbd':
                df['onset'] = df['utt_id'].apply(lambda x: int(x.split('_')[-1].split('-')[0]))
            elif corpus == 'csj':
                df['onset'] = df['utt_id'].apply(lambda x: int(x.split('_')[1]))
            elif corpus == 'tedlium2':
                df['onset'] = df['utt_id'].apply(lambda x: int(x.split('-')[-2]))
            else:
                raise NotImplementedError(corpus)
            df = df.sort_values(by=['session', 'onset'], ascending=True)

            # Extract previous utterances
            groups = df.groupby('session').groups
            df['prev_utt'] = df.apply(
                lambda x: [df.loc[i, 'line_no']
                           for i in groups[x['session']] if df.loc[i, 'onset'] < x['onset']], axis=1)
            df['n_prev_utt'] = df.apply(lambda x: len(x['prev_utt']), axis=1)
            df['n_utt_in_session'] = df.apply(
                lambda x: len([i for i in groups[x['session']]]), axis=1)
            df = df.sort_values(by=['n_utt_in_session'], ascending=short2long)

            # NOTE: this is used only when LM is trained with seliarize: true
            # if is_test and corpus == 'swbd':
            #     # Sort by onset
            #     df['onset'] = df['utt_id'].apply(lambda x: int(x.split('_')[-1].split('-')[0]))
            #     df = df.sort_values(by=['session', 'onset'], ascending=True)

        elif not is_test:
            if sort_by == 'input':
                df = df.sort_values(by=['xlen'], ascending=short2long)
            elif sort_by == 'output':
                df = df.sort_values(by=['ylen'], ascending=short2long)
            elif sort_by == 'shuffle':
                df = df.reindex(np.random.permutation(self.df.index))

        # Re-indexing
        if discourse_aware:
            self.df = df
            for i in range(1, 3):
                if getattr(self, 'df_sub' + str(i)) is not None:
                    setattr(self, 'df_sub' + str(i),
                            getattr(self, 'df_sub' + str(i)).reindex(df.index))
        else:
            self.df = df.reset_index()
            for i in range(1, 3):
                if getattr(self, 'df_sub' + str(i)) is not None:
                    setattr(self, 'df_sub' + str(i),
                            getattr(self, 'df_sub' + str(i)).reindex(df.index).reset_index())

        if discourse_aware:
            self.df_indices_buckets = self.discourse_bucketing(batch_size)
        elif shuffle_bucket:
            self.df_indices_buckets = self.shuffle_bucketing(batch_size)
        else:
            self.df_indices = list(self.df.index)

    def __len__(self):
        return len(self.df)

    @property
    def epoch_detail(self):
        """Percentage of the current epoch."""
        return self.offset / len(self)

    @property
    def n_frames(self):
        return self.df['xlen'].sum()

    def reset(self, batch_size=None):
        """Reset data counter and offset.

            Args:
                batch_size (int): size of mini-batch

        """
        if batch_size is None:
            batch_size = self.batch_size

        if self.discourse_aware:
            self.df_indices_buckets = self.discourse_bucketing(batch_size)
        elif self.shuffle_bucket:
            self.df_indices_buckets = self.shuffle_bucketing(batch_size)
        else:
            self.df_indices = list(self.df.index)
        self.offset = 0

    def __iter__(self):
        return self

    def next(self, batch_size):
        return self.__next__(batch_size)

    def __next__(self, batch_size=None):
        """Generate each mini-batch.

        Args:
            batch_size (int): size of mini-batch
        Returns:
            mini_batch (dict):
            is_new_epoch (bool): flag for the end of the current epoch

        """
        if batch_size is None:
            batch_size = self.batch_size

        if self.epoch >= self.n_epochs:
            raise StopIteration

        indices, is_new_epoch = self.sample_index(batch_size)
        mini_batch = self.__getitem__(indices)

        if is_new_epoch:
            # shuffle the whole data
            if self.epoch + 1 == self.sort_stop_epoch:
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

        return mini_batch, is_new_epoch

    def sample_index(self, batch_size):
        """Sample data indices of mini-batch.

        Args:
            batch_size (int): size of mini-batch
        Returns:
            indices (np.ndarray): indices of dataframe in the current mini-batch
            is_new_epoch (bool): flag for the end of the current epoch

        """
        is_new_epoch = False

        if self.discourse_aware:
            indices = self.df_indices_buckets.pop(0)
            self.offset += len(indices)
            is_new_epoch = (len(self.df_indices_buckets) == 0)

        elif self.shuffle_bucket:
            indices = self.df_indices_buckets.pop(0)
            self.offset += len(indices)
            is_new_epoch = (len(self.df_indices_buckets) == 0)

            # Shuffle uttrances in mini-batch
            indices = random.sample(indices, len(indices))
        else:
            if len(self.df_indices) > batch_size:
                # Change batch size dynamically
                min_xlen = self.df[self.offset:self.offset + 1]['xlen'].values[0]
                min_ylen = self.df[self.offset:self.offset + 1]['ylen'].values[0]
                batch_size = self.set_batch_size(batch_size, min_xlen, min_ylen)

                indices = list(self.df[self.offset:self.offset + batch_size].index)
                self.offset += len(indices)
            else:
                # Last mini-batch
                indices = self.df_indices[:]
                self.offset = len(self)
                is_new_epoch = True

                # Change batch size dynamically
                min_xlen = self.df[indices[0]:indices[0] + 1]['xlen'].values[0]
                min_ylen = self.df[indices[0]:indices[0] + 1]['ylen'].values[0]
                batch_size = self.set_batch_size(batch_size, min_xlen, min_ylen)

                # Remove the rest
                indices = indices[:batch_size]

            # Shuffle uttrances in mini-batch
            indices = random.sample(indices, len(indices))

            for i in indices:
                self.df_indices.remove(i)

        return indices, is_new_epoch

    def __getitem__(self, indices):
        """Create mini-batch per step.

        Args:
            indices (np.ndarray): indices of dataframe in the current mini-batch
        Returns:
            mini_batch_dict (dict):
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
        xs = [kaldiio.load_mat(self.df['feat_path'][i]) for i in indices]
        xlens = [self.df['xlen'][i] for i in indices]
        utt_ids = [self.df['utt_id'][i] for i in indices]
        speakers = [self.df['speaker'][i] for i in indices]
        sessions = [self.df['session'][i] for i in indices]
        texts = [self.df['text'][i] for i in indices]
        feat_paths = [self.df['feat_path'][i] for i in indices]

        # main outputs
        if self.is_test:
            ys = [self.token2idx[0](self.df['text'][i]) for i in indices]
        else:
            ys = [list(map(int, str(self.df['token_id'][i]).split())) for i in indices]

        # sub1 outputs
        ys_sub1 = []
        if self.df_sub1 is not None:
            ys_sub1 = [list(map(int, str(self.df_sub1['token_id'][i]).split())) for i in indices]
        elif self.vocab_sub1 > 0 and not self.is_test:
            ys_sub1 = [self.token2idx[1](self.df['text'][i]) for i in indices]

        # sub2 outputs
        ys_sub2 = []
        if self.df_sub2 is not None:
            ys_sub2 = [list(map(int, str(self.df_sub2['token_id'][i]).split())) for i in indices]
        elif self.vocab_sub2 > 0 and not self.is_test:
            ys_sub2 = [self.token2idx[2](self.df['text'][i]) for i in indices]

        mini_batch_dict = {
            'xs': xs,
            'xlens': xlens,
            'ys': ys,
            'ys_sub1': ys_sub1,
            'ys_sub2': ys_sub2,
            'utt_ids': utt_ids,
            'speakers': speakers,
            'sessions': sessions,
            'text': texts,
            'feat_path': feat_paths,  # for plot
        }
        return mini_batch_dict

    def set_batch_size(self, batch_size, min_xlen, min_ylen):
        if not self.dynamic_batching:
            return batch_size

        if min_xlen <= 800:
            pass
        elif min_xlen <= 1600 or 80 < min_ylen <= 100:
            batch_size //= 2
        else:
            batch_size //= 4

        return max(1, batch_size)

    def shuffle_bucketing(self, batch_size):
        df_indices_buckets = []  # list of list
        offset = 0
        while True:
            min_xlen = self.df[offset:offset + 1]['xlen'].values[0]
            min_ylen = self.df[offset:offset + 1]['ylen'].values[0]
            _batch_size = self.set_batch_size(batch_size, min_xlen, min_ylen)
            indices = list(self.df[offset:offset + _batch_size].index)
            df_indices_buckets.append(indices)
            offset += len(indices)
            if offset + _batch_size >= len(self):
                break

        # shuffle buckets
        random.shuffle(df_indices_buckets)
        return df_indices_buckets

    def discourse_bucketing(self, batch_size):
        df_indices_buckets = []  # list of list
        session_groups = [(k, v) for k, v in self.df.groupby('n_utt_in_session').groups.items()]
        if self.shuffle_bucket:
            random.shuffle(session_groups)
        for n_utt, ids in session_groups:
            first_utt_ids = [i for i in ids if self.df['n_prev_utt'][i] == 0]
            for i in range(0, len(first_utt_ids), batch_size):
                first_utt_ids_mb = first_utt_ids[i:i + batch_size]
                for j in range(n_utt):
                    indices = [k + j for k in first_utt_ids_mb]
                    df_indices_buckets.append(indices)

        return df_indices_buckets
