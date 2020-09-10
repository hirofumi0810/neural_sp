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

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from neural_sp.datasets.token_converter.character import Char2idx
from neural_sp.datasets.token_converter.character import Idx2char
from neural_sp.datasets.token_converter.phone import Idx2phone
from neural_sp.datasets.token_converter.phone import Phone2idx
from neural_sp.datasets.token_converter.word import Idx2word
from neural_sp.datasets.token_converter.word import Word2idx
from neural_sp.datasets.token_converter.wordpiece import Idx2wp
from neural_sp.datasets.token_converter.wordpiece import Wp2idx

from neural_sp.datasets.utils import count_vocab_size
from neural_sp.datasets.utils import discourse_bucketing
from neural_sp.datasets.utils import set_batch_size
from neural_sp.datasets.utils import shuffle_bucketing

random.seed(1)
np.random.seed(1)


def build_dataloader(args, tsv_path, batch_size, n_epochs=1e10, is_test=False,
                     sort_by='utt_id', short2long=False, sort_stop_epoch=1e10,
                     tsv_path_sub1=False, tsv_path_sub2=False,
                     first_n_utterances=-1):

    dataset = CustomDataset(corpus=args.corpus,
                            tsv_path=tsv_path,
                            tsv_path_sub1=tsv_path_sub1,
                            tsv_path_sub2=tsv_path_sub2,
                            dict_path=args.dict,
                            dict_path_sub1=args.dict_sub1,
                            dict_path_sub2=args.dict_sub2,
                            nlsyms=args.nlsyms,
                            unit=args.unit,
                            unit_sub1=args.unit_sub1,
                            unit_sub2=args.unit_sub2,
                            wp_model=args.wp_model,
                            wp_model_sub1=args.wp_model_sub1,
                            wp_model_sub2=args.wp_model_sub2,
                            min_n_frames=args.min_n_frames,
                            max_n_frames=args.max_n_frames,
                            subsample_factor=args.subsample_factor,
                            subsample_factor_sub1=args.subsample_factor_sub1,
                            subsample_factor_sub2=args.subsample_factor_sub2,
                            ctc=args.ctc_weight > 0,
                            ctc_sub1=args.ctc_weight_sub1 > 0,
                            ctc_sub2=args.ctc_weight_sub2 > 0,
                            sort_by=sort_by,
                            short2long=short2long,
                            is_test=is_test)

    batch_sampler = CustomBatchSampler(df=dataset.df,  # filtered
                                       df_sub1=dataset.df_sub1,  # filtered
                                       df_sub2=dataset.df_sub2,  # filtered
                                       batch_size=args.batch_size,
                                       dynamic_batching=args.dynamic_batching,
                                       shuffle_bucket=args.shuffle_bucket,
                                       sort_stop_epoch=args.sort_stop_epoch,
                                       discourse_aware=args.discourse_aware,)

    dataloader = CustomDataLoader(dataset=dataset,
                                  batch_sampler=batch_sampler,
                                  n_epochs=n_epochs,
                                  collate_fn=lambda x: x[0],
                                  num_workers=1,
                                  #   num_workers=2,
                                  pin_memory=False,
                                  #   pin_memory=True,
                                  )

    return dataloader


class CustomDataLoader(DataLoader):

    def __init__(self, dataset, batch_sampler, n_epochs,
                 num_workers=0, collate_fn=None, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):

        super().__init__(dataset=dataset,
                         #  batch_size=batch_size,
                         #  shuffle=shuffle,
                         #  sampler=sampler,
                         batch_sampler=batch_sampler,
                         num_workers=num_workers,
                         collate_fn=collate_fn,
                         pin_memory=pin_memory,
                         drop_last=drop_last,
                         timeout=timeout,
                         worker_init_fn=worker_init_fn)

        self.input_dim = dataset._input_dim
        self.vocab = dataset._vocab
        self.vocab_sub1 = dataset._vocab_sub1
        self.vocab_sub2 = dataset._vocab_sub2
        self.corpus = dataset._corpus
        self.set = dataset._set
        self.unit = dataset._unit
        self.unit_sub1 = dataset._unit_sub1
        self.unit_sub2 = dataset._unit_sub2
        self.idx2token = dataset.idx2token
        self.token2idx = dataset.token2idx

        self.epoch = 0
        self.n_epochs = n_epochs
        self.is_new_epoch = False

    def __len__(self):
        return len(self.dataset.df)

    def __iter__(self):  # hacky
        return self

    def next(self, batch_size=None):  # hacky
        return self.__next__(batch_size)

    def __next__(self, batch_size=None):  # hacky
        """Generate each mini-batch.

        Args:
            batch_size (int): size of mini-batch
        Returns:
            mini_batch (dict):
            is_new_epoch (bool): flag for the end of the current epoch

        """
        if self.epoch >= self.n_epochs:
            raise StopIteration

        indices, self.is_new_epoch = self.batch_sampler.sample_index(batch_size)

        if self.is_new_epoch:
            # shuffle the whole data per epoch
            if self.epoch + 1 == self.batch_sampler.sort_stop_epoch:
                self.batch_sampler.df = self.batch_sampler.df.reindex(
                    np.random.permutation(self.batch_sampler.df.index))
                for i in range(1, 3):
                    if getattr(self.batch_sampler, 'df_sub' + str(i)) is not None:
                        setattr(self.batch_sampler, 'df_sub' + str(i),
                                getattr(self.batch_sampler, 'df_sub' + str(i)).reindex(self.batch_sampler.df.index).reset_index())

                # Re-indexing
                self.batch_sampler.df = self.batch_sampler.df.reset_index()

            self.reset()
            # caclulate iteration again after shuffling
            self.batch_sampler.calculate_iteration()
            self.epoch += 1

        return self.dataset.__getitem__(indices), self.is_new_epoch

    @property
    def epoch_detail(self):
        """Percentage of the current epoch."""
        epoch_ratio = self.batch_sampler._offset / len(self.dataset)
        if self.is_new_epoch:
            epoch_ratio = 1.
        return epoch_ratio
        # return self.batch_sampler.iteration / len(self.batch_sampler)

    @property
    def n_frames(self):
        return self.batch_sampler.df['xlen'].sum()

    def reset(self, batch_size=None):
        """Reset data counter and offset.

            Args:
                batch_size (int): size of mini-batch

        """
        self.batch_sampler._reset(batch_size)


class CustomDataset(Dataset):

    def __init__(self, corpus, tsv_path, dict_path, unit, nlsyms, wp_model,
                 is_test, min_n_frames, max_n_frames, sort_by, short2long,
                 tsv_path_sub1, tsv_path_sub2,
                 ctc, ctc_sub1, ctc_sub2,
                 subsample_factor, subsample_factor_sub1, subsample_factor_sub2,
                 dict_path_sub1, dict_path_sub2,
                 unit_sub1, unit_sub2,
                 wp_model_sub1, wp_model_sub2,
                 discourse_aware=False, first_n_utterances=-1):
        """Custom Dataset class.

        Args:
            corpus (str): name of corpus
            tsv_path (str): path to the dataset tsv file
            dict_path (str): path to the dictionary
            unit (str): word/wp/char/phone/word_char
            nlsyms (str): path to the non-linguistic symbols file
            wp_model (): path to the word-piece model for sentencepiece
            is_test (bool):
            min_n_frames (int): exclude utterances shorter than this value
            max_n_frames (int): exclude utterances longer than this value
            sort_by (str): sort all utterances in the ascending order
                input: sort by input length
                output: sort by output length
                shuffle: shuffle all utterances
            short2long (bool): sort utterances in the descending order
            ctc (bool):
            subsample_factor (int):
            discourse_aware (bool): sort in the discourse order
            first_n_utterances (int): evaluate the first N utterances

        """
        super(Dataset, self).__init__()

        self.epoch = 0

        # meta deta accessed by dataloader
        self._corpus = corpus
        self._set = os.path.basename(tsv_path).split('.')[0]
        self._vocab = count_vocab_size(dict_path)
        self._unit = unit
        self._unit_sub1 = unit_sub1
        self._unit_sub2 = unit_sub2

        self.is_test = is_test
        self.sort_by = sort_by
        assert sort_by in ['input', 'output', 'shuffle', 'utt_id']
        # if shuffle_bucket:
        #     assert sort_by in ['input', 'output']
        if discourse_aware:
            assert not is_test

        self.subsample_factor = subsample_factor

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
                setattr(self, '_vocab_sub' + str(i), count_vocab_size(dict_path_sub))

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
                setattr(self, '_vocab_sub' + str(i), -1)

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
        self._input_dim = kaldiio.load_mat(df['feat_path'][0]).shape[-1]

        # Remove inappropriate utterances
        print('Original utterance num: %d' % len(df))
        n_utts = len(df)
        if is_test or discourse_aware:
            df = df[df.apply(lambda x: x['ylen'] > 0, axis=1)]
            print('Removed %d empty utterances' % (n_utts - len(df)))
            if first_n_utterances > 0:
                n_utts = len(df)
                df = df[df.apply(lambda x: x['ylen'] > 0, axis=1)]
                df = df.truncate(before=0, after=first_n_utterances - 1)
                print('Select first %d utterances' % len(df))
        else:
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

    def __len__(self):
        return len(self.df)

    @property
    def n_frames(self):
        return self.df['xlen'].sum()

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
        elif self._vocab_sub1 > 0 and not self.is_test:
            ys_sub1 = [self.token2idx[1](self.df['text'][i]) for i in indices]

        # sub2 outputs
        ys_sub2 = []
        if self.df_sub2 is not None:
            ys_sub2 = [list(map(int, str(self.df_sub2['token_id'][i]).split())) for i in indices]
        elif self._vocab_sub2 > 0 and not self.is_test:
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


# NOTE: epoch should not be counted in BatchSampler
class CustomBatchSampler(BatchSampler):

    def __init__(self, df, batch_size, dynamic_batching,
                 shuffle_bucket, discourse_aware, sort_stop_epoch,
                 df_sub1=None, df_sub2=None):
        """Custom BatchSampler.

        Args:

            df (pandas.DataFrame): dataframe for the main task
            batch_size (int): size of mini-batch
            dynamic_batching (bool): change batch size dynamically in training
            shuffle_bucket (bool): gather the similar length of utterances and shuffle them
            discourse_aware (bool): sort in the discourse order
            sort_stop_epoch (int): After sort_stop_epoch, training will revert
                back to a random order
            df_sub1 (pandas.DataFrame): dataframe for the first sub task
            df_sub2 (pandas.DataFrame): dataframe for the second sub task

        """
        self.df = df
        self.df_sub1 = df_sub1
        self.df_sub2 = df_sub2
        self.batch_size = batch_size

        self.dynamic_batching = dynamic_batching
        self.shuffle_bucket = shuffle_bucket
        self.sort_stop_epoch = sort_stop_epoch
        self.discourse_aware = discourse_aware

        self._offset = 0

        if discourse_aware:
            self.indices_buckets = discourse_bucketing(self.df, batch_size)
            self._iteration = len(self.indices_buckets)
        elif shuffle_bucket:
            self.indices_buckets = shuffle_bucketing(self.df, batch_size, self.dynamic_batching)
            self._iteration = len(self.indices_buckets)
        else:
            self.indices = list(self.df.index)
            # calculate #iteration in advance
            self.calculate_iteration()

    def __len__(self):
        self._iteration

    def calculate_iteration(self):
        self._iteration = 0
        is_new_epoch = False
        while not is_new_epoch:
            _, is_new_epoch = self.sample_index(self.batch_size)
            self._iteration += 1
        self._reset()

    def _reset(self, batch_size=None):
        """Reset data counter and offset.

            Args:
                batch_size (int): size of mini-batch

        """
        if batch_size is None:
            batch_size = self.batch_size

        if self.discourse_aware:
            self.indices_buckets = discourse_bucketing(self.df, batch_size)
        elif self.shuffle_bucket:
            self.indices_buckets = shuffle_bucketing(self.df, batch_size, self.dynamic_batching)
        else:
            self.indices = list(self.df.index)
        self._offset = 0

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
            indices = self.indices_buckets.pop(0)
            self._offset += len(indices)
            is_new_epoch = (len(self.indices_buckets) == 0)

        elif self.shuffle_bucket:
            indices = self.indices_buckets.pop(0)
            self._offset += len(indices)
            is_new_epoch = (len(self.indices_buckets) == 0)

            # Shuffle uttrances in mini-batch
            indices = random.sample(indices, len(indices))

        else:
            if batch_size is None:
                batch_size = self.batch_size

            if len(self.indices) > batch_size:
                # Change batch size dynamically
                min_xlen = self.df[self._offset:self._offset + 1]['xlen'].values[0]
                min_ylen = self.df[self._offset:self._offset + 1]['ylen'].values[0]
                batch_size = set_batch_size(batch_size, min_xlen, min_ylen,
                                            self.dynamic_batching)

                indices = list(self.df[self._offset:self._offset + batch_size].index)
                self._offset += len(indices)
            else:
                # Last mini-batch
                indices = self.indices[:]
                self._offset = len(self.df)
                is_new_epoch = True

                # Change batch size dynamically
                min_xlen = self.df[indices[0]:indices[0] + 1]['xlen'].values[0]
                min_ylen = self.df[indices[0]:indices[0] + 1]['ylen'].values[0]
                batch_size = set_batch_size(batch_size, min_xlen, min_ylen,
                                            self.dynamic_batching)

                # Remove the rest
                indices = indices[:batch_size]

            # Shuffle uttrances in mini-batch
            indices = random.sample(indices, len(indices))

            for i in indices:
                self.indices.remove(i)

        return indices, is_new_epoch
