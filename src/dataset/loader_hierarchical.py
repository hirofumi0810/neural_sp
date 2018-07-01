#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for loading dataset for the multitask CTC and attention-based model.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename, join
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger('training')

from src.dataset.base import Base, load_feat
from src.utils.parallel import make_parallel
from src.utils.io.labels.word import Idx2word, Word2idx
from src.utils.io.labels.character import Idx2char, Char2idx
from src.utils.io.labels.phone import Idx2phone, Phone2idx


class Dataset(Base):

    def __init__(self, corpus, data_save_path, model_type,
                 input_freq, use_delta, use_double_delta,
                 data_size, data_type, label_type, label_type_sub,
                 batch_size, max_epoch=None,
                 max_frame_num=2000, min_frame_num=40,
                 shuffle=False, sort_utt=False, reverse=False,
                 sort_stop_epoch=None, tool='htk',
                 num_enque=None, dynamic_batching=False,
                 use_ctc=False, subsampling_factor=1,
                 use_ctc_sub=False, subsampling_factor_sub=1):
        """A class for loading dataset.
        Args:
            corpus (string): the name of corpus
            data_save_path (string): path to saved data
            model_type (string):
            input_freq (int): the number of dimensions of acoustics
            use_delta (bool): if True, use the delta feature
            use_double_delta (bool): if True, use the acceleration feature
            data_size (string):
            data_type (string):
            label_type (string):
            label_type_sub (string):
            batch_size (int): the size of mini-batch
            max_epoch (int): the max epoch. None means infinite loop.
            max_frame_num (int): Exclude utteraces longer than this value
            min_frame_num (int): Exclude utteraces shorter than this value
            shuffle (bool): if True, shuffle utterances.
                This is disabled when sort_utt is True.
            sort_utt (bool): if True, sort all utterances in the ascending order
            reverse (bool): if True, sort utteraces in the descending order
            sort_stop_epoch (int): After sort_stop_epoch, training will revert
                back to a random order
            tool (string): htk or librosa or python_speech_features
            num_enque (int): the number of elements to enqueue
            dynamic_batching (bool): if True, batch size will be chainged
                dynamically in training
            use_ctc (bool):
            subsampling_factor (int):
            use_ctc_sub (bool):
            subsampling_factor_sub (int):
        """
        self.corpus = corpus
        self.model_type = model_type
        self.input_freq = input_freq
        self.use_delta = use_delta
        self.use_double_delta = use_double_delta
        self.data_type = data_type
        self.label_type = label_type
        self.label_type_sub = label_type_sub
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.shuffle = shuffle
        self.sort_utt = sort_utt
        self.sort_stop_epoch = sort_stop_epoch
        self.tool = tool
        self.num_enque = num_enque
        self.dynamic_batching = dynamic_batching

        # Corpus depending
        if corpus in ['csj', 'swbd', 'wsj']:
            self.is_test = True if 'eval' in data_type else False
        elif corpus in ['librispeech', 'timit']:
            self.is_test = True if 'test' in data_type else False
        else:
            raise NotImplementedError

        # TODO: fix this
        if corpus == 'librispeech':
            if data_type == 'train':
                data_type += '_' + data_size

        self.vocab_file_path = join(
            data_save_path, 'vocab', data_size, label_type + '.txt')

        # main task
        if label_type == 'word':
            self.idx2word = Idx2word(self.vocab_file_path)
            self.word2idx = Word2idx(self.vocab_file_path)
        elif 'character' in label_type:
            self.idx2char = Idx2char(
                self.vocab_file_path,
                capital_divide=label_type == 'character_capital_divide')
            self.char2idx = Char2idx(
                self.vocab_file_path,
                capital_divide=label_type == 'character_capital_divide')
        else:
            raise ValueError(label_type)

        # sub task
        self.vocab_file_path_sub = join(
            data_save_path, 'vocab', data_size, label_type_sub + '.txt')
        if 'character' in label_type_sub:
            self.idx2char = Idx2char(
                self.vocab_file_path_sub,
                capital_divide=label_type_sub == 'character_capital_divide')
            self.char2idx = Char2idx(
                self.vocab_file_path_sub,
                capital_divide=label_type_sub == 'character_capital_divide')
        elif 'phone' in label_type_sub:
            self.idx2phone = Idx2phone(self.vocab_file_path_sub)
            self.phone2idx = Phone2idx(self.vocab_file_path_sub)
        else:
            raise NotImplementedError(label_type_sub)

        super(Dataset, self).__init__(vocab_file_path=self.vocab_file_path,
                                      vocab_file_path_sub=self.vocab_file_path_sub)

        # Load dataset file
        dataset_path = join(
            data_save_path, 'dataset', tool, data_size, data_type, label_type + '.csv')
        dataset_path_sub = join(
            data_save_path, 'dataset', tool, data_size, data_type, label_type_sub + '.csv')
        df = pd.read_csv(dataset_path)
        df = df.loc[:, ['frame_num', 'input_path', 'transcript']]
        df_sub = pd.read_csv(dataset_path_sub)
        df_sub = df_sub.loc[:, ['frame_num', 'input_path', 'transcript']]

        # Remove inappropriate utteraces
        if not self.is_test:
            print('Original utterance num (main): %d' % len(df))
            print('Original utterance num (sub): %d' % len(df_sub))
            utt_num_orig = len(df)
            utt_num_orig_sub = len(df_sub)

            # For Switchboard
            if corpus == 'swbd' and 'train' in data_type:
                df = df[df.apply(lambda x: not(len(x['transcript'].split(' '))
                                               <= 3 and x['frame_num'] >= 1000), axis=1)]
                df_sub = df_sub[df_sub.apply(lambda x: not(len(x['transcript'].split(' '))
                                                           <= 24 and x['frame_num'] >= 1000), axis=1)]

            # Remove by threshold
            df = df[df.apply(
                lambda x: min_frame_num <= x['frame_num'] <= max_frame_num, axis=1)]
            df_sub = df_sub[df_sub.apply(
                lambda x: min_frame_num <= x['frame_num'] <= max_frame_num, axis=1)]
            print('Removed utterance num (threshold, main): %d' %
                  (utt_num_orig - len(df)))
            print('Removed utterance num (threshold, sub): %d' %
                  (utt_num_orig_sub - len(df_sub)))

            # Remove for CTC loss calculatioon
            if use_ctc and subsampling_factor > 1:
                print('Checking utterances for CTC (main)')
                utt_num_orig = len(df)
                df = df[df.apply(
                    lambda x: len(x['transcript'].split(' ')) <= x['frame_num'] // subsampling_factor, axis=1)]
                print('Removed utterance num (for CTC, main): %d' %
                      (utt_num_orig - len(df)))
            if use_ctc_sub and subsampling_factor_sub > 1:
                print('Checking utterances for CTC (sub)')
                utt_num_orig_sub = len(df_sub)
                df_sub = df_sub[df_sub.apply(
                    lambda x: len(x['transcript'].split(' ')) <= x['frame_num'] // subsampling_factor_sub, axis=1)]
                print('Removed utterance num (for CTC, sub): %d' %
                      (utt_num_orig_sub - len(df_sub)))

            # Make up the number
            if len(df) != len(df_sub):
                df = df.drop(df.index.difference(df_sub.index))
                df_sub = df_sub.drop(df_sub.index.difference(df.index))

        # Sort paths to input & label
        if sort_utt:
            df = df.sort_values(by='frame_num', ascending=not reverse)
            df_sub = df_sub.sort_values(by='frame_num', ascending=not reverse)
        else:
            df = df.sort_values(by='input_path', ascending=True)
            df_sub = df_sub.sort_values(by='input_path', ascending=True)

        assert len(df) == len(df_sub)

        self.df = df
        self.df_sub = df_sub
        self.rest = set(list(df.index))

        # Setting for each corpus
        if corpus == 'timit':
            # Set path to phones.60-48-39.map
            self.phone_map_path = './conf/phones.60-48-39.map'
        elif corpus == 'swbd':
            if 'eval2000' in data_type:
                self.glm_path = join(data_save_path, 'eval2000', 'glm')
                self.acronyms_map_path = join(
                    data_save_path, 'eval2000', 'acronyms.map')
            else:
                self.glm_path = None
                self.acronyms_map_path = None

    def make_batch(self, data_indices):
        """Create mini-batch per step.
        Args:
            data_indices (np.ndarray):
        Returns:
            batch (dict):
                xs (list): input data of size `[B, T, input_size]`
                ys (list): target labels in the main task of size `[B, L]`
                ys_sub (list): target labels in the sub task of size `[B, L_sub]`
                input_names (list): file names of input data of size `[B]`
        """
        # features
        feats = [load_feat(self.df['input_path'][i]) for i in data_indices]

        if (not self.use_delta) and (not self.use_double_delta):
            xs = [feat[:, :self.input_freq] for feat in feats]
        else:
            xs = []
            for b in range(len(data_indices)):
                feat = feats[b]

                # Append delta and double-delta features
                max_freq = feat.shape[-1] // 3
                # NOTE: the last dim should be the pitch feature
                if self.input_freq < max_freq and (self.input_freq - 1) % 10 == 0:
                    x = [feat[:, :self.input_freq - 1]]
                    x += [feat[:, max_freq: max_freq + 1]]
                    if self.use_delta:
                        x += [feat[:, max_freq:max_freq + self.input_freq - 1]]
                        x += [feat[:, max_freq * 2: max_freq * 2 + 1]]
                    if self.use_double_delta:
                        x += [feat[:, max_freq * 2:max_freq *
                                   2 + self.input_freq - 1]]
                        x += [feat[:, -1].reshape(-1, 1)]
                else:
                    x = [feat[:, :self.input_freq]]
                    if self.use_delta:
                        x += [feat[:, max_freq:max_freq + self.input_freq]]
                    if self.use_double_delta:
                        x += [feat[:, max_freq *
                                   2:max_freq * 2 + self.input_freq]]
                xs += [np.concatenate(x, axis=-1)]

        # transcript
        if self.is_test:
            ys = [self.df['transcript'][i] for i in data_indices]
            ys_sub = [self.df_sub['transcript'][i] for i in data_indices]
            # NOTE: transcript is not tokenized
        else:
            ys = [list(map(int, self.df['transcript'][i].split(' ')))
                  for i in data_indices]
            ys_sub = [list(map(int, self.df_sub['transcript'][i].split(' ')))
                      for i in data_indices]

        input_names = list(
            map(lambda path: basename(path).split('.')[0],
                self.df['input_path'][data_indices]))

        return {'xs': xs, 'ys': ys, 'ys_sub': ys_sub, 'input_names': input_names}
