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

# NOTE: Loading numpy is faster than loading htk


class Dataset(Base):

    def __init__(self, corpus, data_save_path,
                 input_freq, use_delta, use_double_delta,
                 data_size, data_type, label_type, label_type_sub,
                 batch_size, max_epoch=None,
                 max_frame_num=2000, min_frame_num=40,
                 shuffle=False, sort_utt=False, reverse=False,
                 sort_stop_epoch=None, num_gpus=1, tool='htk',
                 num_enque=None, dynamic_batching=False):
        """A class for loading dataset.
        Args:
            data_save_path (string): path to saved data
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
            num_gpus (int): the number of GPUs
            tool (string): htk or librosa or python_speech_features
            num_enque (int): the number of elements to enqueue
            dynamic_batching (bool): if True, batch size will be chainged
                dynamically in training
        """
        self.corpus = corpus
        self.input_freq = input_freq
        self.use_delta = use_delta
        self.use_double_delta = use_double_delta
        self.data_type = data_type
        self.label_type = label_type
        self.label_type_sub = label_type_sub
        self.batch_size = batch_size * num_gpus
        self.max_epoch = max_epoch
        self.shuffle = shuffle
        self.sort_utt = sort_utt
        self.sort_stop_epoch = sort_stop_epoch
        self.num_gpus = num_gpus
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
        if label_type == 'word':
            self.idx2word = Idx2word(self.vocab_file_path)
            self.word2idx = Word2idx(self.vocab_file_path)
        else:
            raise NotImplementedError

        self.vocab_file_path_sub = join(
            data_save_path, 'vocab', data_size, label_type_sub + '.txt')
        if 'character' in label_type_sub:
            self.idx2char = Idx2char(
                self.vocab_file_path_sub,
                capital_divide=label_type_sub == 'character_capital_divide')
            self.char2idx = Char2idx(
                self.vocab_file_path_sub,
                capital_divide=label_type_sub == 'character_capital_divide')
        else:
            raise NotImplementedError

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
            logger.info('Original utterance num (main): %d' % len(df))
            logger.info('Original utterance num (sub): %d' % len(df_sub))

            # For Switchboard
            if corpus == 'swbd' and 'train' in data_type:
                df = df[df.apply(lambda x: not(len(x['transcript'].split(' '))
                                               <= 3 and x['frame_num'] >= 1000), axis=1)]
                df_sub = df_sub[df_sub.apply(lambda x: not(len(x['transcript'].split(' '))
                                                           <= 24 and x['frame_num'] >= 1000), axis=1)]

            df = df[df.apply(
                lambda x: min_frame_num <= x['frame_num'] <= max_frame_num, axis=1)]
            df_sub = df_sub[df_sub.apply(
                lambda x: min_frame_num <= x['frame_num'] <= max_frame_num, axis=1)]

            logger.info('Restricted utterance num (main): %d' % len(df))
            logger.info('Restricted utterance num (sub): %d' % len(df_sub))

        # Sort paths to input & label
        if sort_utt:
            df = df.sort_values(by='frame_num', ascending=not reverse)
            df_sub = df_sub.sort_values(
                by='frame_num', ascending=not reverse)
        else:
            df = df.sort_values(by='input_path', ascending=True)
            df_sub = df_sub.sort_values(by='input_path', ascending=True)

        assert len(df) == len(df_sub)

        self.df = df
        self.df_sub = df_sub
        self.rest = set(list(df.index))

    def make_batch(self, data_indices):
        """Create mini-batch per step.
        Args:
            data_indices (np.ndarray):
        Returns:
            batch (dict):
                xs (list): input data of size `[B, T, input_size]`
                ys (list): target labels in the main task of size `[B, L]`
                ys_sub (list): target labels in the sub task of size `[B, L_sub]`
                x_lens (list): lengths of inputs of of size `[B]`
                y_lens (list): lengths of target labels in the main task of size `[B]`
                y_lens_sub (list): lengths of target labels in the sub task of size `[B]`
                input_names (list): file names of input data of size `[B]`
        """
        # Load dataset in mini-batch
        feat_paths = np.array(self.df['input_path'][data_indices])
        transcripts = np.array(self.df['transcript'][data_indices])
        transcripts_sub = np.array(self.df_sub['transcript'][data_indices])

        ##############################
        # features
        ##############################
        # Load features in parallel
        # feats = make_parallel(load_feat, feat_paths, core=4)
        feats = [load_feat(p) for p in feat_paths]

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

        #########################
        # transcript
        #########################
        if self.is_test:
            ys = [self.df['transcript'][data_indices[b]]
                  for b in range(len(xs))]
            ys_sub = [self.df_sub['transcript'][data_indices[b]]
                      for b in range(len(xs))]
            # NOTE: transcript is not tokenized
        else:
            ys = [list(map(int, transcripts[b].split(' ')))
                  for b in range(len(xs))]
            ys_sub = [list(map(int, transcripts_sub[b].split(' ')))
                      for b in range(len(xs))]

        input_names = list(
            map(lambda path: basename(path).split('.')[0],
                self.df['input_path'][data_indices]))

        return {'xs': xs, 'ys': ys, 'ys_sub': ys_sub, 'input_names': input_names}
