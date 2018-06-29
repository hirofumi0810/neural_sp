#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for loading dataset for phoneme-to-word models.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename, isfile, join
import numpy as np
import pandas as pd
import codecs
import logging
from tqdm import tqdm
logger = logging.getLogger('training')

from src.dataset.base import Base
from src.utils.io.labels.word import Idx2word, Word2idx
from src.utils.io.labels.character import Idx2char, Char2idx
from src.utils.io.labels.phone import Idx2phone, Phone2idx
from src.utils.directory import mkdir_join


class Dataset(Base):

    def __init__(self, corpus, data_save_path,
                 data_size, data_type, label_type_in, label_type,
                 batch_size, max_epoch=None,
                 max_frame_num=2000, min_frame_num=40,
                 shuffle=False, sort_utt=False, reverse=False,
                 sort_stop_epoch=None, tool='htk',
                 num_enque=None, dynamic_batching=False, vocab=False,
                 use_ctc=False, subsampling_factor=1):
        """A class for loading dataset.
        Args:
            data_save_path (string): path to saved data
            data_size (string):
            data_type (string):
            label_type_in (string):
            label_type (string):
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
            vocab (bool or string):
            use_ctc (bool):
            subsampling_factor (int):
        """
        self.corpus = corpus
        self.data_type = data_type
        self.data_size = data_size
        self.label_type_in = label_type_in
        self.label_type = label_type
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
        elif corpus in['librispeech', 'timit']:
            self.is_test = True if 'test' in data_type else False
        else:
            raise NotImplementedError

        # TODO: fix this
        if corpus == 'librispeech':
            if data_type == 'train':
                data_type += '_' + data_size

        if vocab and data_size != '' and data_size != vocab:
            self.vocab_file_path_in = join(
                data_save_path, 'vocab', vocab, label_type_in + '.txt')
            self.vocab_file_path = join(
                data_save_path, 'vocab', vocab, label_type + '.txt')
            vocab_file_path_org = join(
                data_save_path, 'vocab', data_size, label_type_in + '.txt')
        else:
            self.vocab_file_path_in = join(
                data_save_path, 'vocab', data_size, label_type_in + '.txt')
            self.vocab_file_path = join(
                data_save_path, 'vocab', data_size, label_type + '.txt')

        if 'phone' in label_type_in:
            self.idx2phone = Idx2phone(self.vocab_file_path_in)
            self.phone2idx = Phone2idx(self.vocab_file_path_in)
        elif 'character' in label_type_in:
            self.idx2char = Idx2char(self.vocab_file_path_in)
            self.char2idx = Char2idx(self.vocab_file_path_in)
        else:
            raise ValueError(label_type_in)
        if label_type == 'word':
            self.idx2word = Idx2word(self.vocab_file_path)
            self.word2idx = Word2idx(self.vocab_file_path)
        elif 'character' in label_type:
            self.idx2char = Idx2char(self.vocab_file_path)
            self.char2idx = Char2idx(self.vocab_file_path)
        else:
            raise ValueError(label_type)

        super(Dataset, self).__init__(vocab_file_path_in=self.vocab_file_path_in,
                                      vocab_file_path=self.vocab_file_path)

        # Load dataset file
        if vocab and data_size != '' and data_size != vocab and not self.is_test:
            dataset_path_in = mkdir_join(
                data_save_path, 'dataset', tool, data_size + '_' + vocab, data_type, label_type_in + '.csv')
            dataset_path = mkdir_join(
                data_save_path, 'dataset', tool, data_size + '_' + vocab, data_type, label_type + '.csv')

            # Change token indices
            if not isfile(dataset_path):
                dataset_path_org = join(
                    data_save_path, 'dataset', tool, data_size, data_type, label_type + '.csv')
                df = pd.read_csv(dataset_path_org, encoding='utf-8')
                df = df.loc[:, ['frame_num', 'input_path', 'transcript']]

                # Change vocabulary
                org2new = {}
                str2idx_org = {}
                str2idx_new = {}
                # new vocab
                with codecs.open(self.vocab_file_path, 'r', 'utf-8') as f:
                    vocab_count = 0
                    for line in f:
                        if line.strip() != '':
                            str2idx_new[line.strip()] = vocab_count
                            vocab_count += 1
                # original vocab
                with codecs.open(vocab_file_path_org, 'r', 'utf-8') as f:
                    vocab_count = 0
                    for line in f:
                        if line.strip() != '':
                            str2idx_org[line.strip()] = vocab_count
                            vocab_count += 1
                for k, v in str2idx_org.items():
                    if k in str2idx_new.keys():
                        org2new[v] = str2idx_new[k]
                    else:
                        org2new[v] = str2idx_new['OOV']

                # Update the transcript
                for i in tqdm(df['transcript'].index):
                    df['transcript'][i] = ' '.join(
                        list(map(lambda x: str(org2new[int(x)]), df['transcript'][i].split(' '))))

                # Save as a new file
                df.to_csv(dataset_path_in, encoding='utf-8')
            else:
                df_in = pd.read_csv(dataset_path_in, encoding='utf-8')
                df_in = df_in.loc[:, ['frame_num', 'input_path', 'transcript']]
                df = pd.read_csv(dataset_path, encoding='utf-8')
                df = df_in.loc[:, [
                    'frame_num', 'input_path', 'transcript']]
        else:
            dataset_path_in = join(
                data_save_path, 'dataset', tool, data_size, data_type, label_type_in + '.csv')
            dataset_path = join(
                data_save_path, 'dataset', tool, data_size, data_type, label_type + '.csv')
            df_in = pd.read_csv(dataset_path_in, encoding='utf-8')
            df_in = df_in.loc[:, ['frame_num', 'input_path', 'transcript']]
            df = pd.read_csv(dataset_path, encoding='utf-8')
            df = df.loc[:, ['frame_num', 'input_path', 'transcript']]

        # Remove inappropriate utteraces
        if not self.is_test:
            print('Original utterance num (input): %d' % len(df_in))
            print('Original utterance num (output): %d' % len(df))
            utt_num_orig_in = len(df_in)
            utt_num_orig = len(df)

            # For Switchboard
            if corpus == 'swbd' and 'train' in data_type:
                df_in = df_in[df_in.apply(lambda x: not(len(x['transcript'].split(' '))
                                                        <= 3 and x['frame_num'] >= 1000), axis=1)]
                df = df[df.apply(lambda x: not(len(x['transcript'].split(' '))
                                               <= 24 and x['frame_num'] >= 1000), axis=1)]

            # Remove by threshold
            df_in = df_in[df_in.apply(
                lambda x: min_frame_num <= x['frame_num'] <= max_frame_num, axis=1)]
            df = df[df.apply(
                lambda x: min_frame_num <= x['frame_num'] <= max_frame_num, axis=1)]
            print('Removed utterance num (threshold, input): %d' %
                  (utt_num_orig_in - len(df_in)))
            print('Removed utterance num (threshold, output): %d' %
                  (utt_num_orig - len(df)))

            # Remove for CTC loss calculatioon
            if subsampling_factor > 1:
                print('Checking utterances for subsampling')
                utt_num_orig_in = len(df_in)
                df_in = df_in[df_in.apply(
                    lambda x: len(x['transcript'].split(' ')) // subsampling_factor > 0, axis=1)]
                print('Removed utterance num (for subsampling): %d' %
                      (utt_num_orig_in - len(df_in)))

                if use_ctc:
                    pass

            # Make up the number
            if len(df_in) != len(df):
                df_in = df_in.drop(df_in.index.difference(df.index))
                df = df.drop(df.index.difference(df_in.index))

        # Sort paths to input & label
        if sort_utt:
            df_in = df_in.sort_values(by='frame_num', ascending=not reverse)
            df = df.sort_values(by='frame_num', ascending=not reverse)
        else:
            df_in = df_in.sort_values(by='input_path', ascending=True)
            df = df.sort_values(by='input_path', ascending=True)

        assert len(df_in) == len(df)

        self.df_in = df_in
        self.df = df
        self.rest = set(list(df_in.index))

    def select_batch_size(self, batch_size, min_frame_num_batch):
        return batch_size

    def make_batch(self, data_indices):
        """Create mini-batch per step.
        Args:
            data_indices (np.ndarray):
        Returns:
            batch (dict):
                xs (list): target labels of size `[B, L_in]`
                ys (list): target labels of size `[B, L]`
                input_names (list): file names of input data of size `[B]`
        """
        # Load dataset in mini-batch
        transcripts_in = np.array(self.df_in['transcript'][data_indices])
        transcripts_out = np.array(self.df['transcript'][data_indices])

        if self.is_test:
            xs = [self.df_in['transcript'][data_indices[b]]
                  for b in range(len(data_indices))]
            ys = [self.df['transcript'][data_indices[b]]
                  for b in range(len(data_indices))]
            # NOTE: transcript is not tokenized
        else:
            xs = [list(map(int, transcripts_in[b].split(' ')))
                  for b in range(len(data_indices))]
            ys = [list(map(int, transcripts_out[b].split(' ')))
                  for b in range(len(data_indices))]

        # TODO: fix later
        try:
            input_names = list(
                map(lambda path: basename(path).split('.')[0],
                    self.df['input_path'][data_indices]))
        except:
            input_names = self.df.index.values.tolist()

        return {'xs': xs, 'ys': ys, 'input_names': input_names}
