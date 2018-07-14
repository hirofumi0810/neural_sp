#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for loading dataset for language models.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename, isfile, join
import pandas as pd
import codecs
import logging
from tqdm import tqdm
logger = logging.getLogger('training')

from src.dataset.base import Base
from src.utils.io.labels.word import Idx2word, Word2idx
from src.utils.io.labels.character import Idx2char, Char2idx
from src.utils.directory import mkdir_join


class Dataset(Base):

    def __init__(self, corpus, data_save_path, model_type,
                 data_size, vocab, data_type, label_type,
                 batch_size, max_epoch=None,
                 shuffle=False, sort_utt=False, reverse=False,
                 sort_stop_epoch=None, tool='htk',
                 num_enque=None, dynamic_batching=False):
        """A class for loading dataset.
        Args:
            corpus (string): the name of corpus
            data_save_path (string): path to saved data
            model_type (string):
            data_size (string):
            vocab (bool or string):
            data_type (string):
            label_type (string):
            batch_size (int): the size of mini-batch
            max_epoch (int): the max epoch. None means infinite loop.
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
        """
        self.corpus = corpus
        self.model_type = model_type
        self.data_type = data_type
        self.data_size = data_size
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
            self.vocab_file_path = join(
                data_save_path, 'vocab', vocab, label_type + '.txt')
        else:
            self.vocab_file_path = join(
                data_save_path, 'vocab', data_size, label_type + '.txt')

        if label_type == 'word':
            self.idx2word = Idx2word(self.vocab_file_path)
            self.word2idx = Word2idx(self.vocab_file_path)
        elif 'character' in label_type:
            self.idx2char = Idx2char(self.vocab_file_path)
            self.char2idx = Char2idx(self.vocab_file_path)
        else:
            raise ValueError(label_type)

        super(Dataset, self).__init__(vocab_file_path=self.vocab_file_path)
        self.eos = self.num_classes

        # Load dataset file
        dataset_path = join(
            data_save_path, 'dataset', tool, data_size, data_type, label_type + '.csv')
        df = pd.read_csv(dataset_path, encoding='utf-8')
        df = df.loc[:, ['frame_num', 'input_path', 'transcript']]

        # Sort paths to input & label
        if sort_utt:
            df = df.sort_values(by='frame_num', ascending=not reverse)
        else:
            df = df.sort_values(by='input_path', ascending=True)

        self.df = df
        self.rest = set(list(df.index))

    def select_batch_size(self, batch_size, min_frame_num_batch):
        return batch_size

    def make_batch(self, data_indices):
        """Create mini-batch per step.
        Args:
            data_indices (np.ndarray):
        Returns:
            batch (dict):
                ys (list): target labels of size `[B * L]`
                input_names (list): file names of input data of size `[B]`
        """
        # NOTE: sample utteraces and concatenate all tokens in mini-batch

        ys = []
        for i in data_indices:
            if self.is_test:
                if self.label_type == 'word':
                    indices = self.word2idx(self.df['transcript'][i])
                elif 'character' in self.label_type:
                    indices = self.char2idx(self.df['transcript'][i])
                else:
                    raise ValueError(self.label_type)
                ys += [self.eos] + indices
                # NOTE: transcript is seperated by space('_')
                # NOTE: add <EOS> between sequences
            else:
                ys += [self.eos] + \
                    list(map(int, self.df['transcript'][i].split(' ')))
        ys += [self.eos]

        # TODO: fix later
        try:
            input_names = list(
                map(lambda path: basename(path).split('.')[0],
                    self.df['input_path'][data_indices]))
        except:
            input_names = self.df.index[data_indices].values.tolist()

        return {'ys': ys, 'input_names': input_names}
