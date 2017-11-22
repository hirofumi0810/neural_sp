#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the hierarchical CTC and attention-based model (CSJ corpus).
   In addition, frame stacking and skipping are used.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import pandas as pd

from utils.dataset.loader_hierarchical import DatasetBase
from utils.io.labels.phone import Phone2idx
from utils.io.labels.character import Char2idx
from utils.io.labels.word import Word2idx


class Dataset(DatasetBase):

    def __init__(self, model_type, data_type, data_size,
                 label_type, label_type_sub,
                 batch_size, vocab_file_path, vocab_file_path_sub,
                 max_epoch=None, splice=1,
                 num_stack=1, num_skip=1,
                 shuffle=False, sort_utt=False, reverse=False,
                 sort_stop_epoch=None, num_gpus=1,
                 use_cuda=False, volatile=False, save_format='numpy'):
        """A class for loading dataset.
        Args:
            model_type (string): hierarchical_attention or hierarchical_ctc
            data_type (string): train or dev or eval1 or eval2 or eval3
            data_size (string): subset or fullset
            label_type (string): kanji or kanji_divide or word_freq1 or
                word_freq5 or word_freq10 or word_freq15
            label_type_sub (string): kanji or kanji_divide or kana or kana_divide
            batch_size (int): the size of mini-batch
            vocab_file_path (string): path to the vocabulary file in the main
                task
            vocab_file_path_sub (string): path to the vocabulary file in the
                sub task
            max_epoch (int, optional): the max epoch. None means infinite loop.
            splice (int, optional): frames to splice. Default is 1 frame.
            num_stack (int, optional): the number of frames to stack
            num_skip (int, optional): the number of frames to skip
            shuffle (bool, optional): if True, shuffle utterances. This is
                disabled when sort_utt is True.
            sort_utt (bool, optional): if True, sort all utterances in the
                ascending order
            reverse (bool, optional): if True, sort utteraces in the
                descending order
            sort_stop_epoch (int, optional): After sort_stop_epoch, training
                will revert back to a random order
            num_gpus (optional, int): the number of GPUs
            use_cuda (bool, optional):
            volatile (boo, optional):
            save_format (string, optional): numpy or htk
        """
        super(Dataset, self).__init__()

        if data_type in ['eval1', 'eval2', 'eval3']:
            self.is_test = True
        else:
            self.is_test = False

        self.model_type = model_type
        self.data_type = data_type
        self.data_size = data_size
        self.label_type = label_type
        self.label_type_sub = label_type_sub
        self.batch_size = batch_size * num_gpus
        self.max_epoch = max_epoch
        self.splice = splice
        self.num_stack = num_stack
        self.num_skip = num_skip
        self.shuffle = shuffle
        self.sort_utt = sort_utt
        self.sort_stop_epoch = sort_stop_epoch
        self.num_gpus = num_gpus
        self.use_cuda = use_cuda
        self.volatile = volatile
        self.save_format = save_format

        # Set mapping function
        dataset_path = join(
            '/n/sd8/inaguma/corpus/csj/dataset',
            data_size, data_type, 'dataset_kanji_' + save_format + '.csv')

        if 'kana' in label_type_sub:
            dataset_path_sub = join(
                '/n/sd8/inaguma/corpus/csj/dataset',
                data_size, data_type, 'dataset_kana_' + save_format + '.csv')
        elif 'kanji' in label_type_sub:
            dataset_path_sub = join(
                '/n/sd8/inaguma/corpus/csj/dataset',
                data_size, data_type, 'dataset_kanji_' + save_format + '.csv')
        elif 'phone' in label_type_sub:
            dataset_path_sub = join(
                '/n/sd8/inaguma/corpus/csj/dataset',
                data_size, data_type, 'dataset_phone_' + save_format + '.csv')

        self.map_fn = Word2idx(vocab_file_path)
        self.map_fn_sub = Char2idx(vocab_file_path_sub, double_letter=True)

        # Load dataset file
        self.df = pd.read_csv(dataset_path)
        self.df = self.df.loc[:, [
            'frame_num', 'input_path', 'transcript']]
        self.df_sub = pd.read_csv(dataset_path_sub)
        self.df_sub = self.df_sub.loc[:, [
            'frame_num', 'input_path', 'transcript']]

        # Sort paths to input & label
        if sort_utt:
            self.df = self.df.sort_values(
                by='frame_num', ascending=not reverse)
            self.df_sub = self.df_sub.sort_values(
                by='frame_num', ascending=not reverse)
        else:
            self.df = self.df.sort_values(by='input_path', ascending=True)
            self.df_sub = self.df_sub.sort_values(
                by='input_path', ascending=True)

        self.rest = set(range(0, len(self.df), 1))
