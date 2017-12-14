#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the CTC and attention-based model (CSJ corpus).
   In addition, frame stacking and skipping are used.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import pandas as pd

from utils.dataset.loader import DatasetBase


class Dataset(DatasetBase):

    def __init__(self, input_channel, use_delta, use_double_delta,
                 model_type, data_type, data_size, label_type,
                 batch_size, vocab_file_path,
                 max_epoch=None, splice=1,
                 num_stack=1, num_skip=1,
                 shuffle=False, sort_utt=False, reverse=False,
                 sort_stop_epoch=None, num_gpus=1, save_format='numpy',
                 num_enque=None):
        """A class for loading dataset.
        Args:
            input_channel (int):
            use_delta (bool):
            use_double_delta (bool):
            model_type (string): attention or ctc
            data_type (string): train or dev or eval1 or eval2 or eval3
            data_size (string): subset or fullset
            label_type (string): kanji or kanji_divide or kana or kana_divide
                or word_freq1 or word_freq5 or word_freq10 or word_freq15
            batch_size (int): the size of mini-batch
            vocab_file_path (string): path to the vocabulary file
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
            save_format (string, optional): numpy or htk
            num_enque (int, optional): the number of elements to enqueue
        """
        super(Dataset, self).__init__(vocab_file_path=vocab_file_path)

        if data_type in ['eval1', 'eval2', 'eval3']:
            self.is_test = True
        else:
            self.is_test = False

        self.input_channel = input_channel
        self.use_delta = use_delta
        self.use_double_delta = use_double_delta
        self.model_type = model_type
        self.data_type = data_type
        self.data_size = data_size
        self.label_type = label_type
        self.batch_size = batch_size * num_gpus
        self.max_epoch = max_epoch
        self.splice = splice
        self.num_stack = num_stack
        self.num_skip = num_skip
        self.shuffle = shuffle
        self.sort_utt = sort_utt
        self.sort_stop_epoch = sort_stop_epoch
        self.num_gpus = num_gpus
        self.save_format = save_format
        self.num_enque = num_enque

        # Load dataset file
        dataset_path = join('/n/sd8/inaguma/corpus/csj/dataset',
                            save_format, data_size, data_type, label_type + '.csv')
        df = pd.read_csv(dataset_path)
        df = df.loc[:, ['frame_num', 'input_path', 'transcript']]

        # Remove inappropriate utteraces
        if not self.is_test:
            print('Original utterance num: %d' % len(df))
            df = df[df.apply(lambda x: 20 <= x['frame_num'] <= 2000, axis=1)]
            print('Restricted utterance num: %d' % len(df))

        # Sort paths to input & label
        if sort_utt:
            df = df.sort_values(by='frame_num', ascending=not reverse)
        else:
            df = df.sort_values(by='input_path', ascending=True)

        self.df = df
        self.rest = set(list(df.index))
