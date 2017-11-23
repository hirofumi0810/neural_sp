#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the CTC and attention-based model (TIMIT corpus).
   In addition, frame stacking and skipping are used.
   You can use only the single GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import pandas as pd

from utils.dataset.loader import DatasetBase
from utils.io.labels.phone import Phone2idx
from utils.io.labels.character import Char2idx


class Dataset(DatasetBase):

    def __init__(self, model_type, data_type, label_type, batch_size,
                 vocab_file_path, max_epoch=None, splice=1,
                 num_stack=1, num_skip=1,
                 shuffle=False, sort_utt=False, reverse=False,
                 sort_stop_epoch=None,
                 use_cuda=False, volatile=False, save_format='numpy'):
        """A class for loading dataset.
        Args:
            model_type (string): ctc or attention
            data_type (string): train or dev or test
            label_type (string): phone39 or phone48 or phone61 or
                character or character_capital_divide
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
            use_cuda (bool, optional):
            volatile (boo, optional):
            save_format (string, optional): numpy or htk
        """
        super(Dataset, self).__init__(vocab_file_path=vocab_file_path)

        self.is_test = True if data_type == 'test' else False

        self.model_type = model_type
        self.data_type = data_type
        self.label_type = label_type
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.splice = splice
        self.num_stack = num_stack
        self.num_skip = num_skip
        self.shuffle = shuffle
        self.sort_utt = sort_utt
        self.sort_stop_epoch = sort_stop_epoch
        self.num_gpus = 1
        self.use_cuda = use_cuda
        self.volatile = volatile
        self.save_format = save_format

        # Set mapping function
        if 'phone' in label_type:
            self.map_fn = Phone2idx(vocab_file_path)
            dataset_path = join('/n/sd8/inaguma/corpus/timit/dataset',
                                data_type, 'dataset_' + save_format + '_' + label_type + '.csv')
        else:
            dataset_path = join('/n/sd8/inaguma/corpus/timit/dataset',
                                data_type, 'dataset_' + save_format + '_character.csv')
            if label_type == 'character':
                self.map_fn = Char2idx(vocab_file_path)

            else:
                self.map_fn = Char2idx(vocab_file_path, capital_divide=True)

        # Load dataset file
        self.df = pd.read_csv(dataset_path)
        self.df = self.df.loc[:, [
            'frame_num', 'input_path', 'transcript']]

        # Sort paths to input & label
        if sort_utt:
            self.df = self.df.sort_values(
                by='frame_num', ascending=not reverse)
        else:
            self.df = self.df.sort_values(by='input_path', ascending=True)
        new_df = pd.DataFrame([0] * len(self), columns=['index'])
        self.df = pd.concat([self.df, new_df], axis=1)

        self.rest = set(range(0, len(self.df), 1))
