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


class Dataset(DatasetBase):

    def __init__(self, backend, input_channel, use_delta, use_double_delta,
                 data_type, label_type, batch_size,
                 vocab_file_path, max_epoch=None, splice=1,
                 num_stack=1, num_skip=1,
                 shuffle=False, sort_utt=False, reverse=False,
                 sort_stop_epoch=None, tool='htk',
                 num_enque=None, dynamic_batching=False):
        """A class for loading dataset.
        Args:
            backend (string): pytorch or chainer
            input_channel (int): the number of channels of acoustics
            use_delta (bool): if True, use the delta feature
            use_double_delta (bool): if True, use the acceleration feature
            data_type (string): train or dev or test
            label_type (string): phone39 or phone48 or phone61
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
            tool (string, optional): htk or librosa or python_speech_features
            num_enque (int, optional): the number of elements to enqueue
            dynamic_batching (bool, optional): if True, batch size will be
                chainged dynamically in training
        """
        self.is_test = True if data_type == 'test' else False

        self.backend = backend
        self.input_channel = input_channel
        self.use_delta = use_delta
        self.use_double_delta = use_double_delta
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
        self.tool = tool
        self.num_enque = num_enque
        self.dynamic_batching = dynamic_batching

        super(Dataset, self).__init__(vocab_file_path=vocab_file_path)

        # Load dataset file
        dataset_path = join('/n/sd8/inaguma/corpus/timit/kaldi/dataset',
                            tool, data_type, label_type + '.csv')
        df = pd.read_csv(dataset_path)
        df = df.loc[:, ['frame_num', 'input_path', 'transcript']]

        # Sort paths to input & label
        if sort_utt:
            df = df.sort_values(by='frame_num', ascending=not reverse)
        else:
            df = df.sort_values(by='input_path', ascending=True)

        self.df = df
        self.rest = set(list(df.index))

    def select_batch_size(self, batch_size, min_frame_num_batch):
        if not self.dynamic_batching:
            return batch_size

        if min_frame_num_batch < 700:
            batch_size = int(batch_size / 2)

        return batch_size
