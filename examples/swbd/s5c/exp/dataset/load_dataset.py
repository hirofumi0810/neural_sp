#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the CTC and attention-based model (Switchboard corpus).
   In addition, frame stacking and skipping are used.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import pandas as pd
import logging
logger = logging.getLogger('training')

from utils.dataset.loader import DatasetBase


class Dataset(DatasetBase):

    def __init__(self, data_save_path,
                 backend, input_channel, use_delta, use_double_delta,
                 data_type, data_size, label_type,
                 batch_size, max_epoch=None, splice=1,
                 num_stack=1, num_skip=1,
                 shuffle=False, sort_utt=False, reverse=False,
                 sort_stop_epoch=None, num_gpus=1, tool='htk',
                 num_enque=None, dynamic_batching=False):
        """A class for loading dataset.
        Args:
            data_save_path (string): path to saved data
            backend (string): pytorch or chainer
            input_channel (int): the number of channels of acoustics
            use_delta (bool): if True, use the delta feature
            use_double_delta (bool): if True, use the acceleration feature
            data_type (string): train or dev or eval2000_swbd or eval2000_ch
            data_size (string): 300h or 2000h
            label_type (string): characater or characater_capital_divide or
                word1 or word5 or word10 or word15
            batch_size (int): the size of mini-batch
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
            num_gpus (int, optional): the number of GPUs
            tool (string, optional): htk or librosa or python_speech_features
            num_enque (int, optional): the number of elements to enqueue
            dynamic_batching (bool, optional): if True, batch size will be
                chainged dynamically in training
        """
        self.is_test = True if 'eval' in data_type else False

        self.backend = backend
        self.input_channel = input_channel
        self.use_delta = use_delta
        self.use_double_delta = use_double_delta
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
        self.tool = tool
        self.num_enque = num_enque
        self.dynamic_batching = dynamic_batching

        if isfile(data_save_path):
            data_save_path = data_save_path[:-3]
        # TODO: fix this

        self.vocab_file_path = join(
            data_save_path, 'vocab', label_type + '.txt')

        super(Dataset, self).__init__(vocab_file_path=self.vocab_file_path)

        # Load dataset file
        dataset_path = join(
            data_save_path, 'dataset', tool, data_type, label_type + '.csv')
        df = pd.read_csv(dataset_path)
        df = df.loc[:, ['frame_num', 'input_path', 'transcript']]

        # Remove inappropriate utteraces
        if not self.is_test:
            logger.info('Original utterance num: %d' % len(df))
            if 'word' in label_type:
                df = df[df.apply(lambda x: not(len(x['transcript'].split(' '))
                                               <= 3 and x['frame_num'] >= 1000), axis=1)]
            else:
                df = df[df.apply(lambda x: not(len(x['transcript'].split(' '))
                                               <= 24 and x['frame_num'] >= 1000), axis=1)]
            logger.info('Restricted utterance num: %d' % len(df))

        # Sort paths to input & label
        if sort_utt and data_type != 'dev':
            df = df.sort_values(by='frame_num', ascending=not reverse)
        else:
            df = df.sort_values(by='input_path', ascending=True)

        self.df = df
        self.rest = set(list(df.index))

    def select_batch_size(self, batch_size, min_frame_num_batch):
        if not self.dynamic_batching:
            return batch_size

        if 'word' in self.data_type:
            if min_frame_num_batch <= 300:
                batch_size = batch_size * 2
            elif min_frame_num_batch <= 600:
                batch_size = int(batch_size * 1.5)
            elif min_frame_num_batch <= 1500:
                pass
            elif min_frame_num_batch <= 1700:
                batch_size = 8
            else:
                batch_size = 1
        else:
            if min_frame_num_batch <= 300:
                batch_size = batch_size * 2
            elif min_frame_num_batch <= 600:
                batch_size = int(batch_size * 1.5)
            elif min_frame_num_batch <= 1200:
                pass
            elif min_frame_num_batch <= 1500:
                batch_size = int(batch_size / 2)
            elif min_frame_num_batch <= 1700:
                batch_size = 8
            else:
                batch_size = 1

        if batch_size < 1:
            batch_size = 1

        return batch_size
