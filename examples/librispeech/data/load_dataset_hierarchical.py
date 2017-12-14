#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the hierarchical CTC and attention-based model (Librispeech corpus).
   In addition, frame stacking and skipping are used.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import pandas as pd

from utils.dataset.loader_hierarchical import DatasetBase


class Dataset(DatasetBase):

    def __init__(self, input_channel, use_delta, use_double_delta,
                 model_type, data_type, data_size,
                 label_type, label_type_sub,
                 batch_size, vocab_file_path, vocab_file_path_sub,
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
            model_type (string): hierarchical_ctc or hierarchical_attention
            data_type (string): train or dev_clean or dev_other or test_clean
                or test_other
            data_size (string): 100h or 460h or 960h
            label_type (string): word_freq1 or word_freq5 or word_freq10 or word_freq15
            label_type_sub (string): characater or characater_capital_divide
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
            num_gpus (int, optional): the number of GPUs
            save_format (string, optional): numpy or htk
            num_enque (int, optional): the number of elements to enqueue
        """
        super(Dataset, self).__init__(vocab_file_path=vocab_file_path,
                                      vocab_file_path_sub=vocab_file_path_sub)

        if data_type in ['test_clean', 'test_other']:
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
        self.save_format = save_format
        self.num_enque = num_enque

        # Load dataset file
        dataset_path = join('/n/sd8/inaguma/corpus/librispeech/dataset',
                            save_format, data_size, data_type, label_type + '.csv')
        dataset_path_sub = join('/n/sd8/inaguma/corpus/librispeech/dataset',
                                save_format, data_size, data_type, label_type_sub + '.csv')
        df = pd.read_csv(dataset_path)
        df = df.loc[:, ['frame_num', 'input_path', 'transcript']]
        df_sub = pd.read_csv(dataset_path_sub)
        df_sub = df_sub.loc[:, ['frame_num', 'input_path', 'transcript']]

        # Remove long utteraces (> 20s)
        if data_type == 'train':
            print('Original utterance num (main): %d' % len(df))
            print('Original utterance num (sub): %d' % len(df_sub))
            df = df[df.apply(lambda x: x['frame_num'] <= 2000, axis=1)]
            df_sub = df_sub[df_sub.apply(
                lambda x: x['frame_num'] <= 2000, axis=1)]
            print('Restricted utterance num (main): %d' % len(df))
            print('Restricted utterance num (sub): %d' % len(df_sub))

        # Sort paths to input & label
        if sort_utt:
            df = df.sort_values(by='frame_num', ascending=not reverse)
        else:
            df = df.sort_values(by='input_path', ascending=True)

        assert len(df) == len(df_sub)

        self.df = df
        self.df_sub = df_sub
        self.rest = set(list(df.index))
