#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the hierarchical attention-based model (Librispeech corpus).
   In addition, frame stacking and skipping are used.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import pickle
import numpy as np

from utils.dataset.hierarchical_attention import DatasetBase


class Dataset(DatasetBase):

    def __init__(self, data_type, data_size, label_type, label_type_sub,
                 batch_size, vocab_file_path, vocab_file_path_sub,
                 max_epoch=None, splice=1,
                 num_stack=1, num_skip=1,
                 shuffle=False, sort_utt=True, reverse=False,
                 sort_stop_epoch=None, num_gpus=1):
        """A class for loading dataset.
        Args:
            data_type (string): train or dev_clean or dev_other or test_clean
                or test_other
            data_size (string): 100h or 460h or 960h
            label_type (string): word_freq1 or word_freq5 or word_freq10 or word_freq15
            label_type_sub (string): characater or characater_capital_divide
            batch_size (int): the size of mini-batch
            vocab_file_path (string): path to the vocabulary file in the main task
            vocab_file_path_sub (string): path to the vocabulary file in the sub task
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
        """
        super(Dataset, self).__init__(vocab_file_path=vocab_file_path,
                                      vocab_file_path_sub=vocab_file_path_sub)

        if data_type in ['test_clean', 'test_other']:
            self.is_test = True
        else:
            self.is_test = False

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

        # paths where datasets exist
        dataset_root = ['/data/inaguma/librispeech',
                        '/n/sd8/inaguma/corpus/librispeech/dataset']

        input_path = join(
            dataset_root[0], 'inputs', data_size, data_type)
        # NOTE: ex.)
        # librispeech/dataset/inputs/data_size/data_type/speaker/*.npy
        label_path = join(
            dataset_root[0], 'labels', data_size, data_type, label_type)
        label_path_sub = join(
            dataset_root[0], 'labels', data_size, data_type, label_type_sub)
        # NOTE: ex.)
        # librispeech/dataset/labels/data_size/data_type/label_type/speaker/*.npy

        # Load the frame number dictionary
        for _ in range(len(dataset_root)):
            if isfile(join(input_path, 'frame_num.pickle')):
                with open(join(input_path, 'frame_num.pickle'), 'rb') as f:
                    self.frame_num_dict = pickle.load(f)
                break
            else:
                if len(dataset_root) == 0:
                    raise ValueError('Dataset was not found.')

                dataset_root.pop(0)
                input_path = join(
                    dataset_root[0], 'inputs', data_size, data_type)
                label_path = join(
                    dataset_root[0], 'labels', data_size, data_type, label_type)
                label_path_sub = join(
                    dataset_root[0], 'labels', data_size, data_type, label_type_sub)
                with open(join(input_path, 'frame_num.pickle'), 'rb') as f:
                    self.frame_num_dict = pickle.load(f)

        # Sort paths to input & label
        axis = 1 if sort_utt else 0
        frame_num_tuple_sorted = sorted(self.frame_num_dict.items(),
                                        key=lambda x: x[axis],
                                        reverse=reverse)

        input_paths, label_paths, label_paths_sub = [], [], []
        for utt_name, frame_num in frame_num_tuple_sorted:
            speaker = utt_name.split('-')[0]
            # ex.) utt_name: speaker-book-utt_index
            input_paths.append(join(input_path, speaker, utt_name + '.npy'))
            label_paths.append(join(label_path, speaker, utt_name + '.npy'))
            label_paths_sub.append(
                join(label_path_sub, speaker, utt_name + '.npy'))
        self.input_paths = np.array(input_paths)
        self.label_paths = np.array(label_paths)
        self.label_paths_sub = np.array(label_paths_sub)
        # NOTE: Not load dataset yet

        self.rest = set(range(0, len(self.input_paths), 1))
