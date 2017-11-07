#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the Attention model (CSJ corpus).
   In addition, frame stacking and skipping are used.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import pickle
import numpy as np

from utils.dataset.attention import DatasetBase


class Dataset(DatasetBase):

    def __init__(self, data_type, train_data_size, label_type, batch_size,
                 map_file_path, max_epoch=None, splice=1,
                 num_stack=1, num_skip=1,
                 shuffle=False, sort_utt=True, sort_stop_epoch=None,
                 progressbar=False, num_gpu=1):
        """A class for loading dataset.
        Args:
            data_type (string): train or dev or eval1 or eval2 or eval3
            train_data_size (string): train_subset or train_fullset
            label_type (string): kanji or kanji_divide or kana or
                kana_divide
            batch_size (int): the size of mini-batch
            map_file_path (string): path to the mapping file
            max_epoch (int, optional): the max epoch. None means infinite loop.
            splice (int, optional): frames to splice. Default is 1 frame.
            num_stack (int, optional): the number of frames to stack
            num_skip (int, optional): the number of frames to skip
            shuffle (bool, optional): if True, shuffle utterances. This is
                disabled when sort_utt is True.
            sort_utt (bool, optional): if True, sort all utterances by the
                number of frames and utteraces in each mini-batch are shuffled.
                Otherwise, shuffle utteraces.
            sort_stop_epoch (int, optional): After sort_stop_epoch, training
                will revert back to a random order
            progressbar (bool, optional): if True, visualize progressbar
        """
        super(Dataset, self).__init__(map_file_path=map_file_path)

        if data_type in ['eval1', 'eval2', 'eval3']:
            self.is_test = True
        else:
            self.is_test = False

        self.data_type = data_type
        self.train_data_size = train_data_size
        self.label_type = label_type
        self.batch_size = batch_size * num_gpu
        self.max_epoch = max_epoch
        self.splice = splice
        self.num_stack = num_stack
        self.num_skip = num_skip
        self.shuffle = shuffle
        self.sort_utt = sort_utt
        self.sort_stop_epoch = sort_stop_epoch
        self.progressbar = progressbar
        self.num_gpu = num_gpu

        # paths where datasets exist
        dataset_root = ['/data/inaguma/csj',
                        '/n/sd8/inaguma/corpus/csj/dataset']

        input_path = join(dataset_root[0], 'inputs',
                          train_data_size, data_type)
        # NOTE: ex.) save_path:
        # csj_dataset_path/inputs/train_data_size/data_type/speaker/***.npy
        label_path = join(dataset_root[0], 'labels',
                          train_data_size, data_type, label_type)
        # NOTE: ex.) save_path:
        # csj_dataset_path/labels/train_data_size/data_type/label_type/speaker/***.npy

        # Load the frame number dictionary
        if isfile(join(input_path, 'frame_num.pickle')):
            with open(join(input_path, 'frame_num.pickle'), 'rb') as f:
                self.frame_num_dict = pickle.load(f)
        else:
            dataset_root.pop(0)
            input_path = join(dataset_root[0], 'inputs',
                              train_data_size, data_type)
            label_path = join(dataset_root[0], 'labels',
                              train_data_size, data_type, label_type)
            with open(join(input_path, 'frame_num.pickle'), 'rb') as f:
                self.frame_num_dict = pickle.load(f)

        # Sort paths to input & label
        axis = 1 if sort_utt else 0
        frame_num_tuple_sorted = sorted(self.frame_num_dict.items(),
                                        key=lambda x: x[axis])
        input_paths, label_paths = [], []
        for utt_name, frame_num in frame_num_tuple_sorted:
            speaker = utt_name.split('_')[0]
            # ex.) utt_name: speaker_uttindex
            input_paths.append(join(input_path, speaker, utt_name + '.npy'))
            label_paths.append(join(label_path, speaker, utt_name + '.npy'))
        self.input_paths = np.array(input_paths)
        self.label_paths = np.array(label_paths)
        # NOTE: Not load dataset yet

        self.rest = set(range(0, len(self.input_paths), 1))
