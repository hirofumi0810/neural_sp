#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for loading dataset for the CTC and attention-based model.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
import math
import numpy as np

from utils.dataset.base import Base
from utils.io.inputs.frame_stacking import stack_frame
from utils.io.inputs.splicing import do_splice

# NOTE: Loading numpy is faster than loading htk


class DatasetBase(Base):

    def __init__(self, *args, **kwargs):
        super(DatasetBase, self).__init__(*args, **kwargs)

    def make_batch(self, data_indices):
        """Create mini-batch per step.
        Args:
            data_indices (np.ndarray):
        Returns:
            batch (dict):
                xs (np.ndarray): input data of size
                    `[B, T_in, input_size]`
                ys (np.ndarray): target labels in the main task of size
                    `[B, T_out]`
                x_lens (np.ndarray): lengths of inputs of of size
                    `[B]`
                y_lens (np.ndarray): lengths of target labels in the main task of size
                    `[B]`
                input_names (np.ndarray): file names of input data of size
                    `[B]`
        """
        input_path_list = np.array(self.df['input_path'][data_indices])
        str_indices_list = np.array(self.df['transcript'][data_indices])

        if not hasattr(self, 'input_size'):
            if self.use_double_delta:
                self.input_size = self.input_freq * 3
            elif self.use_delta:
                self.input_size = self.input_freq * 2
            else:
                self.input_size = self.input_freq
            self.input_size *= self.num_stack
            self.input_size *= self.splice

        # Compute max frame num in mini-batch
        max_frame_num = max(self.df['frame_num'][data_indices])
        max_frame_num = math.ceil(max_frame_num / self.num_skip)

        # Compute max target label length in mini-batch
        max_label_num = max(
            map(lambda x: len(str(x).split(' ')), str_indices_list))
        # TODO: fix POS tag (nan -> 'nan')

        # Initialization
        if self.backend == 'pytorch':
            xs = np.zeros(
                (len(data_indices), max_frame_num, self.input_size),
                dtype=np.float32)
        elif self.backend == 'chainer':
            xs = [None] * len(data_indices)
        if self.is_test:
            ys = np.array(
                [[self.pad_value] * max_label_num] * len(data_indices))
        else:
            ys = np.array(
                [[self.pad_value] * max_label_num] * len(data_indices), dtype=np.int32)
        x_lens = np.zeros((len(data_indices),), dtype=np.int32)
        y_lens = np.zeros((len(data_indices),), dtype=np.int32)
        input_names = np.array(list(
            map(lambda path: basename(path).split('.')[0],
                np.array(self.df['input_path'][data_indices]))))

        # Set values of each data in mini-batch
        for b in range(len(data_indices)):
            # Load input data
            try:
                data_i_tmp = self.load(
                    input_path_list[b].replace(
                        '/n/sd8/inaguma/corpus', '/data/inaguma'))
            except:
                try:
                    data_i_tmp = self.load(
                        input_path_list[b].replace(
                            '/n/sd8/inaguma/corpus', '/tmp/inaguma'))
                except:
                    data_i_tmp = self.load(input_path_list[b])

            if self.use_double_delta:
                data_i = data_i_tmp
            elif self.use_delta:
                data_i = data_i_tmp[:, :self.input_freq * 2]
            else:
                data_i = data_i_tmp[:, :self.input_freq]

            # Frame stacking
            if self.num_stack > 1:
                data_i = stack_frame(data_i, self.num_stack, self.num_skip,
                                     dtype=np.float32)
            frame_num = data_i.shape[0]

            # Splicing
            if self.splice > 1:
                data_i = do_splice(data_i, self.splice, self.num_stack,
                                   dtype=np.float32)

            if self.backend == 'pytorch':
                xs[b, :frame_num, :] = data_i
            elif self.backend == 'chainer':
                xs[b] = data_i.astype(np.float32)
            x_lens[b] = frame_num
            if self.is_test:
                ys[b, 0] = self.df['transcript'][data_indices[b]]
                # NOTE: transcript is not tokenized
            else:
                indices = list(map(int, str_indices_list[b].split(' ')))
                ys[b, :len(indices)] = indices
                y_lens[b] = len(indices)

        batch = {'xs': xs,
                 'ys': ys,
                 'x_lens': x_lens,
                 'y_lens': y_lens,
                 'input_names': input_names}

        return batch
