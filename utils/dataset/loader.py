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
        """
        Args:
            data_indices (np.ndarray):
        Returns:
            inputs: list of input data of size
                `[num_gpus, B, T_in, input_size]`
            labels: list of target labels of size
                `[num_gpus, B, T_out]`
            inputs_seq_len: list of length of inputs of size
                `[num_gpus, B]`
            labels_seq_len: list of length of target labels of size
                `[num_gpus, B]`
            input_names: list of file name of input data of size
                `[num_gpus, B]`
        """
        input_path_list = np.array(self.df['input_path'][data_indices])
        str_indices_list = np.array(self.df['transcript'][data_indices])

        if not hasattr(self, 'input_size'):
            if self.use_double_delta:
                self.input_size = self.input_channel * 3
            elif self.use_delta:
                self.input_size = self.input_channel * 2
            else:
                self.input_size = self.input_channel
            self.input_size *= self.num_stack
            self.input_size *= self.splice

        # Compute max frame num in mini-batch
        max_frame_num = max(self.df['frame_num'][data_indices])
        max_frame_num = math.ceil(max_frame_num / self.num_skip)

        # Compute max target label length in mini-batch
        max_label_num = max(
            map(lambda x: len(x.split(' ')), str_indices_list)) + 2
        # NOTE: add <SOS> and <EOS>

        # Initialization
        if self.backend == 'pytorch':
            inputs = np.zeros(
                (len(data_indices), max_frame_num, self.input_size),
                dtype=np.float32)
        elif self.backend == 'chainer':
            inputs = [None] * len(data_indices)
        if self.is_test:
            labels = np.array(
                [[self.pad_value] * max_label_num] * len(data_indices))
        else:
            labels = np.array(
                [[self.pad_value] * max_label_num] * len(data_indices), dtype=np.int32)
        inputs_seq_len = np.zeros((len(data_indices),), dtype=np.int32)
        labels_seq_len = np.zeros((len(data_indices),), dtype=np.int32)
        input_names = np.array(list(
            map(lambda path: basename(path).split('.')[0],
                np.array(self.df['input_path'][data_indices]))))

        # Set values of each data in mini-batch
        for i_batch in range(len(data_indices)):
            # Load input data
            try:
                data_i_tmp = self.load(
                    input_path_list[i_batch].replace(
                        '/n/sd8/inaguma/corpus', '/data/inaguma'))
            except:
                data_i_tmp = self.load(input_path_list[i_batch])

            if self.use_double_delta:
                data_i = data_i_tmp
            elif self.use_delta:
                data_i = data_i_tmp[:, :self.input_channel * 2]
            else:
                data_i = data_i_tmp[:, :self.input_channel]

            # Frame stacking
            if self.num_stack > 1:
                data_i = stack_frame(data_i, self.num_stack, self.num_skip)
            frame_num = data_i.shape[0]

            # Splicing
            if self.splice > 1:
                data_i = do_splice(data_i, self.splice, self.num_stack)

            if self.backend == 'pytorch':
                inputs[i_batch, :frame_num, :] = data_i
            elif self.backend == 'chainer':
                inputs[i_batch] = data_i
            inputs_seq_len[i_batch] = frame_num
            if self.is_test:
                labels[i_batch, 0] = self.df['transcript'][data_indices[i_batch]]
                # NOTE: transcript is not tokenized
            else:
                indices = list(map(int, str_indices_list[i_batch].split(' ')))
                label_num = len(indices)
                if self.model_type == 'attention':
                    labels[i_batch, 0] = self.sos_index
                    labels[i_batch, 1:label_num + 1] = indices
                    labels[i_batch, label_num + 1] = self.eos_index
                    labels_seq_len[i_batch] = label_num + 2
                    # NOTE: include <SOS> and <EOS>
                elif self.model_type == 'ctc':
                    labels[i_batch, 0:label_num] = indices
                    labels_seq_len[i_batch] = label_num
                else:
                    raise TypeError

        return inputs, labels, inputs_seq_len, labels_seq_len, input_names
