#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for loading dataset for the multitask CTC and attention-based model.
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

    def __getitem__(self, index):
        feature = self._load_npy([self.df['input_path'][index]])
        transcript = self.df['transcript'][index]
        transcript_sub = self.df_sub['transcript'][index]
        return (feature, transcript, transcript_sub)

    def make_batch(self, data_indices):
        """
        Args:
            data_indices (np.ndarray):
        Returns:
            inputs: list of input data of size
                `[num_gpus, B, T_in, input_size]`
            labels: list of target labels in the main task, size
                `[num_gpus, B, T_out]`
            labels_sub: list of target labels in the sub task, size
                `[num_gpus, B, T_out_sub]`
            inputs_seq_len: list of length of inputs of size
                `[num_gpus, B]`
            labels_seq_len: list of length of target labels in the main
                task, size `[num_gpus, B]`
            labels_seq_len_sub: list of length of target labels in the sub
                task, size `[num_gpus, B]`
            input_names: list of file name of input data of size
                `[num_gpus, B]`
        """
        # Load dataset in mini-batch
        input_path_list = np.array(self.df['input_path'][data_indices])
        str_indices_list = np.array(self.df['transcript'][data_indices])
        str_indices_list_sub = np.array(
            self.df_sub['transcript'][data_indices])

        if not hasattr(self, 'input_size'):
            self.input_size = self.input_channel * \
                (1 + int(self.use_delta) + int(self.use_double_delta))
            self.input_size *= self.num_stack
            self.input_size *= self.splice

        # Compute max frame num in mini-batch
        max_inputs_seq_len = max(self.df['frame_num'][data_indices])
        max_inputs_seq_len = math.ceil(max_inputs_seq_len / self.num_skip)

        # Compute max target label length in mini-batch
        max_lables_seq_len = max(
            map(lambda x: len(str(x).split(' ')), str_indices_list)) + 2
        # TODO: fix POS tag (nan -> 'nan')
        max_labels_seq_len_sub = max(
            map(lambda x: len(x.split(' ')), str_indices_list_sub)) + 2
        # NOTE: add <SOS> and <EOS>

        # Initialization
        inputs = np.zeros(
            (len(data_indices), max_inputs_seq_len, self.input_size * self.splice),
            dtype=np.float32)
        labels = np.array(
            [[self.pad_value] * max_lables_seq_len] * len(data_indices))
        labels_sub = np.array(
            [[self.pad_value_sub] * max_labels_seq_len_sub] * len(data_indices))
        inputs_seq_len = np.zeros((len(data_indices),), dtype=np.int32)
        labels_seq_len = np.zeros((len(data_indices),), dtype=np.int32)
        labels_seq_len_sub = np.zeros((len(data_indices),), dtype=np.int32)
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
                data_i = data_i_tmp[:self.input_channel * 2]
            else:
                data_i = data_i_tmp[:self.input_channel]

            # Frame stacking
            if self.num_stack > 1:
                data_i = stack_frame(data_i, self.num_stack, self.num_skip)
            frame_num = data_i.shape[0]

            # Splicing
            if self.splice > 1:
                data_i = do_splice(data_i, self.splice, self.num_stack)

            inputs[i_batch, :frame_num, :] = data_i
            inputs_seq_len[i_batch] = frame_num
            if self.is_test:
                labels[i_batch, 0] = self.df['transcript'][data_indices[i_batch]]
                labels_sub[i_batch,
                           0] = self.df_sub['transcript'][data_indices[i_batch]]
                # NOTE: transcript is not tokenized
            else:
                indices = list(map(int, str_indices_list[i_batch].split(' ')))
                indices_sub = list(
                    map(int, str_indices_list_sub[i_batch].split(' ')))
                label_num = len(indices)
                label_num_sub = len(indices_sub)
                if self.model_type in ['hierarchical_attention', 'nested_attention']:
                    labels[i_batch, 0] = self.sos_index
                    labels[i_batch, 1:label_num + 1] = indices
                    labels[i_batch, label_num + 1] = self.eos_index
                    labels_seq_len[i_batch] = label_num + 2
                    # NOTE: include <SOS> and <EOS>

                    labels_sub[i_batch, 0] = self.sos_index_sub
                    labels_sub[i_batch, 1: label_num_sub + 1] = indices_sub
                    labels_sub[i_batch, label_num_sub + 1] = self.eos_index_sub
                    labels_seq_len_sub[i_batch] = label_num_sub + 2
                elif self.model_type == 'hierarchical_ctc':
                    labels[i_batch, 0:label_num] = indices
                    labels_seq_len[i_batch] = label_num

                    labels_sub[i_batch, 0: label_num_sub] = indices_sub
                    labels_seq_len_sub[i_batch] = label_num_sub
                else:
                    raise TypeError

        return (inputs, labels, labels_sub, inputs_seq_len,
                labels_seq_len, labels_seq_len_sub, input_names)
