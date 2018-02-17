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
        """Create mini-batch per step.
        Args:
            data_indices (np.ndarray):
        Returns:
            batch (dict):
                xs (np.ndarray): input data of size
                    `[B, T_in, input_size]`
                ys (np.ndarray): target labels in the main task of size
                    `[B, T_out]`
                ys_sub (np.ndarray): target labels in the sub task of size
                    `[B, T_out_sub]`
                x_lens (np.ndarray): lengths of inputs of of size
                    `[B]`
                y_lens (np.ndarray): lengths of target labels in the main task of size
                    `[B]`
                y_lens_sub (np.ndarray): lengths of target labels in the sub task of size
                    `[B]`
                input_names (np.ndarray): file names of input data of size
                    `[B]`
        """
        # Load dataset in mini-batch
        input_path_list = np.array(self.df['input_path'][data_indices])
        str_indices_list = np.array(self.df['transcript'][data_indices])
        str_indices_list_sub = np.array(
            self.df_sub['transcript'][data_indices])

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
            map(lambda x: len(str(x).split(' ')), str_indices_list)) + 2
        # TODO: fix POS tag (nan -> 'nan')
        max_labels_seq_len_sub = max(
            map(lambda x: len(x.split(' ')), str_indices_list_sub)) + 2
        # NOTE: add <SOS> and <EOS>

        # Initialization
        if self.backend == 'pytorch':
            xs = np.zeros(
                (len(data_indices), max_frame_num, self.input_size * self.splice),
                dtype=np.float32)
        elif self.backend == 'chainer':
            xs = [None] * len(data_indices)
        if self.is_test:
            ys = np.array(
                [[self.pad_value] * max_label_num] * len(data_indices))
            ys_sub = np.array(
                [[self.pad_value_sub] * max_labels_seq_len_sub] * len(data_indices))
        else:
            ys = np.array(
                [[self.pad_value] * max_label_num] * len(data_indices), dtype=np.int32)
            ys_sub = np.array(
                [[self.pad_value_sub] * max_labels_seq_len_sub] * len(data_indices), dtype=np.int32)
        x_lens = np.zeros((len(data_indices),), dtype=np.int32)
        y_lens = np.zeros((len(data_indices),), dtype=np.int32)
        y_lens_sub = np.zeros((len(data_indices),), dtype=np.int32)
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
                xs[i_batch, :frame_num, :] = data_i
            elif self.backend == 'chainer':
                xs[i_batch] = data_i
            x_lens[i_batch] = frame_num
            if self.is_test:
                ys[i_batch, 0] = self.df['transcript'][data_indices[i_batch]]
                ys_sub[i_batch,
                       0] = self.df_sub['transcript'][data_indices[i_batch]]
                # NOTE: transcript is not tokenized
            else:
                indices = list(map(int, str_indices_list[i_batch].split(' ')))
                indices_sub = list(
                    map(int, str_indices_list_sub[i_batch].split(' ')))
                label_num = len(indices)
                label_num_sub = len(indices_sub)
                if self.model_type == 'hierarchical_ctc':
                    ys[i_batch, 0:label_num] = indices
                    y_lens[i_batch] = label_num

                    ys_sub[i_batch, 0: label_num_sub] = indices_sub
                    y_lens_sub[i_batch] = label_num_sub
                else:
                    ys[i_batch, 0] = self.sos_index
                    ys[i_batch, 1:label_num + 1] = indices
                    ys[i_batch, label_num + 1] = self.eos_index
                    y_lens[i_batch] = label_num + 2
                    # NOTE: include <SOS> and <EOS>

                    ys_sub[i_batch, 0] = self.sos_index_sub
                    ys_sub[i_batch, 1: label_num_sub + 1] = indices_sub
                    ys_sub[i_batch, label_num_sub + 1] = self.eos_index_sub
                    y_lens_sub[i_batch] = label_num_sub + 2

        batch = {'xs': xs,
                 'ys': ys,
                 'ys_sub': ys_sub,
                 'x_lens': x_lens,
                 'y_lens': y_lens,
                 'y_lens_sub': y_lens_sub,
                 'input_names': input_names}

        return batch
