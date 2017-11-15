#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for loading dataset for the attention-based model.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
import math
import random
import numpy as np

from utils.dataset.base import Base
from utils.io.inputs.frame_stacking import stack_frame
from utils.io.inputs.splicing import do_splice


class DatasetBase(Base):

    def __init__(self, *args, **kwargs):
        super(DatasetBase, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        input_i = np.array(self.input_paths[index])
        label_i = np.array(self.label_paths[index])
        return (input_i, label_i)

    def __next__(self, batch_size=None):
        """Generate each mini-batch.
        Args:
            batch_size (int, optional): the size of mini-batch
        Returns:
            A tuple of `(inputs, labels, inputs_seq_len, labels_seq_len, input_names)`
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
            is_new_epoch (bool): If true, 1 epoch is finished
        """
        if self.max_epoch is not None and self.epoch >= self.max_epoch:
            raise StopIteration
        # NOTE: max_epoch = None means infinite loop

        if batch_size is None:
            batch_size = self.batch_size

        # reset
        if self.is_new_epoch:
            self.is_new_epoch = False

        if self.sort_utt:
            # Sort all uttrances by length
            if len(self.rest) > batch_size:
                data_indices = sorted(list(self.rest))[:batch_size]
                self.rest -= set(data_indices)
                # NOTE: rest is uttrance length order
            else:
                # Last mini-batch
                data_indices = list(self.rest)
                self.reset()
                self.is_new_epoch = True
                self.epoch += 1
                if self.epoch == self.sort_stop_epoch:
                    self.sort_utt = False
                    self.shuffle = True

            # Shuffle data in the mini-batch
            random.shuffle(data_indices)

        elif self.shuffle:
            # Randomly sample uttrances
            if len(self.rest) > batch_size:
                data_indices = random.sample(list(self.rest), batch_size)
                self.rest -= set(data_indices)
            else:
                # Last mini-batch
                data_indices = list(self.rest)
                self.reset()
                self.is_new_epoch = True
                self.epoch += 1

                # Shuffle selected mini-batch
                random.shuffle(data_indices)

        else:
            if len(self.rest) > batch_size:
                data_indices = sorted(list(self.rest))[:batch_size]
                self.rest -= set(data_indices)
                # NOTE: rest is in name order
            else:
                # Last mini-batch
                data_indices = list(self.rest)
                self.reset()
                self.is_new_epoch = True
                self.epoch += 1

        # Load dataset in mini-batch
        input_list = self._load_npy(
            np.take(self.input_paths, data_indices, axis=0))
        label_list = self._load_npy(
            np.take(self.label_paths, data_indices, axis=0))

        if not hasattr(self, 'input_size'):
            self.input_size = input_list[0].shape[1]
            self.input_size *= self.num_stack
            self.input_size *= self.splice

        # Compute max frame num in mini-batch
        max_frame_num = max(map(lambda x: x.shape[0], input_list))
        max_frame_num = math.ceil(max_frame_num / self.num_skip)

        # Compute max target label length in mini-batch
        max_seq_len = max(map(len, label_list)) + 2
        # NOTE: + <SOS> and <EOS>

        # Initialization
        inputs = np.zeros(
            (len(data_indices), max_frame_num, self.input_size),
            dtype=np.float32)
        labels = np.array(
            [[self.att_padded_value] * max_seq_len] * len(data_indices))
        inputs_seq_len = np.zeros((len(data_indices),), dtype=np.int32)
        labels_seq_len = np.zeros((len(data_indices),), dtype=np.int32)
        input_names = np.array(list(
            map(lambda path: basename(path).split('.')[0],
                np.take(self.input_paths, data_indices, axis=0))))

        # Set values of each data in mini-batch
        for i_batch in range(len(data_indices)):
            data_i = input_list[i_batch]

            # Frame stacking
            data_i = stack_frame(data_i, self.num_stack, self.num_skip)
            frame_num = data_i.shape[0]

            # Splicing
            data_i = do_splice(data_i, self.splice, self.num_stack)

            inputs[i_batch, : frame_num, :] = data_i
            if self.is_test:
                labels[i_batch, 0] = label_list[i_batch]
                # NOTE: transcript is saved as string
            else:
                labels[i_batch, 0] = self.sos_index
                labels[i_batch, 1:len(label_list[i_batch]) +
                       1] = label_list[i_batch]
                labels[i_batch, len(label_list[i_batch]) + 1] = self.eos_index
            inputs_seq_len[i_batch] = frame_num
            labels_seq_len[i_batch] = len(label_list[i_batch]) + 2
            # NOTE: include <SOS> and <EOS>

        # Now we split the mini-batch data by num_gpus
        inputs = self.split_per_device(inputs, self.num_gpus)
        labels = self.split_per_device(labels, self.num_gpus)
        inputs_seq_len = self.split_per_device(inputs_seq_len, self.num_gpus)
        labels_seq_len = self.split_per_device(labels_seq_len, self.num_gpus)
        input_names = self.split_per_device(input_names, self.num_gpus)

        self.iteration += len(data_indices)

        return (inputs, labels, inputs_seq_len, labels_seq_len,
                input_names), self.is_new_epoch
