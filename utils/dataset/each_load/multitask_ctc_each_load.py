#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the multitask CTC model.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
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
        label_main_i = np.array(self.label_main_paths[index])
        label_sub_i = np.array(self.label_sub_paths[index])
        return (input_i, label_main_i, label_sub_i)

    def __next__(self, batch_size=None):
        """Generate each mini-batch.
        Args:
            batch_size (int, optional): the size of mini-batch
        Returns:
            A tuple of `(inputs, labels, inputs_seq_len, labels_seq_len, input_names)`
                inputs: list of input data of size
                    `[num_gpu, B, T, input_dim]`
                labels_main: list of target labels in the main task, of size
                    `[num_gpu, B, T]`
                labels_sub: list of target labels in the sub task, of size
                    `[num_gpu, B, T]`
                inputs_seq_len: list of length of inputs of size
                    `[num_gpu, B]`
                input_names: list of file name of input data of size
                    `[num_gpu, B]`
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

        if not self.is_test:
            self.padded_value = -1
        else:
            self.padded_value = None
        # TODO(hirofumi): move this

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
        input_list = np.array(list(
            map(lambda path: np.load(path),
                np.take(self.input_paths, data_indices, axis=0))))
        label_main_list = np.array(list(
            map(lambda path: np.load(path),
                np.take(self.label_main_paths, data_indices, axis=0))))
        label_sub_list = np.array(list(
            map(lambda path: np.load(path),
                np.take(self.label_sub_paths, data_indices, axis=0))))

        if not hasattr(self, 'input_size'):
            self.input_size = input_list[0].shape[1]
            if self.num_stack is not None and self.num_skip is not None:
                self.input_size *= self.num_stack

        # Frame stacking
        input_list = stack_frame(input_list,
                                 self.input_paths[data_indices],
                                 self.frame_num_dict,
                                 self.num_stack,
                                 self.num_skip,
                                 progressbar=False)

        # Compute max frame num in mini-batch
        max_frame_num = max(map(lambda x: x.shape[0], input_list))

        # Compute max target label length in mini-batch
        max_seq_len_main = max(map(len, label_main_list))
        max_seq_len_sub = max(map(len, label_sub_list))

        # Initialization
        inputs = np.zeros(
            (len(data_indices), max_frame_num, self.input_size * self.splice),
            dtype=np.float32)
        labels_main = np.array(
            [[self.padded_value] * max_seq_len_main] * len(data_indices))
        labels_sub = np.array(
            [[self.padded_value] * max_seq_len_sub] * len(data_indices))
        inputs_seq_len = np.zeros((len(data_indices),), dtype=np.int32)
        input_names = list(
            map(lambda path: basename(path).split('.')[0],
                np.take(self.input_paths, data_indices, axis=0)))

        # Set values of each data in mini-batch
        for i_batch in range(len(data_indices)):
            data_i = input_list[i_batch]
            frame_num, input_size = data_i.shape

            # Splicing
            data_i = data_i.reshape(1, frame_num, input_size)
            data_i = do_splice(data_i,
                               splice=self.splice,
                               batch_size=1).reshape(frame_num, -1)

            inputs[i_batch, :frame_num, :] = data_i
            if self.is_test:
                labels_main[i_batch, 0] = label_main_list[i_batch]
            else:
                labels_main[i_batch, :len(
                    label_main_list[i_batch])] = label_main_list[i_batch]
            labels_sub[i_batch, :len(
                label_sub_list[i_batch])] = label_sub_list[i_batch]
            inputs_seq_len[i_batch] = frame_num

        ###############
        # Multi-GPUs
        ###############
        if self.num_gpu > 1:
            # Now we split the mini-batch data by num_gpu
            inputs = np.array_split(inputs, self.num_gpu, axis=0)
            labels_main = np.array_split(labels_main, self.num_gpu, axis=0)
            labels_sub = np.array_split(labels_sub, self.num_gpu, axis=0)
            inputs_seq_len = np.array_split(
                inputs_seq_len, self.num_gpu, axis=0)
            input_names = np.array_split(input_names, self.num_gpu, axis=0)
        else:
            inputs = inputs[np.newaxis, :, :, :]
            labels_main = labels_main[np.newaxis, :, :]
            labels_sub = labels_sub[np.newaxis, :, :]
            inputs_seq_len = inputs_seq_len[np.newaxis, :]
            input_names = np.array(input_names)[np.newaxis, :]

        self.iteration += len(data_indices)

        return (inputs, labels_main, labels_sub, inputs_seq_len,
                input_names), self.is_new_epoch
