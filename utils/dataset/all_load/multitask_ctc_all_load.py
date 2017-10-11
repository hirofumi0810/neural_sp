#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for loading dataset for the multi-task CTC model.
   In this class, all data will be loaded at once.
   You can use only the single GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
import random
import numpy as np

from utils.dataset.base import Base
from utils.io.inputs.splicing import do_splice


class DatasetBase(Base):

    def __init__(self, *args, **kwargs):
        super(DatasetBase, self).__init__(*args, **kwargs)

    def __next__(self, batch_size=None):
        """Generate each mini-batch.
        Args:
            batch_size (int, optional): the size of mini-batch
        Returns:
            A tuple of `(inputs, labels, inputs_seq_len, labels_seq_len, input_names)`
                inputs: list of input data of size
                    `[B, T, input_dim]`
                labels_main: list of target labels in the main task, of size
                    `[B, T]`
                labels_sub: list of target labels in the sub task, of size
                    `[B, T]`
                inputs_seq_len: list of length of inputs of size
                    `[B]`
                input_names: list of file name of input data of size
                    `[B]`
            is_new_epoch (bool): If true, one epoch is finished
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

        # Compute max frame num in mini-batch
        max_frame_num = max(map(lambda x: x.shape[0],
                                self.input_list[data_indices]))

        # Compute max target label length in mini-batch
        max_seq_len_main = max(map(len, self.label_main_list[data_indices]))
        max_seq_len_sub = max(map(len, self.label_sub_list[data_indices]))

        # Initialization
        inputs = np.zeros(
            (len(data_indices), max_frame_num,
             self.input_list[0].shape[-1] * self.splice), dtype=np.float32)
        labels_main = np.array(
            [[self.padded_value] * max_seq_len_main] * len(data_indices),
            dtype=np.int32)
        labels_sub = np.array(
            [[self.padded_value] * max_seq_len_sub] * len(data_indices),
            dtype=np.int32)
        inputs_seq_len = np.zeros((len(data_indices),), dtype=np.int32)
        input_names = np.array(list(
            map(lambda path: basename(path).split('.')[0],
                np.take(self.input_paths, data_indices, axis=0))))

        # Set values of each data in mini-batch
        for i_batch, x in enumerate(data_indices):
            data_i = self.input_list[x]
            frame_num, input_size = data_i.shape

            # Splicing
            data_i = data_i.reshape(1, frame_num, input_size)
            data_i = do_splice(data_i,
                               splice=self.splice,
                               batch_size=1).reshape(frame_num, -1)

            inputs[i_batch, :frame_num, :] = data_i
            labels_main[i_batch, :len(
                self.label_main_list[x])] = self.label_main_list[x]
            labels_sub[i_batch, :len(self.label_sub_list[x])
                       ] = self.label_sub_list[x]
            inputs_seq_len[i_batch] = frame_num

        self.iteration += len(data_indices)

        return (inputs, labels_main, labels_sub, inputs_seq_len,
                input_names), self.is_new_epoch
