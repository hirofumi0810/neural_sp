#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for all dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import logging
import numpy as np
import random
import six
import time
from torch.multiprocessing import Process
from torch.multiprocessing import Queue

logger = logging.getLogger('training')


class Base(object):

    def __init__(self):
        self.epoch = 0
        self.iteration = 0
        self.offset = 0

        # for multiprocessing
        self._epoch = 0

        # Setting for multiprocessing
        self.preloading_process = None
        self.queue = Queue()
        self.queue_size = 0

    def count_vocab_size(self, dict_path):
        vocab_count = 0
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                if line.strip() != '':
                    vocab_count += 1
        return vocab_count

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        feature = self.load_npy(self.df['input_path'][index])
        transcript = self.df['transcript'][index]
        return (feature, transcript)

    def __iter__(self):
        """Returns self."""
        return self

    @property
    def pad_value(self):
        return -1 if not self.is_test else None

    @property
    def epoch_detail(self):
        # Floating point version of epoch
        return self.epoch + self.offset / len(self)

    @property
    def current_batch_size(self):
        return self._current_batch_size

    def __next__(self, batch_size=None):
        """Generate each mini-batch.

        Args:
            batch_size (int, optional): the size of mini-batch
        Returns:
            batch (tuple):
            is_new_epoch (bool): If true, 1 epoch is finished

        """
        if batch_size is None:
            batch_size = self.batch_size

        if self.num_enques is None:
            if self.max_epoch is not None and self.epoch >= self.max_epoch:
                raise StopIteration
            # NOTE: max_epoch == None means infinite loop

            data_indices, is_new_epoch = self.sample_index(batch_size)
            self._current_batch_size = len(data_indices)
            batch = self.make_batch(data_indices)
            self.iteration += len(data_indices)
        else:
            # Clean up multiprocessing
            if self.preloading_process is not None and self.queue_size == 0:
                self.preloading_process.terminate()
                self.preloading_process.join()

            if self.max_epoch is not None and self.epoch >= self.max_epoch:
                # Clean up multiprocessing
                self.preloading_process.terminate()
                self.preloading_process.join()
                raise StopIteration
            # NOTE: max_epoch == None means infinite loop

            # Enqueue mini-batches
            if self.queue_size == 0:
                self.data_indices_list = []
                self.is_new_epoch_list = []
                for _ in six.moves.range(self.num_enques):
                    data_indices, is_new_epoch = self.sample_index(batch_size)
                    self._current_batch_size = len(data_indices)
                    self.data_indices_list.append(data_indices)
                    self.is_new_epoch_list.append(is_new_epoch)
                self.preloading_process = Process(self.preloading_loop,
                                                  args=(self.queue, self.data_indices_list))
                self.preloading_process.start()
                self.queue_size += self.num_enques
                time.sleep(3)

            # print(self.queue.qsize())
            # print(self.queue_size)

            self.iteration += len(self.data_indices_list[self.num_enques - self.queue_size])
            self.queue_size -= 1
            batch = self.queue.get()
            is_new_epoch = self.is_new_epoch_list.pop(0)

        if is_new_epoch:
            self.epoch += 1

        return batch, is_new_epoch

    def next(self, batch_size=None):
        # For python2
        return self.__next__(batch_size)

    def sample_index(self, batch_size):
        """Sample data indices of mini-batch.

        Args:
            batch_size (int): the size of mini-batch
        Returns:
            data_indices (np.ndarray):
            is_new_epoch (bool):

        """
        is_new_epoch = False

        if self.sort_by_input_length or not self.shuffle:
            if self.sort_by_input_length:
                # Change batch size dynamically
                if hasattr(self, 'df_in'):
                    min_num_frames_batch = self.df_in[self.offset:self.offset + 1]['x_len'].values[0]
                else:
                    min_num_frames_batch = self.df[self.offset:self.offset + 1]['x_len'].values[0]
                _batch_size = self.select_batch_size(batch_size, min_num_frames_batch)
            else:
                _batch_size = batch_size

            if len(self.rest) > _batch_size:
                data_indices = list(self.df[self.offset:self.offset + _batch_size].index)
                self.rest -= set(data_indices)
                # NOTE: rest is in uttrance length order when sort_by_input_length == True
                # NOTE: otherwise in name length order when shuffle == False
                self.offset += len(data_indices)
            else:
                # Last mini-batch
                data_indices = list(self.df[self.offset: self.offset + len(self.rest)].index)
                self._reset()
                is_new_epoch = True
                self._epoch += 1
                if self._epoch == self.sort_stop_epoch:
                    self.sort_by_input_length = False
                    self.shuffle = True

            # Sort in the descending order for pytorch
            data_indices = data_indices[::-1]
        else:
            # Randomly sample uttrances
            if len(self.rest) > batch_size:
                data_indices = random.sample(list(self.rest), batch_size)
                self.rest -= set(data_indices)
            else:
                # Last mini-batch
                data_indices = list(self.rest)
                self._reset()
                is_new_epoch = True
                self._epoch += 1

            self.offset += len(data_indices)

        return data_indices, is_new_epoch

    def select_batch_size(self, batch_size, min_num_frames_batch):
        if not self.dynamic_batching:
            return batch_size

        if min_num_frames_batch <= 800:
            pass
        elif min_num_frames_batch <= 1600:
            batch_size = int(batch_size / 2)
        else:
            batch_size = int(batch_size / 4)

        if batch_size < 1:
            batch_size = 1

        return batch_size

    def reset(self):
        self._reset()

        self.queue = Queue()
        self.queue_size = 0

        # Clean up multiprocessing
        if self.preloading_process is not None:
            self.preloading_process.terminate()
            self.preloading_process.join()

    def _reset(self):
        """Reset data counter and offset."""
        self.rest = set(list(self.df.index))
        self.offset = 0

    def preloading_loop(self, queue, data_indices_list):
        """.

        Args:
            queue ():
            data_indices_list (np.ndarray):

        """
        # print("Pre-loading started.")
        for i in six.moves.range(len(data_indices_list)):
            queue.put(self.make_batch(data_indices_list[i]))
        # print("Pre-loading done.")
