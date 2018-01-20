#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for all dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import random
import numpy as np
from struct import unpack
from torch.multiprocessing import Queue, Process
import logging
logger = logging.getLogger('training')


class Base(object):

    def __init__(self, *args, **kwargs):
        self.epoch = 0
        self.iteration = 0
        self.offset = 0

        # for multiprocessing
        self._epoch = 0

        # Setting for multiprocessing
        self.preloading_process = None
        self.queue = Queue()
        self.queue_size = 0

        # Read the vocabulary file
        vocab_count = 0
        with open(kwargs['vocab_file_path'], 'r') as f:
            for line in f:
                vocab_count += 1
        self.num_classes = vocab_count

        if 'vocab_file_path_sub' in kwargs.keys():
            vocab_count_sub = 0
            with open(kwargs['vocab_file_path_sub'], 'r') as f:
                for line in f:
                    vocab_count_sub += 1
            self.num_classes_sub = vocab_count_sub

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
    def sos_index(self):
        return self.num_classes + 1

    @property
    def sos_index_sub(self):
        return self.num_classes_sub + 1

    @property
    def eos_index(self):
        return self.num_classes

    @property
    def eos_index_sub(self):
        return self.num_classes_sub

    @property
    def pad_value(self):
        return self.sos_index if not self.is_test else None

    @property
    def pad_value_sub(self):
        return self.sos_index_sub if not self.is_test else None

    @property
    def epoch_detail(self):
        # Floating point version of epoch
        return self.iteration / len(self)

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

        if self.num_enque is None:
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
                for _ in range(self.num_enque):
                    data_indices, is_new_epoch = self.sample_index(batch_size)
                    self._current_batch_size = len(data_indices)
                    self.data_indices_list.append(data_indices)
                    self.is_new_epoch_list.append(is_new_epoch)
                self.preloading_process = Process(
                    target=self.preloading_loop,
                    args=(self.queue, self.data_indices_list))
                self.preloading_process.start()
                self.queue_size += self.num_enque
                time.sleep(3)

            # logging.info(self.queue.qsize())
            # logging.info(self.queue_size)

            self.iteration += len(
                self.data_indices_list[self.num_enque - self.queue_size])
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

        if self.sort_utt or not self.shuffle:
            if self.sort_utt:
                # Change batch size dynamically
                min_frame_num_batch = self.df[self.offset:self.offset +
                                              1]['frame_num'].values[0]
                batch_size_tmp = self.select_batch_size(
                    batch_size, min_frame_num_batch)
                # NOTE: this depends on each corpus
            else:
                batch_size_tmp = batch_size

            if len(self.rest) > batch_size_tmp:
                df_tmp = self.df[self.offset:self.offset + batch_size_tmp]
                data_indices = list(df_tmp.index)
                self.rest -= set(data_indices)
                # NOTE: rest is in uttrance length order when sort_utt == True
                # NOTE: otherwise in name length order when shuffle == False
                self.offset += len(data_indices)
            else:
                # Last mini-batch
                data_indices = list(self.rest)
                self._reset()
                is_new_epoch = True
                self._epoch += 1
                if self._epoch == self.sort_stop_epoch:
                    self.sort_utt = False
                    self.shuffle = True

            # Shuffle data in the mini-batch
            # random.shuffle(data_indices)
            # NOTE: stop shuffle now!!!
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

                # Shuffle selected mini-batch
                random.shuffle(data_indices)

        return data_indices, is_new_epoch

    def select_batch_size(self, batch_size, min_frame_num_batch):
        raise NotImplementedError

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

    def load(self, path):
        ext = os.path.basename(path).split('.')[-1]

        if ext == 'npy':
            return self._load_npy(path)
        elif ext == 'htk':
            return self._load_htk(path)

    def _load_npy(self, path):
        """Load npy files.
        Args:
            path (string):
        Returns:
            input_data (np.ndarray): A tensor of size (frame_num, feature_dim)
        """
        return np.load(path)

    def _load_htk(htk_path):
        """Load each HTK file.
        Args:
            htk_path (string): path to a HTK file
        Returns:
            input_data (np.ndarray): A tensor of size (frame_num, feature_dim)
        """
        with open(htk_path, "rb") as f:
            # Read header
            spam = f.read(12)
            frame_num, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)

            # for debug
            # print(frame_num)  # frame num
            # print(sampPeriod)  # 10ms
            # print(sampSize)  # feature dim * 4 (byte)
            # print(parmKind)

            # Read data
            feature_dim = int(sampSize / 4)
            f.seek(12, 0)
            input_data = np.fromfile(f, 'f')
            input_data = input_data.reshape(-1, feature_dim)
            input_data.byteswap(True)

        return input_data

    def split_per_device(self, x, num_gpus):
        if num_gpus > 1:
            return np.array_split(x, num_gpus, axis=0)
        else:
            return x[np.newaxis]

    def preloading_loop(self, queue, data_indices_list):
        """
        Args:
            queue ():
            data_indices_list (np.ndarray):
        """
        # print("Pre-loading started.")
        for i in range(len(data_indices_list)):
            queue.put(self.make_batch(data_indices_list[i]))
        # print("Pre-loading done.")
