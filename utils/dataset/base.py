#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for all dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random
import numpy as np
from struct import unpack
from multiprocessing import Queue, Process


class Base(object):

    def __init__(self, *args, **kwargs):
        self.epoch = 0
        self.iteration = 0
        self.is_new_epoch = False
        self.offset = 0

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

    def __next__(self, batch_size=None):
        """Generate each mini-batch.
        Args:
            batch_size (int, optional): the size of mini-batch
        Returns:
            A tuple of batch (tuple):
            is_new_epoch (bool): If true, 1 epoch is finished
        """
        # Clean up multiprocessing
        if self.preloading_process is not None and self.queue_size == 0:
            self.preloading_process.terminate()
            self.preloading_process.join()

        if self.max_epoch is not None and self.epoch >= self.max_epoch:
            # Clean up multiprocessing
            self.preloading_process.terminate()
            self.preloading_process.join()
            raise StopIteration
        # NOTE: max_epoch = None means infinite loop

        # Enqueue mini-batches
        if self.queue_size == 0:
            self.data_indices_list = []
            for _ in range(self.num_enque):
                self.data_indices_list.append(self.sample_index(batch_size))
            self.preloading_process = Process(
                target=self.preloading_loop, args=(self.queue, self.data_indices_list))
            self.preloading_process.start()
            self.queue_size += self.num_enque
            time.sleep(5)

        # print(self.queue.qsize())
        # print(self.queue_size)

        self.iteration += len(
            self.data_indices_list[self.num_enque - self.queue_size])
        self.queue_size -= 1

        return self.queue.get(), self.is_new_epoch

    def next(self, batch_size=None):
        # For python2
        return self.__next__(batch_size)

    def sample_index(self, batch_size=None):
        """Sample data indices of mini-batch.
        Args:
            batch_size (int, optional):
        Returns:
            data_indices (np.ndarray):
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Reset flag
        if self.is_new_epoch:
            self.is_new_epoch = False

        if self.sort_utt or not self.shuffle:
            if len(self.rest) > batch_size:
                data_indices = self.df[batch_size *
                                       self.offset:batch_size * (self.offset + 1)].index
                data_indices = list(data_indices)
                self.rest -= set(list(data_indices))
                # NOTE: rest is in uttrance length order when sort_utt == True
                # NOTE: otherwise in name length order when shuffle == False
                self.offset += 1
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
        else:
            # Randomly sample uttrances
            if len(self.rest) > batch_size:
                data_indices = random.sample(list(self.rest), batch_size)
                self.rest -= set(data_indices)
                self.offset += 1
            else:
                # Last mini-batch
                data_indices = list(self.rest)
                self.reset()
                self.is_new_epoch = True
                self.epoch += 1

                # Shuffle selected mini-batch
                random.shuffle(data_indices)

        return data_indices

    def reset(self):
        """Reset data counter and offset."""
        self.rest = set(list(self.df.index))
        self.offset = 0

    def load_npy(self, path):
        """Load npy files.
        Args:
            path (string):
        Returns:
            input_data (np.ndarray): A tensor of size (frame_num, feature_dim)
        """
        return np.load(path)

    def load_htk(htk_path):
        """Load each HTK file.
        Args:
            htk_path (string): path to a HTK file
        Returns:
            input_data (np.ndarray): A tensor of size (frame_num, feature_dim)
        """
        # print('...Reading: %s' % htk_path)
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
