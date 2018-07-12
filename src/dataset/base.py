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
import codecs
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

        # Load the vocabulary file
        if 'vocab_file_path' in kwargs.keys():
            vocab_count = 0
            with codecs.open(kwargs['vocab_file_path'], 'r', 'utf-8') as f:
                for line in f:
                    if line.strip() != '':
                        vocab_count += 1
            self.num_classes = vocab_count

        if 'vocab_file_path_sub' in kwargs.keys():
            vocab_count_sub = 0
            with codecs.open(kwargs['vocab_file_path_sub'], 'r', 'utf-8') as f:
                for line in f:
                    vocab_count_sub += 1
            self.num_classes_sub = vocab_count_sub

        if 'vocab_file_path_in' in kwargs.keys():
            vocab_count = 0
            with codecs.open(kwargs['vocab_file_path_in'], 'r', 'utf-8') as f:
                for line in f:
                    if line.strip() != '':
                        vocab_count += 1
            self.num_classes_in = vocab_count

        if hasattr(self, 'use_double_delta'):
            if self.use_double_delta:
                self.input_size = self.input_freq * 3
            elif self.use_delta:
                self.input_size = self.input_freq * 2
            else:
                self.input_size = self.input_freq

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

            # print(self.queue.qsize())
            # print(self.queue_size)

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
                if hasattr(self, 'df_in'):
                    min_frame_num_batch = self.df_in[self.offset:self.offset +
                                                     1]['frame_num'].values[0]
                else:
                    min_frame_num_batch = self.df[self.offset:self.offset +
                                                  1]['frame_num'].values[0]
                _batch_size = self.select_batch_size(
                    batch_size, min_frame_num_batch)
                # NOTE: depends on each corpus
            else:
                _batch_size = batch_size

            if len(self.rest) > _batch_size:
                data_indices = list(
                    self.df[self.offset:self.offset + _batch_size].index)
                self.rest -= set(data_indices)
                # NOTE: rest is in uttrance length order when sort_utt == True
                # NOTE: otherwise in name length order when shuffle == False
                self.offset += len(data_indices)
            else:
                # Last mini-batch
                data_indices = list(self.df[self.offset: self.offset +
                                            len(self.rest)].index)
                self._reset()
                is_new_epoch = True
                self._epoch += 1
                if self._epoch == self.sort_stop_epoch:
                    self.sort_utt = False
                    self.shuffle = True

            # Shuffle data in the mini-batch
            # random.shuffle(data_indices)

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

                # Shuffle selected mini-batch
                random.shuffle(data_indices)

            self.offset += len(data_indices)

        return data_indices, is_new_epoch

    def select_batch_size(self, batch_size, min_frame_num_batch):
        if not self.dynamic_batching:
            return batch_size

        if self.corpus == 'timit':
            if min_frame_num_batch < 700:
                batch_size = int(batch_size / 2)
        elif self.model_type in ['nested_attention', 'hierarchical_attention']:
            if min_frame_num_batch <= 800:
                pass
            elif min_frame_num_batch <= 1400:
                batch_size = int(batch_size / 2)
            else:
                batch_size = int(batch_size / 4)
        else:
            if min_frame_num_batch <= 800:
                pass
            elif min_frame_num_batch <= 1600:
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
        """
        Args:
            queue ():
            data_indices_list (np.ndarray):
        """
        # print("Pre-loading started.")
        for i in range(len(data_indices_list)):
            queue.put(self.make_batch(data_indices_list[i]))
        # print("Pre-loading done.")


def split_per_device(x, num_gpus):
    if num_gpus > 1:
        return np.array_split(x, num_gpus, axis=0)
    else:
        return x[np.newaxis]


def _load_feat(path):
    ext = os.path.basename(path).split('.')[-1]
    if ext == 'npy':
        feat = np.load(path)

    elif ext == 'htk':
        with open(path, "rb") as f:
            # Read header
            spam = f.read(12)
            frame_num, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)

            # Load data
            feat_dim = int(sampSize / 4)
            f.seek(12, 0)
            feat = np.fromfile(f, 'f')
            feat = feat.reshape(-1, feat_dim)
            feat.byteswap(True)

    elif ext == 'ark':
        raise NotImplementedError
    else:
        raise ValueError(ext)

    return feat


def load_feat(feat_path):
    try:
        feat = _load_feat(
            feat_path.replace(
                '/n/sd8/inaguma/corpus', '/data/inaguma'))
    except:
        try:
            feat = _load_feat(
                feat_path.replace(
                    '/n/sd8/inaguma/corpus', '/tmp/inaguma'))
        except:
            feat = _load_feat(feat_path)
    return feat
