#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for all dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import copy
import logging
import random
import time
from torch.multiprocessing import Process
from torch.multiprocessing import Queue

random.seed(1)

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
        vocab_count = 1  # for <blank>
        with codecs.open(dict_path, 'r', 'utf-8') as f:
            for line in f:
                if line.strip() != '':
                    vocab_count += 1
        return vocab_count

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        raise NotImplementedError()

    def __iter__(self):
        """Returns self."""
        return self

    @property
    def epoch_detail(self):
        # percentage of the current epoch
        return self.offset / len(self)

    def next(self, batch_size=None):
        """Generate each mini-batch.

        Args:
            batch_size (int): size of mini-batch
        Returns:
            batch (tuple):
            is_new_epoch (bool): If true, 1 epoch is finished

        """
        if batch_size is None:
            batch_size = self.batch_size

        if self.n_ques is None:
            if self.max_epoch is not None and self.epoch >= self.max_epoch:
                raise StopIteration
            # NOTE: max_epoch == None means infinite loop

            data_indices, is_new_epoch = self.sample_index(batch_size)
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
                self.df_indices_list = []
                self.is_new_epoch_list = []
                for _ in range(self.n_ques):
                    data_indices, is_new_epoch = self.sample_index(batch_size)
                    self.df_indices_list.append(data_indices)
                    self.is_new_epoch_list.append(is_new_epoch)
                self.preloading_process = Process(self.preloading_loop,
                                                  args=(self.queue, self.df_indices_list))
                self.preloading_process.start()
                self.queue_size += self.n_ques
                time.sleep(3)

            self.iteration += len(self.df_indices_list[self.n_ques - self.queue_size])
            self.queue_size -= 1
            batch = self.queue.get()
            is_new_epoch = self.is_new_epoch_list.pop(0)

        if is_new_epoch:
            self.epoch += 1

        return batch, is_new_epoch

    def sample_index(self, batch_size):
        """Sample data indices of mini-batch.

        Args:
            batch_size (int): the size of mini-batch
        Returns:
            data_indices (np.ndarray):
            is_new_epoch (bool):

        """
        is_new_epoch = False

        if self.discourse_aware:
            n_utt = min(self.n_utt_session_dict_epoch.keys())
            assert self.utt_offset < n_utt
            data_indices = [self.df[self.session_offset_dict[session_id] + self.utt_offset:self.session_offset_dict[session_id] + self.utt_offset + 1].index[0]
                            for session_id in self.n_utt_session_dict_epoch[n_utt][:batch_size]]

            self.utt_offset += 1
            if self.utt_offset == n_utt:
                if len(self.n_utt_session_dict_epoch[n_utt][batch_size:]) > 0:
                    self.n_utt_session_dict_epoch[n_utt] = self.n_utt_session_dict_epoch[n_utt][batch_size:]
                else:
                    self.n_utt_session_dict_epoch.pop(n_utt)
                self.utt_offset = 0

                # reset for the new epoch
                if len(self.n_utt_session_dict_epoch.keys()) == 0:
                    self.n_utt_session_dict_epoch = copy.deepcopy(self.n_utt_session_dict)
                    is_new_epoch = True
                    self._epoch += 1

        elif self.sort_by_input_length or not self.shuffle:
            if self.sort_by_input_length:
                # Change batch size dynamically
                min_xlen = self.df[self.offset:self.offset + 1]['xlen'].values[0]
                min_ylen = self.df[self.offset:self.offset + 1]['ylen'].values[0]
                batch_size_tmp = self.select_batch_size(batch_size, min_xlen, min_ylen)
            else:
                batch_size_tmp = batch_size

            if len(self.rest) > batch_size_tmp:
                data_indices = list(self.df[self.offset:self.offset + batch_size_tmp].index)
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

    def select_batch_size(self, batch_size, min_xlen, min_ylen):
        if not self.dynamic_batching:
            return batch_size

        if min_xlen <= 800:
            pass
        elif min_xlen <= 1600 or 70 < min_ylen <= 100:
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

    def preloading_loop(self, queue, df_indices_list):
        """.

        Args:
            queue ():
            df_indices_list (np.ndarray):

        """
        for i in range(len(df_indices_list)):
            queue.put(self.make_batch(df_indices_list[i]))
