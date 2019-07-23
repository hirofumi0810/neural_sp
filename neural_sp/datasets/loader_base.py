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
        return 1 - (len(self.df_indices) / len(self))

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

            df_indices, is_new_epoch = self.sample_index(batch_size)
            batch = self.make_batch(df_indices)
            self.iteration += len(df_indices)
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
                    df_indices, is_new_epoch = self.sample_index(batch_size)
                    self.df_indices_list.append(df_indices)
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
            batch_size (int): size of mini-batch
        Returns:
            df_indices (np.ndarray):
            is_new_epoch (bool):

        """
        is_new_epoch = False

        if self.discourse_aware:
            n_utt = min(self.n_utt_session_dict_epoch.keys())
            assert self.utt_offset < n_utt
            df_indices = [self.df[self.session_offset_dict[session_id] + self.utt_offset:self.session_offset_dict[session_id] + self.utt_offset + 1].index[0]
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

        else:
            if len(self.df_indices) > batch_size:
                if self.shuffle_bucket:
                    # Sample offset randomly
                    offset = random.sample(self.df_indices, 1)[0]
                    df_indices_offset = self.df_indices.index(offset)
                else:
                    offset = self.offset

                # Change batch size dynamically
                if self.sort_by is not None:
                    # assert offset in list(self.df.index)
                    min_xlen = self.df[offset:offset + 1]['xlen'].values[0]
                    min_ylen = self.df[offset:offset + 1]['ylen'].values[0]
                    batch_size = self.set_batch_size(batch_size, min_xlen, min_ylen)

                if self.shuffle_bucket:
                    if len(self.df_indices[df_indices_offset:]) < batch_size:
                        df_indices = self.df_indices[df_indices_offset:][:]
                    else:
                        df_indices = self.df_indices[df_indices_offset:df_indices_offset + batch_size][:]
                else:
                    df_indices = list(self.df[offset:offset + batch_size].index)
                    self.offset += len(df_indices)

                # Shuffle uttrances in mini-batch
                df_indices = random.sample(df_indices, len(df_indices))
                for i in df_indices:
                    self.df_indices.remove(i)
            else:
                # Last mini-batch
                df_indices = self.df_indices[:]
                self._reset()
                is_new_epoch = True
                self._epoch += 1
                if self._epoch == self.sort_stop_epoch:
                    self.sort_by = None

        return df_indices, is_new_epoch

    def set_batch_size(self, batch_size, min_xlen, min_ylen):
        if not self.dynamic_batching:
            return batch_size

        if min_xlen <= 800:
            pass
        elif min_xlen <= 1600 or 80 < min_ylen <= 100:
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
        self.df_indices = list(self.df.index)
        self.offset = 0

    def preloading_loop(self, queue, df_indices_list):
        """

        Args:
            queue ():
            df_indices_list (np.ndarray):

        """
        for i in range(len(df_indices_list)):
            queue.put(self.make_batch(df_indices_list[i]))
