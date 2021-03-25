# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Custom Sampler."""

import numpy as np
import random

from torch.utils.data.sampler import BatchSampler

from neural_sp.datasets.utils import discourse_bucketing
from neural_sp.datasets.utils import longform_bucketing
from neural_sp.datasets.utils import set_batch_size
from neural_sp.datasets.utils import shuffle_bucketing


class CustomBatchSampler(BatchSampler):

    def __init__(self, df, batch_size, dynamic_batching,
                 shuffle_bucket, discourse_aware, sort_stop_epoch,
                 longform_max_n_frames=0, seed=1):
        """Custom BatchSampler.

        Args:

            df (pandas.DataFrame): dataframe for the main task
            batch_size (int): size of mini-batch
            dynamic_batching (bool): change batch size dynamically in training
            shuffle_bucket (bool): gather similar length of utterances and shuffle them
            discourse_aware (bool): sort in the discourse order
            sort_stop_epoch (int): After sort_stop_epoch, training will revert
                back to a random order
            longform_max_n_frames (int): maximum input length for long-form evaluation

        """
        random.seed(seed)
        np.random.seed(seed)

        self.df = df
        self.batch_size = batch_size
        self.batch_size_tmp = None

        self.dynamic_batching = dynamic_batching
        self.shuffle_bucket = shuffle_bucket
        self.sort_stop_epoch = sort_stop_epoch
        self.discourse_aware = discourse_aware
        self.longform_xmax = longform_max_n_frames

        self._offset = 0
        # NOTE: epoch should not be counted in BatchSampler

        if discourse_aware:
            self.indices_buckets = discourse_bucketing(df, batch_size)
            self._iteration = len(self.indices_buckets)
        elif longform_max_n_frames > 0:
            self.indices_buckets = longform_bucketing(df, batch_size, longform_max_n_frames)
            self._iteration = len(self.indices_buckets)
        elif shuffle_bucket:
            self.indices_buckets = shuffle_bucketing(df, batch_size, self.dynamic_batching, seed)
            self._iteration = len(self.indices_buckets)
        else:
            self.indices = list(df.index)
            # calculate #iteration in advance
            self.calculate_iteration()

    def __len__(self):
        """Number of mini-batches."""
        return self._iteration

    def __iter__(self):
        while True:
            indices, is_new_epoch = self.sample_index()
            if is_new_epoch:
                self.reset()
            yield indices
            if is_new_epoch:
                break

    @property
    def offset(self):
        return self._offset

    def calculate_iteration(self):
        self._iteration = 0
        is_new_epoch = False
        while not is_new_epoch:
            _, is_new_epoch = self.sample_index()
            self._iteration += 1
        random.seed(1)  # reset seed
        self.reset()

    def reset(self, batch_size=None, epoch=None):
        """Reset data counter and offset.

            Args:
                batch_size (int): size of mini-batch
                epoch (int): current epoch

        """
        if batch_size is None:
            batch_size = self.batch_size

        self._offset = 0

        if self.discourse_aware:
            self.indices_buckets = discourse_bucketing(self.df, batch_size)
        elif self.longform_xmax > 0:
            self.indices_buckets = longform_bucketing(self.df, batch_size, self.longform_xmax)
        elif self.shuffle_bucket:
            self.indices_buckets = shuffle_bucketing(self.df, batch_size, self.dynamic_batching, seed=epoch)
        else:
            self.indices = list(self.df.index)
            self.batch_size_tmp = batch_size

    def sample_index(self):
        """Sample data indices of mini-batch.

        Returns:
            indices (np.ndarray): indices of dataframe in the current mini-batch

        """
        if self.discourse_aware or self.longform_xmax > 0 or self.shuffle_bucket:
            indices = self.indices_buckets.pop(0)
            self._offset += len(indices)
            is_new_epoch = (len(self.indices_buckets) == 0)

            if self.shuffle_bucket:
                # Shuffle utterances in mini-batch
                indices = random.sample(indices, len(indices))
        else:
            if self.batch_size_tmp is not None:
                batch_size = self.batch_size_tmp
            else:
                batch_size = self.batch_size

            # Change batch size dynamically
            min_xlen = self.df[self._offset:self._offset + 1]['xlen'].values[0]
            min_ylen = self.df[self._offset:self._offset + 1]['ylen'].values[0]
            batch_size = set_batch_size(batch_size, min_xlen, min_ylen, self.dynamic_batching)
            is_new_epoch = (len(self.indices) <= batch_size)

            if is_new_epoch:
                # Last mini-batch
                indices = self.indices[:]
                self._offset = len(self.df)
            else:
                indices = list(self.df[self._offset:self._offset + batch_size].index)
                self._offset += len(indices)

            # Shuffle utterances in mini-batch
            indices = random.sample(indices, len(indices))

            for i in indices:
                self.indices.remove(i)

        return indices, is_new_epoch
