# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Custom Sampler."""

import numpy as np
import random
import torch.distributed as dist

from neural_sp.datasets.utils import (
    discourse_bucketing,
    longform_bucketing,
    shuffle_bucketing,
    sort_bucketing
)


if dist.is_available():
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler
else:
    from torch.utils.data.sampler import BatchSampler
    sampler = BatchSampler


class CustomBatchSampler(sampler):

    def __init__(self, dataset, distributed, batch_size, dynamic_batching,
                 shuffle_bucket, discourse_aware, sort_stop_epoch,
                 longform_max_n_frames=0, seed=1, resume_epoch=0):
        """Custom BatchSampler.

        Args:
            dataset (Dataset): pytorch Dataset class
            batch_size (int): size of mini-batch
            dynamic_batching (bool): change batch size dynamically in training
            shuffle_bucket (bool): gather similar length of utterances and shuffle them
            discourse_aware (bool): sort in the discourse order
            sort_stop_epoch (int): training will revert back to a random order after sort_stop_epoch
            longform_max_n_frames (int): maximum input length for long-form evaluation
            seed (int): seed for randamization
            resume_epoch (int): epoch to resume training

        """
        if distributed:
            super().__init__(dataset=dataset,
                             num_replicas=dist.get_world_size(),
                             rank=dist.get_rank())
        else:
            self.rank = 0
            self.num_replicas = 1
            self.total_size = len(dataset.df.index) * self.num_replicas

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.df = dataset.df
        self.batch_size = batch_size

        self.dynamic_batching = dynamic_batching
        self.shuffle_bucket = shuffle_bucket
        self.sort_stop_epoch = sort_stop_epoch
        self.discourse_aware = discourse_aware
        self.longform_xmax = longform_max_n_frames

        self._offset = 0
        # NOTE: epoch should not be counted in BatchSampler

        if discourse_aware:
            assert distributed
            self.indices_buckets = discourse_bucketing(self.df, batch_size)
        elif longform_max_n_frames > 0:
            assert not distributed
            self.indices_buckets = longform_bucketing(self.df, batch_size, longform_max_n_frames)
        elif shuffle_bucket:
            self.indices_buckets = shuffle_bucketing(self.df, batch_size, self.dynamic_batching,
                                                     seed=resume_epoch + seed,
                                                     num_replicas=self.num_replicas)
        else:
            self.indices_buckets = sort_bucketing(self.df, batch_size, self.dynamic_batching,
                                                  num_replicas=self.num_replicas)
        self._iteration = len(self.indices_buckets)

    def __len__(self):
        """Number of mini-batches."""
        return self._iteration

    def __iter__(self):
        while True:
            indices, is_new_epoch = self.sample_index()
            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]
            if is_new_epoch:
                self.reset()
            yield indices
            if is_new_epoch:
                break

    @property
    def offset(self):
        return self._offset

    def reset(self, batch_size=None, epoch=0):
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
            self.indices_buckets = shuffle_bucketing(self.df, batch_size, self.dynamic_batching,
                                                     seed=epoch + self.seed,
                                                     num_replicas=self.num_replicas)
        else:
            self.indices_buckets = sort_bucketing(self.df, batch_size, self.dynamic_batching,
                                                  num_replicas=self.num_replicas)

    def sample_index(self):
        """Sample data indices of mini-batch.

        Returns:
            indices (np.ndarray): indices of dataframe in the current mini-batch

        """
        indices = self.indices_buckets.pop(0)
        self._offset += len(indices)
        is_new_epoch = (len(self.indices_buckets) == 0)

        if self.shuffle_bucket:
            # Shuffle utterances in a mini-batch
            indices = random.sample(indices, len(indices))

        return indices, is_new_epoch
