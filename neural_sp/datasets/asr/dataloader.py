# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Custom DataLoader."""

import numpy as np
from torch.utils.data import DataLoader


class CustomDataLoader(DataLoader):

    def __init__(self, dataset, batch_sampler, sort_stop_epoch,
                 num_workers=0, collate_fn=None, pin_memory=False,
                 timeout=0, worker_init_fn=None):

        super().__init__(dataset=dataset,
                         shuffle=False,
                         sampler=None,
                         batch_sampler=batch_sampler,
                         num_workers=num_workers,
                         collate_fn=collate_fn,
                         pin_memory=pin_memory,
                         drop_last=False,
                         timeout=timeout,
                         worker_init_fn=worker_init_fn)
        # NOTE: dynamic batch size and shuffling are controlled in batch_sampler

        np.random.seed(1)

        # keep meta information
        self.input_dim = dataset._input_dim
        self.vocab = dataset._vocab
        self.vocab_sub1 = dataset._vocab_sub1
        self.vocab_sub2 = dataset._vocab_sub2
        self.corpus = dataset._corpus
        self.set = dataset._set
        self.unit = dataset._unit
        self.unit_sub1 = dataset._unit_sub1
        self.unit_sub2 = dataset._unit_sub2
        self.idx2token = dataset._idx2token
        self.token2idx = dataset._token2idx

        self.epoch = 0  # counter
        self.sort_stop_epoch = sort_stop_epoch

    def __len__(self):
        """Number of utterances."""
        return len(self.dataset)

    @property
    def epoch_detail(self):
        """Progress of the current epoch."""
        epoch_ratio = self.batch_sampler.offset / len(self)
        # NOTE: this is not accurate when num_workers > 0
        return epoch_ratio

    @property
    def n_frames(self):
        return self.dataset.n_frames

    def reset(self, batch_size=None, batch_size_type=None, is_new_epoch=False):
        """Reset data counter and offset.

        Args:
            batch_size (int): size of mini-batch
            batch_size_type (str): type of batch size counting
            is_new_epoch (bool): flag for new epoch

        """
        if is_new_epoch:
            self.epoch += 1

            # shuffle the whole data per epoch (sort -> shuffle)
            if self.epoch >= self.sort_stop_epoch:
                self.batch_sampler.shuffle_bucket = True

                # This changes not only the order of buckets but also how buckets are constructed
                self.batch_sampler.df = self.batch_sampler.df.reindex(
                    np.random.permutation(self.batch_sampler.df.index))
                for i in range(1, 3):
                    if getattr(self.batch_sampler, 'df_sub' + str(i)) is not None:
                        setattr(self.batch_sampler, 'df_sub' + str(i),
                                getattr(self.batch_sampler, 'df_sub' + str(i)).reindex(self.batch_sampler.df.index).reset_index())

                # Re-indexing
                self.batch_sampler.df = self.batch_sampler.df.reset_index()

        self.batch_sampler.reset(batch_size, batch_size_type, epoch=self.epoch)
