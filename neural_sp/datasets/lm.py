# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for loading dataset for language model.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

import logging
import numpy as np
import os
import pandas as pd
import random
import torch.distributed as dist

random.seed(1)
np.random.seed(1)

logger = logging.getLogger(__name__)


class Dataset(object):

    def __init__(self, tsv_path, batch_size, bptt, distributed=False,
                 is_test=False, min_n_tokens=1,
                 shuffle=False, backward=False, serialize=False, corpus=''):
        """A class for loading dataset.

        Args:
            tsv_path (str): path to the dataset tsv file
            batch_size (int): size of mini-batch
            bptt (int): BPTT length
            distributed (bool): use distributed training
            is_test (bool): flag for test mode
            min_n_tokens (int): exclude utterances shorter than this value
            shuffle (bool): shuffle utterances per epoch.
            backward (bool): flip all text in the corpus
            serialize (bool): serialize text according to contexts in dialogue
            corpus (str): name of corpus

        """
        super(Dataset, self).__init__()

        if distributed:
            self.rank = dist.get_rank()
            self.num_replicas = dist.get_world_size()
        else:
            self.rank = 0
            self.num_replicas = 1

        self.epoch = 0
        self.iteration = 0
        self.offset = 0

        self.set = os.path.basename(tsv_path).split('.')[0]
        self.is_test = is_test
        self.batch_size = batch_size
        if self.num_replicas > 1 and self.rank == 0:
            logger.info(f"Batch size is automatically increased from {batch_size} to {batch_size * self.num_replicas}.")
        self.bptt = bptt
        self.batch_size_tmp = None
        self.bptt_tmp = None
        self.eos = 2
        self.shuffle = shuffle
        self.backward = backward
        assert bptt >= 2

        # Load dataset tsv file
        chunk = pd.read_csv(tsv_path, encoding='utf-8',
                            delimiter='\t', chunksize=1000000)
        self.df = pd.concat(chunk)
        self.df = self.df.loc[:, ['utt_id', 'speaker', 'feat_path',
                                  'xlen', 'xdim', 'text', 'token_id', 'ylen', 'ydim']]

        # Remove inappropriate utterances
        n_utts = len(self.df)
        print(f"Original utterance num: {n_utts}")
        self.df = self.df[self.df.apply(lambda x: x['ylen'] >= min_n_tokens, axis=1)]
        print(f"Removed {n_utts - len(self.df)} utterances (threshold)")

        # Sort tsv records
        if shuffle:
            assert not serialize
            self.df = self.df.reindex(np.random.permutation(self.df.index))
        elif serialize:
            assert not shuffle
            assert corpus == 'swbd'
            self.df['session'] = self.df['speaker'].apply(lambda x: str(x).split('-')[0])
            self.df['onset'] = self.df['utt_id'].apply(lambda x: int(x.split('_')[-1].split('-')[0]))
            self.df = self.df.sort_values(by=['session', 'onset'], ascending=True)
        else:
            self.df = self.df.sort_values(by='utt_id', ascending=True)

        # Concatenate into a single sentence
        self.concat_ids = self.concat_utterances(self.df)

    def concat_utterances(self, df):
        batch_size = self.batch_size_tmp if self.batch_size_tmp is not None else self.batch_size

        indices = list(df.index)
        if self.backward:
            indices = indices[::-1]
        concat_ids = []
        for i in indices:
            assert df['token_id'][i] != ''
            concat_ids += [self.eos] + list(map(int, df['token_id'][i].split()))
        concat_ids += [self.eos]  # for the last sentence
        # NOTE: <sos> and <eos> have the same index

        # Reshape
        n_utts_org = len(concat_ids)
        n_utts = len(concat_ids) // (batch_size * self.num_replicas) * batch_size * self.num_replicas
        concat_ids = concat_ids[:n_utts]
        logger.debug(f"Removed {n_utts_org - len(concat_ids)} tokens / {n_utts_org} tokens")
        concat_ids = np.array(concat_ids).reshape((self.num_replicas, batch_size, -1))

        return concat_ids

    @property
    def epoch_detail(self):
        """Percentage of the current epoch."""
        return float(self.offset * self.batch_size * self.num_replicas) / len(self)

    def reset(self, batch_size=None, bptt=None, is_new_epoch=False):
        """Reset data counter and offset.

        Args:
            batch_size (int): size of mini-batch
            bptt (int): BPTT length

        """
        self.batch_size_tmp = batch_size
        self.bptt_tmp = bptt

        if self.shuffle:
            self.df = self.df.reindex(np.random.permutation(self.df.index))

        self.concat_ids = self.concat_utterances(self.df)
        self.offset = 0

    def __len__(self):
        return len(self.concat_ids.reshape((-1,)))

    def __iter__(self):
        """Generate each mini-batch.

        Returns:
            ys (np.ndarray): target labels in the main task of size `[B, bptt]`
            is_new_epoch (bool): flag for the end of the current epoch

        """
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        bptt = self.bptt_tmp if self.bptt_tmp is not None else self.bptt

        ys = self.concat_ids[self.rank, :, self.offset:self.offset + bptt + 1]
        self.offset += bptt
        # NOTE: the last token in ys must be feeded as inputs in the next mini-batch

        is_new_epoch = self.offset >= self.concat_ids.shape[-1] - 1
        if is_new_epoch:
            self.reset()
            self.epoch += 1

        return ys, is_new_epoch
