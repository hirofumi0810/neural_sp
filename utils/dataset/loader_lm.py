#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for loading dataset for language models.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
import numpy as np

from utils.dataset.base import Base

# NOTE: Loading numpy is faster than loading htk


class DatasetBase(Base):

    def __init__(self, *args, **kwargs):
        super(DatasetBase, self).__init__(*args, **kwargs)

    def make_batch(self, data_indices):
        """Create mini-batch per step.
        Args:
            data_indices (np.ndarray):
        Returns:
            batch (dict):
                ys (list): target labels in the main task of size `[B, L]`
                input_names (list): file names of input data of size `[B]`
        """
        # Load dataset in mini-batch
        transcripts = np.array(self.df['transcript'][data_indices])

        #########################
        # transcript
        #########################
        if self.is_test:
            ys = [self.df['transcript'][data_indices[b]]
                  for b in range(len(data_indices))]
            # NOTE: transcript is not tokenized
        else:
            ys = [list(map(int, transcripts[b].split(' ')))
                  for b in range(len(data_indices))]

        input_names = list(
            map(lambda path: basename(path).split('.')[0],
                self.df['input_path'][data_indices]))

        return {'ys': ys, 'input_names': input_names}
