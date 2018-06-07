#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for loading dataset for the multitask CTC and attention-based model.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
import numpy as np

from utils.dataset.base import Base, load_feat
from utils.io.inputs.frame_stacking import stack_frame
from utils.io.inputs.splicing import do_splice
from utils.parallel import make_parallel
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
                xs (list): input data of size `[B, T, input_size]`
                ys (list): target labels in the main task of size `[B, L]`
                ys_sub (list): target labels in the sub task of size `[B, L_sub]`
                x_lens (list): lengths of inputs of of size `[B]`
                y_lens (list): lengths of target labels in the main task of size `[B]`
                y_lens_sub (list): lengths of target labels in the sub task of size `[B]`
                input_names (list): file names of input data of size `[B]`
        """
        # Load dataset in mini-batch
        feat_paths = np.array(self.df['input_path'][data_indices])
        transcripts = np.array(self.df['transcript'][data_indices])
        transcripts_sub = np.array(self.df_sub['transcript'][data_indices])

        ##############################
        # features
        ##############################
        # Load features in parallel
        # feats = make_parallel(load_feat, feat_paths, core=4)
        feats = [load_feat(p) for p in feat_paths]

        if (not self.use_delta) and (not self.use_double_delta):
            xs = [feat[:, :self.input_freq] for feat in feats]
        else:
            xs = []
            for b in range(len(data_indices)):
                feat = feats[b]

                # Append delta and double-delta features
                max_freq = feat.shape[-1] // 3
                # NOTE: the last dim should be the pitch feature
                if self.input_freq < max_freq and (self.input_freq - 1) % 10 == 0:
                    x = [feat[:, :self.input_freq - 1]]
                    x += [feat[:, max_freq: max_freq + 1]]
                    if self.use_delta:
                        x += [feat[:, max_freq:max_freq + self.input_freq - 1]]
                        x += [feat[:, max_freq * 2: max_freq * 2 + 1]]
                    if self.use_double_delta:
                        x += [feat[:, max_freq * 2:max_freq *
                                   2 + self.input_freq - 1]]
                        x += [feat[:, -1].reshape(-1, 1)]
                else:
                    x = [feat[:, :self.input_freq]]
                    if self.use_delta:
                        x += [feat[:, max_freq:max_freq + self.input_freq]]
                    if self.use_double_delta:
                        x += [feat[:, max_freq *
                                   2:max_freq * 2 + self.input_freq]]
                xs += [np.concatenate(x, axis=-1)]
        # Frame stacking
        if self.num_stack > 1:
            xs = [stack_frame(x, self.num_stack, self.num_skip)
                  for x in xs]

        # Splicing
        if self.splice > 1:
            xs = [do_splice(x, self.splice, self.num_stack) for x in xs]

        #########################
        # transcript
        #########################
        if self.is_test:
            ys = [self.df['transcript'][data_indices[b]]
                  for b in range(len(xs))]
            ys_sub = [self.df_sub['transcript'][data_indices[b]]
                      for b in range(len(xs))]
            # NOTE: transcript is not tokenized
        else:
            ys = [list(map(int, transcripts[b].split(' ')))
                  for b in range(len(xs))]
            ys_sub = [list(map(int, transcripts_sub[b].split(' ')))
                      for b in range(len(xs))]

        input_names = np.array(list(
            map(lambda path: basename(path).split('.')[0],
                np.array(self.df['input_path'][data_indices]))))

        batch = {'xs': xs, 'ys': ys, 'ys_sub': ys_sub,
                 'input_names': input_names}

        return batch
