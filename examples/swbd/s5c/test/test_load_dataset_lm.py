#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

sys.path.append(os.path.abspath('../../../../'))
from src.dataset.loader_lm import Dataset
from src.utils.measure_time_func import measure_time


class TestLoadDataset(unittest.TestCase):

    def test(self):

        # data_type
        # self.check(label_type='word', data_type='train')
        self.check(label_type='word', data_type='train_fisher')
        self.check(label_type='word', data_type='dev')
        self.check(label_type='word', data_type='eval2000_swbd')
        self.check(label_type='word', data_type='eval2000_ch')

        # label_type
        self.check(label_type='character')

    @measure_time
    def check(self, label_type, data_type='dev'):

        print('========================================')
        print('  label_type: %s' % label_type)
        print('  data_type: %s' % data_type)
        print('========================================')

        dataset = Dataset(
            corpus='swbd',
            data_save_path='/n/sd8/inaguma/corpus/swbd/kaldi',
            data_size='', data_type=data_type,
            label_type=label_type, batch_size=64, max_epoch=1,
            shuffle=False, sort_utt=True,
            reverse=False, sort_stop_epoch=None,
            num_gpus=1, num_enque=None)

        print('=> Loading mini-batch...')
        if label_type == 'word':
            map_fn = dataset.idx2word
        else:
            map_fn = dataset.idx2char

        for batch, is_new_epoch in dataset:
            str_ref = batch['ys'][0]
            if not dataset.is_test:
                str_ref = map_fn(str_ref)

            print('----- %s (epoch: %.3f, batch: %d) -----' %
                  (batch['input_names'][0], dataset.epoch_detail, len(batch['ys'])))
            print(str_ref)

            if dataset.epoch_detail >= 1:
                break


if __name__ == '__main__':
    unittest.main()
