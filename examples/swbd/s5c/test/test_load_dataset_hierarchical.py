#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

sys.path.append(os.path.abspath('../../../../'))
from src.dataset.loader_hierarchical import Dataset
from src.utils.measure_time_func import measure_time


class TestLoadDatasetHierarchical(unittest.TestCase):

    def test(self):

        # data_type
        self.check(label_type='word', label_type_sub='character',
                   data_type='train')
        self.check(label_type='word', label_type_sub='character',
                   data_type='dev')
        self.check(label_type='word', label_type_sub='character',
                   data_type='eval2000_swbd')
        self.check(label_type='word', label_type_sub='character',
                   data_type='eval2000_ch')

        # label_type
        self.check(label_type='word', label_type_sub='phone_wb')
        self.check(label_type='character', label_type_sub='phone_wb')

    @measure_time
    def check(self, label_type, label_type_sub, data_type='dev', data_size='swbd',
              shuffle=False, sort_utt=True, sort_stop_epoch=None):

        print('========================================')
        print('  label_type: %s' % label_type)
        print('  label_type_sub: %s' % label_type_sub)
        print('  data_type: %s' % data_type)
        print('  shuffle: %s' % str(shuffle))
        print('  sort_utt: %s' % str(sort_utt))
        print('  sort_stop_epoch: %s' % str(sort_stop_epoch))
        print('========================================')

        dataset = Dataset(
            corpus='swbd',
            data_save_path='/n/sd8/inaguma/corpus/swbd/kaldi',
            input_freq=80, use_delta=False, use_double_delta=False,
            data_size=data_size, data_type=data_type,
            label_type=label_type, label_type_sub=label_type_sub,
            batch_size=64, max_epoch=1,
            shuffle=shuffle, sort_utt=sort_utt,
            reverse=True, sort_stop_epoch=sort_stop_epoch,
            tool='htk', num_enque=None)

        print('=> Loading mini-batch...')
        if label_type == 'word':
            map_fn = dataset.idx2word
        elif 'character' in label_type:
            map_fn = dataset.idx2char
        if 'character' in label_type_sub:
            map_fn_sub = dataset.idx2char
        elif 'phone' in label_type_sub:
            map_fn_sub = dataset.idx2phone

        for batch, is_new_epoch in dataset:
            str_ref = batch['ys'][0]
            str_ref_sub = batch['ys_sub'][0]
            if not dataset.is_test:
                str_ref = map_fn(str_ref)
                str_ref_sub = map_fn_sub(str_ref_sub)

            print('----- %s (epoch: %.3f, batch: %d) -----' %
                  (batch['input_names'][0], dataset.epoch_detail, len(batch['xs'])))
            print('=' * 20)
            print(str_ref)
            print('-' * 10)
            print(str_ref_sub)
            print('x_lens: %d' % (len(batch['xs'][0])))

            if dataset.epoch_detail >= 1:
                break


if __name__ == '__main__':
    unittest.main()
