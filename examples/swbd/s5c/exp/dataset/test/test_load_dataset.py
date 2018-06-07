#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

sys.path.append(os.path.abspath('../../../../../../'))
from examples.swbd.s5c.exp.dataset.load_dataset import Dataset
from utils.measure_time_func import measure_time


class TestLoadDataset(unittest.TestCase):

    def test(self):

        # data_type
        self.check(data_type='train')
        self.check(data_type='dev')
        self.check(data_type='eval2000_swbd')
        self.check(data_type='eval2000_ch')

        # label_type
        self.check(label_type='word')
        self.check(label_type='character_capital_divide')

        # sort
        self.check(sort_utt=True)
        self.check(sort_utt=True, sort_stop_epoch=2)
        self.check(shuffle=True)

        # frame stacking
        self.check(frame_stacking=True)

        # splicing
        self.check(splice=11)

        # multi-GPU
        # self.check(num_gpus=8)

    @measure_time
    def check(self, label_type='character', data_type='dev',
              data_size='300h', backend='pytorch',
              shuffle=False, sort_utt=True, sort_stop_epoch=None,
              frame_stacking=False, splice=1, num_gpus=1):

        print('========================================')
        print('  backend: %s' % backend)
        print('  label_type: %s' % label_type)
        print('  data_type: %s' % data_type)
        print('  data_size: %s' % data_size)
        print('  shuffle: %s' % str(shuffle))
        print('  sort_utt: %s' % str(sort_utt))
        print('  sort_stop_epoch: %s' % str(sort_stop_epoch))
        print('  frame_stacking: %s' % str(frame_stacking))
        print('  splice: %d' % splice)
        print('  num_gpus: %d' % num_gpus)
        print('========================================')

        num_stack = 3 if frame_stacking else 1
        num_skip = 3 if frame_stacking else 1
        dataset = Dataset(
            data_save_path='/n/sd8/inaguma/corpus/swbd/kaldi',
            backend=backend,
            input_freq=80, use_delta=False, use_double_delta=False,
            data_type=data_type, data_size=data_size,
            label_type=label_type, batch_size=64,
            max_epoch=1, splice=splice,
            num_stack=num_stack, num_skip=num_skip,
            min_frame_num=40, shuffle=shuffle,
            sort_utt=sort_utt, reverse=True, sort_stop_epoch=sort_stop_epoch,
            num_gpus=num_gpus, tool='htk', num_enque=None)

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
                  (batch['input_names'][0], dataset.epoch_detail, len(batch['xs'])))
            print(str_ref)
            print('x_lens: %d' % (len(batch['xs'][0]) * num_stack))

            if dataset.epoch_detail >= 1:
                break


if __name__ == '__main__':
    unittest.main()
