#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

sys.path.append(os.path.abspath('../../../../../../'))
from examples.timit.s5.exp.dataset.load_dataset import Dataset
from utils.measure_time_func import measure_time


class TestLoadDataset(unittest.TestCase):

    def test(self):

        # data_type
        self.check(label_type='phone61', data_type='dev')
        self.check(label_type='phone61', data_type='test')

        # label_type
        self.check(label_type='phone61')
        self.check(label_type='phone48')
        self.check(label_type='phone39')

        # sort
        self.check(label_type='phone61', sort_utt=True)
        self.check(label_type='phone61', sort_utt=True,
                   sort_stop_epoch=2)
        self.check(label_type='phone61', shuffle=True)

        # frame stacking
        self.check(label_type='phone61', frame_stacking=True)

        # splicing
        self.check(label_type='phone61', splice=11)

    @measure_time
    def check(self, label_type, data_type='dev', backend='pytorch',
              shuffle=False, sort_utt=False, sort_stop_epoch=None,
              frame_stacking=False, splice=1):

        print('========================================')
        print('  backend: %s' % backend)
        print('  label_type: %s' % label_type)
        print('  data_type: %s' % data_type)
        print('  shuffle: %s' % str(shuffle))
        print('  sort_utt: %s' % str(sort_utt))
        print('  sort_stop_epoch: %s' % str(sort_stop_epoch))
        print('  frame_stacking: %s' % str(frame_stacking))
        print('  splice: %d' % splice)
        print('========================================')

        num_stack = 3 if frame_stacking else 1
        num_skip = 3 if frame_stacking else 1
        dataset = Dataset(
            data_save_path='/n/sd8/inaguma/corpus/timit/kaldi',
            backend=backend,
            input_freq=41, use_delta=True, use_double_delta=True,
            data_type=data_type, label_type=label_type,
            batch_size=32, max_epoch=1,
            splice=splice, num_stack=num_stack, num_skip=num_skip,
            shuffle=shuffle,
            sort_utt=sort_utt, sort_stop_epoch=sort_stop_epoch,
            tool='htk',
            num_enque=None)

        print('=> Loading mini-batch...')

        for batch, is_new_epoch in dataset:
            if data_type == 'train' and backend == 'pytorch':
                for i in range(len(batch['xs'])):
                    if batch['xs'].shape[1] < batch['ys'].shape[1]:
                        raise ValueError(
                            'input length must be longer than label length.')

            if dataset.is_test:
                str_ref = batch['ys'][0][0]
            else:
                str_ref = dataset.idx2phone(
                    batch['ys'][0][:batch['y_lens'][0]])

            print('----- %s (epoch: %.3f, batch: %d) -----' %
                  (batch['input_names'][0], dataset.epoch_detail, len(batch['xs'])))
            print(str_ref)
            print('x_lens: %d' % (batch['x_lens'][0] * num_stack))
            if not dataset.is_test:
                print('y_lens: %d' % batch['y_lens'][0])


if __name__ == '__main__':
    unittest.main()
