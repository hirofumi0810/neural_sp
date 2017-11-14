#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

sys.path.append(os.path.abspath('../../../../'))
from examples.timit.data.load_dataset_attention import Dataset
from utils.io.labels.character import Idx2char
from utils.io.labels.phone import Idx2phone
from utils.measure_time_func import measure_time


class TestLoadDatasetAttention(unittest.TestCase):

    def test(self):

        # data_type
        self.check(label_type='phone61', data_type='train')
        self.check(label_type='phone61', data_type='dev')
        self.check(label_type='phone61', data_type='test')

        # label_type
        self.check(label_type='phone61')
        self.check(label_type='character')
        self.check(label_type='character_capital_divide')

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
    def check(self, label_type, data_type='dev',
              shuffle=False, sort_utt=False, sort_stop_epoch=None,
              frame_stacking=False, splice=1):

        print('========================================')
        print('  label_type: %s' % label_type)
        print('  data_type: %s' % data_type)
        print('  shuffle: %s' % str(shuffle))
        print('  sort_utt: %s' % str(sort_utt))
        print('  sort_stop_epoch: %s' % str(sort_stop_epoch))
        print('  frame_stacking: %s' % str(frame_stacking))
        print('  splice: %d' % splice)
        print('========================================')

        vocab_file_path = '../../metrics/vocab_files/' + label_type + '.txt'

        num_stack = 3 if frame_stacking else 1
        num_skip = 3 if frame_stacking else 1
        dataset = Dataset(
            data_type=data_type, label_type=label_type,
            vocab_file_path=vocab_file_path,
            batch_size=64, max_epoch=1,
            splice=splice, num_stack=num_stack, num_skip=num_skip,
            shuffle=shuffle,
            sort_utt=sort_utt, sort_stop_epoch=sort_stop_epoch)

        print('=> Loading mini-batch...')
        if label_type in ['character', 'character_capital_divide']:
            map_fn = Idx2char(vocab_file_path)
        else:
            map_fn = Idx2phone(vocab_file_path)

        for data, is_new_epoch in dataset:
            inputs, labels, inputs_seq_len, labels_seq_len, input_names = data

            if data_type != 'test':
                str_true = map_fn(labels[0][0][:labels_seq_len[0][0]])
            else:
                str_true = labels[0][0][0]

            print('----- %s ----- (epoch: %.3f)' %
                  (input_names[0][0], dataset.epoch_detail))
            print(inputs[0][0].shape)
            print(str_true)


if __name__ == '__main__':
    unittest.main()
