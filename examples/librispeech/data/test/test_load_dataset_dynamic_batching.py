#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

sys.path.append(os.path.abspath('../../../../'))
from examples.librispeech.data.load_dataset_dynamic_batching import Dataset
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.measure_time_func import measure_time


class TestLoadDatasetDynamicBatching(unittest.TestCase):

    def test(self):

        # data_type
        self.check(label_type='character')

    @measure_time
    def check(self, label_type, data_size='100h',
              shuffle=False, sort_utt=True, sort_stop_epoch=None,
              frame_stacking=False, splice=1, num_gpus=1):

        print('========================================')
        print('  label_type: %s' % label_type)
        print('  data_size: %s' % data_size)
        print('  shuffle: %s' % str(shuffle))
        print('  sort_utt: %s' % str(sort_utt))
        print('  sort_stop_epoch: %s' % str(sort_stop_epoch))
        print('  frame_stacking: %s' % str(frame_stacking))
        print('  splice: %d' % splice)
        print('  num_gpus: %d' % num_gpus)
        print('========================================')

        vocab_file_path = '../../metrics/vocab_files/' + \
            label_type + '_' + data_size + '.txt'

        num_stack = 3 if frame_stacking else 1
        num_skip = 3 if frame_stacking else 1
        dataset = Dataset(
            model_type='attention',
            data_size=data_size,
            label_type=label_type, batch_size=64,
            vocab_file_path=vocab_file_path,
            max_epoch=1, splice=splice,
            num_stack=num_stack, num_skip=num_skip,
            shuffle=shuffle,
            sort_utt=sort_utt, reverse=True, sort_stop_epoch=sort_stop_epoch,
            num_gpus=num_gpus)

        print('=> Loading mini-batch...')
        if 'word' in label_type:
            map_fn = Idx2word(vocab_file_path, space_mark=' ')
        else:
            map_fn = Idx2char(vocab_file_path)

        for data, is_new_epoch in dataset:
            inputs, labels, inputs_seq_len, labels_seq_len, input_names = data

            for i in range(len(inputs)):
                if inputs.shape[1] < labels.shape[1]:
                    raise ValueError(
                        'input length must be longer than label length.')

            str_true = map_fn(labels[0][:labels_seq_len[0]])

            print('----- %s (epoch: %.3f) -----' %
                  (input_names[0], dataset.epoch_detail))
            print(str_true)
            print('inputs_seq_len: %d' % inputs_seq_len[0])
            print('labels_seq_len: %d' % labels_seq_len[0])

            if dataset.epoch_detail >= 0.1:
                break


if __name__ == '__main__':
    unittest.main()
