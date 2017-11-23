#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

sys.path.append(os.path.abspath('../../../../'))
from examples.csj.data.load_dataset_hierarchical import Dataset
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.measure_time_func import measure_time


class TestLoadDatasetHierarchical(unittest.TestCase):

    def test(self):

        # data_type
        self.check(label_type='word_freq5', label_type_sub='kana',
                   data_type='train')
        self.check(label_type='word_freq5', label_type_sub='kana',
                   data_type='dev')
        self.check(label_type='word_freq5', label_type_sub='kana',
                   data_type='eval1')
        # self.check(label_type='word_freq5', label_type_sub='kana',
        #            data_type='eval2')
        # self.check(label_type='word_freq5', label_type_sub='kana',
        #            data_type='eval3')

        # label_type
        self.check(label_type='word_freq5', label_type_sub='kanji')
        # self.check(label_type='kanji', label_type_sub='kana')

    @measure_time
    def check(self, label_type, label_type_sub, data_type='dev',
              data_size='subset',
              shuffle=False, sort_utt=True, sort_stop_epoch=None,
              frame_stacking=False, splice=1, num_gpus=1):

        print('========================================')
        print('  label_type: %s' % label_type)
        print('  label_type_sub: %s' % label_type_sub)
        print('  data_type: %s' % data_type)
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
        vocab_file_path_sub = '../../metrics/vocab_files/' + \
            label_type_sub + '_' + data_size + '.txt'

        num_stack = 3 if frame_stacking else 1
        num_skip = 3 if frame_stacking else 1
        dataset = Dataset(
            model_type='hierarchical_attention',
            data_type=data_type, data_size=data_size,
            label_type=label_type, label_type_sub=label_type_sub,
            batch_size=64,
            vocab_file_path=vocab_file_path,
            vocab_file_path_sub=vocab_file_path_sub,
            max_epoch=1, splice=splice,
            num_stack=num_stack, num_skip=num_skip,
            shuffle=shuffle,
            sort_utt=sort_utt, reverse=True, sort_stop_epoch=sort_stop_epoch,
            num_gpus=num_gpus, save_format='numpy')

        print('=> Loading mini-batch...')
        idx2word = Idx2word(vocab_file_path, space_mark='_')
        idx2char = Idx2char(vocab_file_path_sub)

        for data, is_new_epoch in dataset:
            inputs, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub, input_names = data

            if data_type == 'train':
                for i, l in zip(inputs, labels):
                    if len(i) < len(l):
                        raise ValueError(
                            'input length must be longer than label length.')

            if num_gpus > 1:
                for inputs_gpu in inputs:
                    print(inputs_gpu.shape)

            if dataset.is_test:
                str_true = labels[0][0]
                str_true_sub = labels_sub[0][0]
            else:
                str_true = idx2word(
                    labels.data[0][:labels_seq_len.data[0]])
                str_true_sub = idx2char(
                    labels_sub.data[0][:labels_seq_len_sub.data[0]])

            print('----- %s (epoch: %.3f) -----' %
                  (input_names[0], dataset.epoch_detail))
            print(inputs.data.numpy().shape)
            # print(labels.data.shape)
            print('-' * 20)
            print(str_true)
            print('-' * 10)
            print(str_true_sub)
            print('-' * 20)

            if dataset.epoch_detail >= 0.05:
                break


if __name__ == '__main__':
    unittest.main()
