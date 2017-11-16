#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

sys.path.append(os.path.abspath('../../../../'))
from examples.librispeech.data.load_dataset_hierarchical_attention import Dataset
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.measure_time_func import measure_time


class TestLoadDatasetHierarchicalAttention(unittest.TestCase):

    def test(self):

        # data_type
        self.check(label_type='word_freq5', label_type_sub='character',
                   data_type='train')
        self.check(label_type='word_freq5', label_type_sub='character',
                   data_type='dev_clean')
        # self.check(label_type='word_freq5', label_type_sub='character',
        #            data_type='dev_other')
        self.check(label_type='word_freq5', label_type_sub='character',
                   data_type='test_clean')
        # self.check(label_type='word_freq5', label_type_sub='character',
        #            data_type='test_other')

        # label_type
        self.check(label_type='word_freq1', label_type_sub='character')
        self.check(label_type='word_freq10', label_type_sub='character')
        self.check(label_type='word_freq15', label_type_sub='character')

    @measure_time
    def check(self, label_type, label_type_sub,
              data_type='dev_clean', data_size='100h',
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
        if label_type == 'character':
            vocab_file_path_sub = '../../metrics/vocab_files/character.txt'
        else:
            vocab_file_path_sub = '../../metrics/vocab_files/' + \
                label_type_sub + '_' + data_size + '.txt'

        num_stack = 3 if frame_stacking else 1
        num_skip = 3 if frame_stacking else 1
        dataset = Dataset(
            data_type=data_type, data_size=data_size,
            label_type=label_type, label_type_sub=label_type_sub,
            batch_size=64,
            vocab_file_path=vocab_file_path,
            vocab_file_path_sub=vocab_file_path_sub,
            max_epoch=2, splice=splice,
            num_stack=num_stack, num_skip=num_skip,
            shuffle=shuffle,
            sort_utt=sort_utt, reverse=True, sort_stop_epoch=sort_stop_epoch,
            num_gpus=num_gpus)

        print('=> Loading mini-batch...')

        idx2word = Idx2word(vocab_file_path)
        idx2char = Idx2char(vocab_file_path_sub)

        for data, is_new_epoch in dataset:
            inputs, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub, input_names = data

            if data_type == 'train':
                for i, l in zip(inputs[0], labels[0]):
                    if len(i) < len(l):
                        raise ValueError(
                            'input length must be longer than label length.')

            if num_gpus > 1:
                for inputs_gpu in inputs:
                    print(inputs_gpu.shape)

            if dataset.is_test:
                str_true = labels[0][0][0]
                str_true_sub = labels[0][0][0]
            else:
                word_list = idx2word(labels[0][0][:labels_seq_len[0][0]])
                str_true = '_'.join(word_list)
                str_true_sub = idx2char(
                    labels_sub[0][0][:labels_seq_len_sub[0][0]])

            print('----- %s (epoch: %.3f) -----' %
                  (input_names[0][0], dataset.epoch_detail))
            print(inputs[0].shape)
            print(labels[0].shape)
            print(str_true)
            print('---')
            print(str_true_sub)

            if dataset.epoch_detail >= 0.05:
                break


if __name__ == '__main__':
    unittest.main()
