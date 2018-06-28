#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

sys.path.append(os.path.abspath('../../../../'))
from src.dataset.loader_p2w import Dataset
from src.utils.measure_time_func import measure_time


class TestLoadDataset(unittest.TestCase):

    def test(self):

        # Change vocab
        # self.check(label_type_in='phone_wb', label_type_out='word',
        #            data_type='train', data_size='all', vocab='aps_other')
        # self.check(label_type_in='phone_wb', label_type_out='character_wb',
        #            data_type='train', data_size='all', vocab='aps_other')

        # data_type
        self.check(label_type_in='phone_wb', label_type_out='word',
                   data_type='train')
        self.check(label_type_in='phone_wb', label_type_out='word',
                   data_type='dev')

        # label_type
        self.check(label_type_in='character_wb', label_type_out='word')

    @measure_time
    def check(self, label_type_in, label_type_out, data_type='dev',
              data_size='aps_other', vocab=False):

        print('========================================')
        print('  label_type_in: %s' % label_type_in)
        print('  label_type_out: %s' % label_type_out)
        print('  data_type: %s' % data_type)
        print('  data_size: %s' % data_size)
        print('  vocab: %s' % str(vocab))
        print('========================================')

        dataset = Dataset(
            corpus='csj',
            data_save_path='/n/sd8/inaguma/corpus/csj/kaldi',
            data_size=data_size, data_type=data_type,
            label_type_in=label_type_in, label_type_out=label_type_out,
            batch_size=64, max_epoch=1,
            shuffle=False, sort_utt=True,
            reverse=False, sort_stop_epoch=None,
            num_enque=None, vocab=vocab)

        print('=> Loading mini-batch...')
        if 'phone' in label_type_in:
            map_fn_in = dataset.idx2phone
        else:
            map_fn_in = dataset.idx2char
        if label_type_out == 'word':
            map_fn_out = dataset.idx2word
        else:
            map_fn_out = dataset.idx2char

        for batch, is_new_epoch in dataset:
            str_ref_in = map_fn_in(batch['xs'][0])
            str_ref_out = map_fn_out(batch['ys'][0])

            print('----- %s (epoch: %.3f, batch: %d) -----' %
                  (batch['input_names'][0], dataset.epoch_detail, len(batch['xs'])))
            print(str_ref_in)
            print(str_ref_out)
            print('x_lens: %d' % (len(batch['xs'][0])))
            print('y_lens: %d' % (len(batch['ys'][0])))

            if dataset.epoch_detail >= 1:
                break


if __name__ == '__main__':
    unittest.main()
