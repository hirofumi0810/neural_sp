#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test encoders in pytorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

sys.path.append('../../')
from models.pytorch.encoders.load_encoder import load
from models.test.data import generate_data, np2var_pytorch
from models.test.util import measure_time


class TestEncoders(unittest.TestCase):

    def test(self):
        print("Encoders Working check.")

        # RNNs
        self.check(encoder_type='lstm')
        self.check(encoder_type='lstm', bidirectional=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   num_classes=27)
        self.check(encoder_type='lstm', bidirectional=True,
                   batch_first=True)
        self.check(encoder_type='gru')
        self.check(encoder_type='gru', bidirectional=True)
        self.check(encoder_type='rnn')
        self.check(encoder_type='rnn', bidirectional=True)

        # self.check_encode(encoder_type='conv_lstm')
        # self.check_encode('vgg_lstm')

        # CNNs
        # self.check_encode(encoder_type='resnet')
        # self.check_encode(encoder_type='vgg')

    @measure_time
    def check(self, encoder_type, bidirectional=False, num_classes=0,
              batch_first=False):

        print('==================================================')
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  num_classes: %s' % str(num_classes))
        print('  batch_first: %s' % str(batch_first))
        print('==================================================')

        # Load batch data
        batch_size = 2
        inputs, labels, inputs_seq_len, labels_seq_len = generate_data(
            model='ctc',
            batch_size=batch_size,
            splice=1)

        # Wrap by Variable
        inputs = np2var_pytorch(inputs)
        labels = np2var_pytorch(labels)
        inputs_seq_len = np2var_pytorch(inputs_seq_len)

        max_time = inputs.size(1)

        # Load encoder
        encoder = load(encoder_type=encoder_type)

        # Initialize encoder
        if encoder_type in ['lstm', 'gru', 'rnn']:
            encoder = encoder(input_size=inputs.size(-1),
                              num_units=256,
                              num_layers=2,
                              num_classes=num_classes,
                              rnn_type=encoder_type,
                              bidirectional=bidirectional,
                              dropout=0.8,
                              parameter_init=0.1,
                              batch_first=batch_first)
        else:
            raise NotImplementedError

        outputs, final_state = encoder(inputs)

        print('----- final state -----')
        print(final_state.size())
        self.assertEqual((encoder.num_layers * encoder.num_directions,
                          batch_size,
                          encoder.num_units),
                         final_state.size())

        print('----- outputs -----')
        print(outputs.size())
        num_directions = 2 if bidirectional else 1
        if batch_first:
            if num_classes == 0:
                self.assertEqual((batch_size, max_time, 256 * num_directions),
                                 outputs.size())
            else:
                self.assertEqual((batch_size, max_time, num_classes),
                                 outputs.size())
        else:
            if num_classes == 0:
                self.assertEqual((max_time, batch_size, 256 * num_directions),
                                 outputs.size())
            else:
                self.assertEqual((max_time, batch_size, num_classes),
                                 outputs.size())


if __name__ == '__main__':
    unittest.main()
