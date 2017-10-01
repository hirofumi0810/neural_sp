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

    def test_encoders(self):
        print("Encoders Working check.")

        # RNNs
        self.check_encode(encoder_type='lstm')
        self.check_encode(encoder_type='lstm', bidirectional=True)
        self.check_encode(encoder_type='gru')
        self.check_encode(encoder_type='gru', bidirectional=True)
        self.check_encode(encoder_type='rnn')
        self.check_encode(encoder_type='rnn', bidirectional=True)

        # self.check_encode(encoder_type='conv_lstm')
        # self.check_encode('vgg_lstm')

        # CNNs
        # self.check_encode(encoder_type='resnet')
        # self.check_encode(encoder_type='vgg')

    @measure_time
    def check_encode(self, encoder_type, bidirectional=False):

        print('==================================================')
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
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

        # Load encoder
        encoder = load(encoder_type=encoder_type)

        # Initialize encoder
        if encoder_type in ['lstm', 'gru', 'rnn']:
            encoder = encoder(input_size=inputs.size(-1),
                              num_units=256,
                              num_layers=2,
                              num_classes=0,
                              rnn_type=encoder_type,
                              bidirectional=bidirectional,
                              dropout=0.8,
                              parameter_init=0.1)
        else:
            raise NotImplementedError

        outputs, final_state = encoder(inputs)

        print('----- final state -----')
        print(final_state.size())
        self.assertEqual((encoder.num_layers * encoder.num_directions,
                          batch_size, encoder.num_units), final_state.size())

        print('----- outputs -----')
        print(outputs.size())
        num_directions = 2 if bidirectional else 1
        self.assertEqual((inputs.size(1), inputs.size(0), 256 * num_directions), outputs.size())


if __name__ == '__main__':
    unittest.main()
