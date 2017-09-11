#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test encoders in pytorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

sys.path.append('../../')
from models.pytorch.layers.encoders.load_encoder import load
from models.test.data import generate_data
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
        inputs, labels, inputs_seq_len = generate_data(
            model='ctc',
            batch_size=batch_size)

        # Load encoder
        encoder = load(encoder_type=encoder_type)

        # Initialize encoder
        if encoder_type in ['lstm', 'gru', 'rnn']:
            encoder = encoder(input_size=inputs.size(-1),
                              num_units=256,
                              num_layers=2,
                              num_classes=27,
                              rnn_type=encoder_type,
                              bidirectional=bidirectional,
                              parameter_init=0.1)
        else:
            raise NotImplementedError

        outputs, final_state = encoder(inputs)

        if encoder_type == 'lstm':
            print(final_state[0].size())
            print(final_state[1].size())
            self.assertEqual((encoder.num_layers * encoder.num_directions,
                              batch_size, encoder.num_units), final_state[0].size())
            self.assertEqual((encoder.num_layers * encoder.num_directions,
                              batch_size, encoder.num_units), final_state[1].size())
        else:
            print(final_state.size())
            self.assertEqual((encoder.num_layers * encoder.num_directions,
                              batch_size, encoder.num_units), final_state.size())

        print(outputs.size())
        self.assertEqual((inputs.size(1), inputs.size(0), 27), outputs.size())


if __name__ == '__main__':
    unittest.main()
