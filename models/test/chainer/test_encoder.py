#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test encoders in chainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

sys.path.append('../../')
from models.chainer.encoders.load_encoder import load
from models.test.data import generate_data, np2var_chainer, np2varlist_chainer
from models.test.util import measure_time


class TestEncoders(unittest.TestCase):

    def test(self):
        print("Encoders Working check.")

        # RNNs
        self.check(encoder_type='lstm')
        self.check(encoder_type='lstm', bidirectional=True)
        self.check(encoder_type='gru')
        self.check(encoder_type='gru', bidirectional=True)
        self.check(encoder_type='rnn_tanh')
        self.check(encoder_type='rnn_tanh', bidirectional=True)
        self.check(encoder_type='rnn_relu')
        self.check(encoder_type='rnn_relu', bidirectional=True)

        # self.check(encoder_type='conv_lstm')
        # self.check('vgg_lstm')

        # CNNs
        # self.check(encoder_type='resnet')
        # self.check(encoder_type='vgg')

    @measure_time
    def check(self, encoder_type, bidirectional=False):

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
        inputs = np2varlist_chainer(inputs)
        labels = np2var_chainer(labels)
        inputs_seq_len = np2var_chainer(inputs_seq_len)

        # Load encoder
        encoder = load(encoder_type=encoder_type)

        # Initialize encoder
        if encoder_type in ['lstm', 'gru', 'rnn_tanh', 'rnn_relu']:
            encoder = encoder(input_size=inputs[0].shape[-1],
                              num_units=256,
                              num_layers=5,
                              num_classes=0,
                              rnn_type=encoder_type,
                              bidirectional=bidirectional,
                              parameter_init=0.1)
        else:
            raise NotImplementedError

        outputs, final_state = encoder(inputs)

        assert isinstance(outputs, list)

        print('----- final state -----')
        print(final_state.shape)
        self.assertEqual((encoder.num_layers * encoder.num_directions,
                          batch_size, encoder.num_units), final_state.shape)

        print('----- outputs -----')
        # Expected list of Variable of size `(T, num_units * num_directions]`
        print((len(outputs), outputs[0].shape[0], outputs[0].shape[1]))
        self.assertEqual(
            (len(inputs), inputs[0].shape[0],
             encoder.num_units * encoder.num_directions),
            (len(outputs), outputs[0].shape[0], outputs[0].shape[1]))


if __name__ == '__main__':
    unittest.main()
