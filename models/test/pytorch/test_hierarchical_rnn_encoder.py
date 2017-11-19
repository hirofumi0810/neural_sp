#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test hierarchical RNN encoders in pytorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

sys.path.append('../../../')
from models.pytorch.encoders.load_encoder import load
from models.test.data import generate_data
from utils.io.variable import np2var
from utils.measure_time_func import measure_time


class TestHierarchicalRNNEncoders(unittest.TestCase):

    def test(self):
        print("Hierarchical RNN Encoders Working check.")

        self.check(encoder_type='lstm')
        self.check(encoder_type='lstm', bidirectional=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   batch_first=False)

        self.check(encoder_type='gru')
        self.check(encoder_type='gru', bidirectional=True)
        self.check(encoder_type='gru', bidirectional=True,
                   batch_first=False)

        self.check(encoder_type='rnn')
        self.check(encoder_type='rnn', bidirectional=True)

    @measure_time
    def check(self, encoder_type, bidirectional=False, batch_first=True,
              mask_sequence=True):

        print('==================================================')
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  batch_first: %s' % str(batch_first))
        print('  mask_sequence: %s' % str(mask_sequence))
        print('==================================================')

        # Load batch data
        batch_size = 4
        inputs, _, inputs_seq_len, _ = generate_data(
            model_type='ctc',
            batch_size=batch_size,
            splice=1)

        # Wrap by Variable
        inputs = np2var(inputs)
        inputs_seq_len = np2var(inputs_seq_len)

        max_time = inputs.size(1)

        # Load encoder
        encoder = load(encoder_type=encoder_type + '_hierarchical')

        # Initialize encoder
        if encoder_type in ['lstm', 'gru', 'rnn']:
            encoder = encoder(
                input_size=inputs.size(-1),
                rnn_type=encoder_type,
                bidirectional=bidirectional,
                num_units=256,
                num_proj=0,
                num_layers=5,
                num_layers_sub=3,
                dropout=0.2,
                parameter_init=0.1,
                batch_first=batch_first)
        else:
            raise NotImplementedError

        outputs, final_state, outputs_sub, final_state_sub, perm_indices = encoder(
            inputs, inputs_seq_len, mask_sequence=mask_sequence)

        print('----- final state -----')
        print(final_state_sub.size())
        print(final_state.size())
        self.assertEqual((1, batch_size, encoder.num_units),
                         final_state_sub.size())
        self.assertEqual((1, batch_size, encoder.num_units),
                         final_state.size())

        print('----- outputs -----')
        print(outputs_sub.size())
        print(outputs.size())
        num_directions = 2 if bidirectional else 1
        if batch_first:
            self.assertEqual((batch_size, max_time, encoder.num_units * num_directions),
                             outputs_sub.size())
            self.assertEqual((batch_size, max_time, encoder.num_units * num_directions),
                             outputs.size())
        else:
            self.assertEqual((max_time, batch_size, encoder.num_units * num_directions),
                             outputs_sub.size())
            self.assertEqual((max_time, batch_size, encoder.num_units * num_directions),
                             outputs.size())


if __name__ == '__main__':
    unittest.main()
