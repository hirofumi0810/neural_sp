#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test RNN encoders in pytorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest
import numpy as np

sys.path.append('../../../')
from models.pytorch.encoders.load_encoder import load
from models.test.data import generate_data
from utils.io.variable import np2var, var2np
from utils.measure_time_func import measure_time


class TestRNNEncoders(unittest.TestCase):

    def test(self):
        print("RNN Encoders Working check.")

        # LSTM
        self.check(encoder_type='lstm', bidirectional=False)
        self.check(encoder_type='lstm', bidirectional=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   batch_first=False)

        # GRU
        self.check(encoder_type='gru', bidirectional=False)
        self.check(encoder_type='gru', bidirectional=True)
        self.check(encoder_type='gru', bidirectional=True,
                   batch_first=False)

        # RNN
        self.check(encoder_type='rnn', bidirectional=False)
        self.check(encoder_type='rnn', bidirectional=True)
        self.check(encoder_type='rnn', bidirectional=True,
                   batch_first=False)

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
            model='ctc',
            batch_size=batch_size,
            splice=1)

        # Wrap by Variable
        inputs = np2var(inputs)
        inputs_seq_len = np2var(inputs_seq_len)

        max_time = inputs.size(1)

        # Load encoder
        encoder = load(encoder_type=encoder_type)

        # Initialize encoder
        if encoder_type in ['lstm', 'gru', 'rnn']:
            encoder = encoder(
                input_size=inputs.size(-1),
                rnn_type=encoder_type,
                bidirectional=bidirectional,
                num_units=256,
                num_proj=0,
                num_layers=5,
                dropout=0.2,
                parameter_init=0.1,
                batch_first=batch_first)
        else:
            raise NotImplementedError

        outputs, final_state, perm_indices = encoder(
            inputs, inputs_seq_len, mask_sequence=mask_sequence)

        # Check final state (forward)
        print('----- Check hidden states (forward) -----')
        if batch_first:
            outputs_fw_final = outputs.transpose(
                0, 1)[-1, 0, :encoder.num_units]
        else:
            outputs_fw_final = outputs[-1, 0, :encoder.num_units]
        assert np.all(var2np(outputs_fw_final) == var2np(final_state[0, 0, :]))

        # Check final state (backward)
        # if bidirectional:
        #     print('----- Check hidden states (backward) -----')
        #     if batch_first:
        #         outputs_bw = outputs.transpose(0, 1)[0, -1, encoder.num_units:]
        #     else:
        #         outputs_bw = outputs[-1, 0, encoder.num_units:]
        #     top_final_state_bw = final_state[-1, 0, :]
        #     assert np.all(outputs_bw.data == top_final_state_bw.data)

        print('----- final state -----')
        print(final_state.size())
        self.assertEqual((1, batch_size, encoder.num_units),
                         final_state.size())

        print('----- outputs -----')
        print(outputs.size())
        num_directions = 2 if bidirectional else 1
        if batch_first:
            self.assertEqual(
                (batch_size, max_time, encoder.num_units * num_directions),
                outputs.size())

        else:
            self.assertEqual(
                (max_time, batch_size, encoder.num_units * num_directions),
                outputs.size())


if __name__ == '__main__':
    unittest.main()
