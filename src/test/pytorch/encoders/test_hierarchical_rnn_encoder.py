#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test hierarchical RNN encoders (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

sys.path.append('../../../../../')
from src.models.pytorch_v3.encoders.load_encoder import load
from src.models.test.data import generate_data
from src.utils.measure_time_func import measure_time
from src.models.pytorch_v3.utils import np2var, pad_list


class TestHierarchicalRNNEncoders(unittest.TestCase):

    def test(self):
        print("Hierarchical RNN Encoders Working check.")

        # Projection layer
        self.check(encoder_type='lstm', bidirectional=False,
                   projection=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   projection=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   batch_first=False, projection=True)

        # Residual connection
        self.check(encoder_type='lstm', bidirectional=False,
                   residual=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   residual=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   batch_first=False, residual=True)
        self.check(encoder_type='lstm', bidirectional=False,
                   dense_residual=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   dense_residual=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   batch_first=False, dense_residual=True)

        # Conv
        self.check(encoder_type='lstm', bidirectional=True,
                   conv=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   batch_first=False, conv=True)
        self.check(encoder_type='gru', bidirectional=True,
                   conv=True)
        self.check(encoder_type='gru', bidirectional=True,
                   batch_first=False, conv=True)
        self.check(encoder_type='rnn', bidirectional=True,
                   conv=True)
        self.check(encoder_type='rnn', bidirectional=True,
                   batch_first=False, conv=True)

        # LSTM, GRU, RNN
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
        self.check(encoder_type='rnn', bidirectional=True,
                   batch_first=False)

        # merge_bidirectional
        self.check(encoder_type='lstm', bidirectional=True,
                   merge_bidirectional=True)
        self.check(encoder_type='gru', bidirectional=True,
                   merge_bidirectional=True)
        self.check(encoder_type='rnn', bidirectional=True,
                   merge_bidirectional=True)

    @measure_time
    def check(self, encoder_type, bidirectional=False, batch_first=True,
              conv=False, merge_bidirectional=False,
              projection=False, residual=False, dense_residual=False):

        print('==================================================')
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  batch_first: %s' % str(batch_first))
        print('  conv: %s' % str(conv))
        print('  merge_bidirectional: %s' % str(merge_bidirectional))
        print('  projection: %s' % str(projection))
        print('  residual: %s' % str(residual))
        print('  dense_residual: %s' % str(dense_residual))
        print('==================================================')

        if conv:
            # pattern 1
            # conv_channels = [32, 32]
            # conv_kernel_sizes = [[41, 11], [21, 11]]
            # conv_strides = [[2, 2], [2, 1]]
            # poolings = [[], []]

            # pattern 2 (VGG like)
            conv_channels = [64, 64]
            conv_kernel_sizes = [[3, 3], [3, 3]]
            conv_strides = [[1, 1], [1, 1]]
            poolings = [[2, 2], [2, 2]]
        else:
            conv_channels = []
            conv_kernel_sizes = []
            conv_strides = []
            poolings = []

        # Load batch data
        batch_size = 4
        splice = 1
        num_stack = 1
        xs, _ = generate_data(batch_size=batch_size)

        # Wrap by Variable
        xs = [np2var(x).float() for x in xs]
        x_lens = [len(x) for x in xs]
        xs = pad_list(xs)

        # Load encoder
        encoder = load(encoder_type=encoder_type)

        # Initialize encoder
        encoder = encoder(
            input_size=xs[0].shape[-1] // splice // num_stack,  # 120
            rnn_type=encoder_type,
            bidirectional=bidirectional,
            num_units=256,
            num_proj=256 if projection else 0,
            num_layers=5,
            num_layers_sub=3,
            dropout_input=0.2,
            dropout_hidden=0.2,
            subsample_list=[],
            batch_first=batch_first,
            merge_bidirectional=merge_bidirectional,
            splice=splice,
            num_stack=num_stack,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            poolings=poolings,
            batch_norm=True,
            residual=residual,
            dense_residual=dense_residual)

        max_time = max([x.shape[0] for x in xs])
        if conv:
            max_time = encoder.conv.get_conv_out_size(max_time, 1)

        outputs, _, outputs_sub, _ = encoder(xs, x_lens)

        print('----- outputs -----')
        print(outputs_sub.size())
        print(outputs.size())
        num_directions = 2 if bidirectional and not merge_bidirectional else 1
        self.assertEqual((batch_size, max_time, encoder.num_units * num_directions),
                         outputs_sub.size())
        self.assertEqual((batch_size, max_time, encoder.num_units * num_directions),
                         outputs.size())


if __name__ == '__main__':
    unittest.main()
