#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test praimidal RNN encoders (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest
import math

sys.path.append('../../../../')
from models.chainer.encoders.load_encoder import load
from models.test.data import generate_data
from utils.io.variable import np2var
from utils.measure_time_func import measure_time


class TestPyramidRNNEncoders(unittest.TestCase):

    def test(self):
        print("Pyramidal RNN Encoders Working check.")

        # Projection layer
        self.check(encoder_type='lstm', bidirectional=False,
                   projection=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   projection=True)

        # Residual connection
        self.check(encoder_type='lstm', bidirectional=False,
                   residual=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   residual=True)
        self.check(encoder_type='lstm', bidirectional=False,
                   dense_residual=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   dense_residual=True)

        # Conv
        self.check(encoder_type='lstm', bidirectional=True,
                   conv=True)
        self.check(encoder_type='gru', bidirectional=True,
                   conv=True)
        self.check(encoder_type='rnn', bidirectional=True,
                   conv=True)

        # drop
        self.check(encoder_type='lstm', bidirectional=False,
                   subsample_type='drop')
        self.check(encoder_type='lstm', bidirectional=True,
                   subsample_type='drop')
        self.check(encoder_type='gru', bidirectional=False,
                   subsample_type='drop')
        self.check(encoder_type='gru', bidirectional=True,
                   subsample_type='drop')
        self.check(encoder_type='rnn', bidirectional=False,
                   subsample_type='drop')
        self.check(encoder_type='rnn', bidirectional=True,
                   subsample_type='drop')

        # concat
        self.check(encoder_type='lstm', bidirectional=False,
                   subsample_type='concat')
        self.check(encoder_type='lstm', bidirectional=True,
                   subsample_type='concat')
        self.check(encoder_type='gru', bidirectional=False,
                   subsample_type='concat')
        self.check(encoder_type='gru', bidirectional=True,
                   subsample_type='concat')
        self.check(encoder_type='rnn', bidirectional=False,
                   subsample_type='concat')
        self.check(encoder_type='rnn', bidirectional=True,
                   subsample_type='concat')

        # merge_bidirectional
        self.check(encoder_type='lstm', bidirectional=True,
                   merge_bidirectional=True)
        self.check(encoder_type='gru', bidirectional=True,
                   merge_bidirectional=True)
        self.check(encoder_type='rnn', bidirectional=True,
                   merge_bidirectional=True)

    @measure_time
    def check(self, encoder_type, bidirectional=False,
              subsample_type='concat',
              conv=False, merge_bidirectional=False,
              projection=False, residual=False, dense_residual=False):

        print('==================================================')
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  subsample_type: %s' % subsample_type)
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
        xs, _, x_lens, _ = generate_data(batch_size=batch_size,
                                         num_stack=num_stack,
                                         splice=splice,
                                         backend='chainer')

        # Wrap by Variable
        xs = np2var(xs, backend='chainer')

        # Load encoder
        encoder = load(encoder_type=encoder_type)

        # Initialize encoder
        if encoder_type in ['lstm', 'gru', 'rnn']:
            encoder = encoder(
                input_size=xs[0].shape[-1] // splice // num_stack,  # 120
                rnn_type=encoder_type,
                bidirectional=bidirectional,
                num_units=256,
                num_proj=256 if projection else 0,
                num_layers=6,
                num_layers_sub=4,
                dropout_input=0.2,
                dropout_hidden=0.2,
                subsample_list=[False, True, True, False, False, False],
                subsample_type=subsample_type,
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
        else:
            raise NotImplementedError

        max_time = xs[0].shape[0]
        if conv:
            max_time = encoder.conv.get_conv_out_size(max_time, 1)
        max_time_sub = max_time / \
            (2 ** sum(encoder.subsample_list[:encoder.num_layers_sub]))
        max_time /= (2 ** sum(encoder.subsample_list))
        if subsample_type == 'drop':
            max_time_sub = math.ceil(max_time_sub)
            max_time = math.ceil(max_time)
        elif subsample_type == 'concat':
            max_time_sub = int(max_time_sub)
            max_time = int(max_time)

        outputs, _, outputs_sub, _ = encoder(xs, x_lens)

        print('----- outputs -----')
        print((len(xs), xs[0].shape[0], xs[0].shape[1]))
        print((len(outputs), outputs[0].shape[0], outputs[0].shape[1]))
        print(
            (len(outputs_sub), outputs_sub[0].shape[0], outputs_sub[0].shape[1]))
        num_directions = 2 if bidirectional and not merge_bidirectional else 1
        self.assertEqual((batch_size, max_time_sub, encoder.num_units * num_directions),
                         (len(outputs_sub), outputs_sub[0].shape[0], outputs_sub[0].shape[1]))
        self.assertEqual((batch_size, max_time, encoder.num_units * num_directions),
                         (len(outputs), outputs[0].shape[0], outputs[0].shape[1]))


if __name__ == '__main__':
    unittest.main()
