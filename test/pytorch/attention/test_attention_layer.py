#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test the attention layer (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

import torch

sys.path.append('../../../../../')
from src.models.pytorch_v3.attention.attention_layer import AttentionMechanism, MultiheadAttentionMechanism
from src.utils.measure_time_func import measure_time


class TestAttentionLayer(unittest.TestCase):

    def test(self):
        print("Attention layer Working check.")

        self.check(att_type='content')
        self.check(att_type='location')
        self.check(att_type='dot_product')
        # self.check(att_type='rnn_attention')
        # self.check(att_type='coverage')

        # multi-head attention
        self.check(att_type='content', n_heads=4)
        self.check(att_type='location', n_heads=4)
        self.check(att_type='dot_product', n_heads=4)
        # self.check(att_type='rnn_attention', n_heads=4)
        # self.check(att_type='coverage', n_heads=4)

    @measure_time
    def check(self, att_type, n_heads=1):

        print('==================================================')
        print('  att_type: %s' % att_type)
        print('  n_heads: %d' % n_heads)
        print('==================================================')

        batch_size = 4
        max_time = 200
        dec_n_units = 256
        enc_n_units = dec_n_units

        if n_heads == 1:
            attend = AttentionMechanism(
                enc_n_units=dec_n_units,
                dec_n_units=dec_n_units,
                att_type=att_type,
                att_dim=128,
                sharpening_factor=2,
                sigmoid_smoothing=False,
                out_channels=10,
                kernel_size=101)
        else:
            attend = MultiheadAttentionMechanism(
                enc_n_units=dec_n_units,
                dec_n_units=dec_n_units,
                att_type=att_type,
                att_dim=128,
                sharpening_factor=2,
                sigmoid_smoothing=False,
                out_channels=10,
                kernel_size=101,
                n_heads=n_heads)

        # NOTE: not work for 0.4
        enc_out = torch.randn((batch_size, max_time, enc_n_units))

        x_lens = torch.ones(batch_size) * max_time
        dec_state_step = torch.randn((batch_size, 1, dec_n_units))
        aw_step = None

        context_vec, aw_step = attend(enc_out,
                                      x_lens,
                                      dec_state_step,
                                      aw_step)

        assert context_vec.size() == (batch_size, 1, enc_n_units)
        if n_heads > 1:
            assert aw_step.size() == (batch_size, max_time, n_heads)
        else:
            assert aw_step.size() == (batch_size, max_time)


if __name__ == '__main__':
    unittest.main()
