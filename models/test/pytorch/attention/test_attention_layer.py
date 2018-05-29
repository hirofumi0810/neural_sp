#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test the attention layer (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

import torch

sys.path.append('../../../../')
from models.pytorch_v3.attention.attention_layer import AttentionMechanism
from utils.measure_time_func import measure_time


class TestAttentionLayer(unittest.TestCase):

    def test(self):
        print("Attention layer Working check.")

        self.check(attention_type='content')
        self.check(attention_type='location')
        self.check(attention_type='dot_product')
        # self.check(attention_type='rnn_attention')
        # self.check(attention_type='coverage')

        # multi-head attention
        self.check(attention_type='content', num_heads=4)
        self.check(attention_type='location', num_heads=4)
        self.check(attention_type='dot_product', num_heads=4)
        # self.check(attention_type='rnn_attention', num_heads=4)
        # self.check(attention_type='coverage', num_heads=4)

    @measure_time
    def check(self, attention_type, num_heads=1):

        print('==================================================')
        print('  attention_type: %s' % attention_type)
        print('  num_heads: %d' % num_heads)
        print('==================================================')

        batch_size = 4
        max_time = 200
        decoder_num_units = 256
        encoder_num_units = decoder_num_units

        attend = AttentionMechanism(
            encoder_num_units=decoder_num_units,
            decoder_num_units=decoder_num_units,
            attention_type=attention_type,
            attention_dim=128,
            sharpening_factor=2,
            sigmoid_smoothing=False,
            out_channels=10,
            kernel_size=101,
            num_heads=num_heads)

        # NOTE: not work for 0.4
        enc_out = torch.randn((batch_size, max_time, encoder_num_units))
        if attention_type == 'dot_product':
            enc_out_a = torch.randn(
                (batch_size, max_time, decoder_num_units, num_heads))
        else:
            enc_out_a = torch.randn((batch_size, max_time, 128, num_heads))
        x_lens = torch.ones(batch_size) * max_time
        dec_state_step = torch.randn((batch_size, 1, decoder_num_units))
        aw_step = torch.randn((batch_size, max_time, num_heads))

        context_vec, aw_step = attend(enc_out,
                                      enc_out_a,
                                      x_lens,
                                      dec_state_step,
                                      aw_step)

        assert context_vec.size() == (batch_size, 1, encoder_num_units)
        assert aw_step.size() == (batch_size, max_time, num_heads)


if __name__ == '__main__':
    unittest.main()
