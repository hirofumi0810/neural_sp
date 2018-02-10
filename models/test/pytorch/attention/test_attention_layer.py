#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test the attention layer (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

import torch
from torch.autograd import Variable

sys.path.append('../../../../')
from models.pytorch.attention.attention_layer import AttentionMechanism
from utils.measure_time_func import measure_time


class TestAttentionLayer(unittest.TestCase):

    def test(self):
        print("Attention layer Working check.")

        self.check(attention_type='content')
        self.check(attention_type='location')
        self.check(attention_type='dot_product')
        # self.check(attention_type='rnn_attention')

    @measure_time
    def check(self, attention_type):

        print('==================================================')
        print('  attention_type: %s' % attention_type)
        print('==================================================')

        batch_size = 4
        max_time = 200
        decoder_num_units = 256
        encoder_num_units = decoder_num_units

        attend = AttentionMechanism(
            decoder_num_units=decoder_num_units,
            attention_type=attention_type,
            attention_dim=128,
            sharpening_factor=2,
            sigmoid_smoothing=False,
            out_channels=10,
            kernel_size=101)

        enc_out = Variable(torch.randn(
            (batch_size, max_time, encoder_num_units)))
        x_lens = Variable(torch.ones(batch_size)) * max_time
        dec_state_step = Variable(torch.randn(
            (batch_size, 1, decoder_num_units)))
        att_weights_step = Variable(torch.randn((batch_size, max_time)))

        context_vec, att_weights_step = attend(enc_out,
                                               x_lens,
                                               dec_state_step,
                                               att_weights_step)

        assert context_vec.size() == (batch_size, 1, encoder_num_units)
        assert att_weights_step.size() == (batch_size, max_time)


if __name__ == '__main__':
    unittest.main()
