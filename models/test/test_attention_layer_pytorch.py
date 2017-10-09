#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test the attention layer in pytorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

import torch
from torch.autograd import Variable

sys.path.append('../../')
from models.pytorch.attention.attention_layer import AttentionMechanism
from models.test.util import measure_time

torch.manual_seed(1)


class TestAttentionLayer(unittest.TestCase):

    def test(self):
        print("Attention layer Working check.")

        # Luong's implementation
        self.check(attention_type='dot_product')
        self.check(attention_type='general')
        # self.check(attention_type='concat')

        # Baudanau's implementation
        self.check(attention_type='content')
        # self.check(attention_type='location')
        # self.check(attention_type='hybrid')

    @measure_time
    def check(self, attention_type):

        print('==================================================')
        print('  attention_type: %s' % attention_type)
        print('==================================================')

        attend = AttentionMechanism(
            encoder_num_units=256,
            decoder_num_units=256,
            attention_type=attention_type,
            attention_dim=128,
            att_softmax_temperature=1)

        batch_size = 4
        max_time = 20
        encoder_num_units = 256
        decoder_num_units = 256

        encoder_outputs = Variable(torch.randn(
            (batch_size, max_time, encoder_num_units)))
        decoder_state_step = Variable(torch.randn(
            (batch_size, 1, decoder_num_units)))
        attention_weights_step = Variable(torch.randn(
            (batch_size, max_time)))

        context_vector, attention_weights_step = attend(
            encoder_outputs,
            decoder_state_step,
            attention_weights_step)

        assert context_vector.size() == (batch_size, 1, encoder_num_units)
        assert attention_weights_step.size() == (batch_size, max_time)


if __name__ == '__main__':
    unittest.main()
