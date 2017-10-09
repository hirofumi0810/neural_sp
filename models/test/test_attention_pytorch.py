#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test Attention-besed models in pytorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import unittest

import torch
import torch.nn as nn

sys.path.append('../../')
from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq
from models.test.data import generate_data, np2var_pytorch
from models.test.util import measure_time

torch.manual_seed(1)


class TestAttention(unittest.TestCase):

    def test(self):
        print("Attention Working check.")

        # self.check(encoder_type='lstm', bidirectional=False)
        # self.check(encoder_type='lstm', bidirectional=True)
        # self.check(encoder_type='gru', bidirectional=False)
        self.check(encoder_type='gru', bidirectional=True)

    @measure_time
    def check(self, encoder_type, bidirectional=False):

        print('==================================================')
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('==================================================')

        # Load batch data
        batch_size = 4
        inputs, labels, inputs_seq_len, labels_seq_len = generate_data(
            model='attention',
            batch_size=batch_size)

        # Wrap by Variable
        inputs = np2var_pytorch(inputs)
        labels = np2var_pytorch(labels, dtype='long')
        inputs_seq_len = np2var_pytorch(inputs_seq_len, dtype='long')
        labels_seq_len = np2var_pytorch(labels_seq_len, dtype='long')

        # Load model
        model = AttentionSeq2seq(
            input_size=inputs.size(-1),
            encoder_type=encoder_type,
            encoder_bidirectional=bidirectional,
            encoder_num_units=128,
            #  encoder_num_proj,
            encoder_num_layers=2,
            encoder_dropout=0,
            attention_type='dot_product',
            attention_dim=128,
            decoder_type='gru',
            decoder_num_units=256,
            #   decdoder_num_layers,
            embedding_dim=64,
            num_classes=27,
            decoder_dropout=0,
            max_decode_length=100,
            splice=1,
            parameter_init=0.1,
            att_softmax_temperature=1.,
            logits_softmax_temperature=1,
            clip_grad=None)

        # Initialize parameters
        model.init_weights()

        # Count total parameters
        print("Total %s M parameters" %
              ("{:,}".format(model.total_parameters / 1000000)))

        # Define optimizer
        optimizer, _ = model.set_optimizer(
            'adam', learning_rate_init=1e-3, weight_decay=0,
            lr_schedule=True, factor=0.1, patience_epoch=5)

        # if args.cuda:
        #     inputs = inputs.cuda()

        # Train model
        max_step = 1000
        start_time_global = time.time()
        start_time_step = time.time()
        # ler_train_pre = 1
        # not_improved_count = 0
        for step in range(max_step):

            # Clear gradients before
            optimizer.zero_grad()

            # Make prediction
            outputs = model(inputs, labels)

            # Compute loss
            loss = model.compute_loss(outputs, labels)

            # Compute gradient
            optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm
            nn.utils.clip_grad_norm(model.parameters(), 10)

            # Update parameters
            optimizer.step()

            if (step + 1) % 10 == 0:
                print(loss.data)

                # Change to evaluation mode

                # Compute accuracy
                # ler_train =

                # Visualize

                # Update learning rate
                # scheduler.step(ler_train)
                # TODO: confirm learning rate


if __name__ == "__main__":
    unittest.main()
