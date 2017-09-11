#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test CTC models in pytorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

import torch

sys.path.append('../../')
from models.pytorch.ctc.ctc import CTC
from models.test.data import generate_data, np2var
from models.test.util import measure_time

torch.manual_seed(1)


class TestCTC(unittest.TestCase):

    def test_ctc(self):
        print("CTC Working check.")

        # RNNs
        self.check_training(encoder_type='lstm', bidirectional=False)
        self.check_training(encoder_type='lstm', bidirectional=True)
        self.check_training(encoder_type='gru', bidirectional=False)
        self.check_training(encoder_type='gru', bidirectional=True)
        self.check_training(encoder_type='rnn', bidirectional=False)
        self.check_training(encoder_type='rnn', bidirectional=True)

        # self.check_encode(encoder_type='conv_lstm')
        # self.check_encode('vgg_lstm')

        # CNNs
        # self.check_encode(encoder_type='resnet')
        # self.check_encode(encoder_type='vgg')

    @measure_time
    def check_training(self, encoder_type, bidirectional=False):

        print('==================================================')
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('==================================================')

        # Load batch data
        batch_size = 2
        inputs, labels, inputs_seq_len = generate_data(
            model='ctc',
            batch_size=batch_size)

        # Wrap by Variable
        inputs = np2var(inputs)
        labels = np2var(labels)
        inputs_seq_len = np2var(inputs_seq_len)

        # load model
        model = CTC(input_size=inputs.size(-1),
                    num_units=256,
                    num_layers=2,
                    num_classes=27,  # alphabets + space (excluding a blank)
                    encoder_type=encoder_type,
                    bidirectional=bidirectional,
                    use_peephole=True,
                    splice=1,
                    parameter_init=0.1,
                    clip_grad=None,
                    clip_activation=None,
                    num_proj=None,
                    weight_decay=0.0,
                    bottleneck_dim=None)

        # Initialize parameters
        model.init_weights()

        # Count total parameters
        print("Total %s M parameters" %
              ("{:,}".format(model.total_parameters / 1000000)))

        # define loss function
        # loss_fn = CTC_loss()

        # define optimizer
        optimizer, scheduler = model.set_optimizer(
            'adam', learning_rate_init=1e-3, weight_decay=0,
            lr_schedule=True, factor=0.1, patience_epoch=5)

        return 0

        for step in range(500):
            # Clear gradients before
            model.zoro_grad()

            # Clear hidden state
            model_hidden = model.init_hidden()
            # NOTE: the first step is redundant because it is included in
            # __init__ function
            # TODO: change to reset()

            # Make prediction
            logits = model(inputs)

            # Compute the loss, gradients, and update parameters
            model.compute_loss(loss_fn, logits, labels)
            model.update()

            if (step + 1) % 10 == 0:
                # Change to evaluation mode

                # Compute accuracy

                # Visualize

                # Update learning rate
                scheduler.step(val_acc)
                # TODO: confirm learning rate


if __name__ == "__main__":
    unittest.main()
