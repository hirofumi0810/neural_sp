#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test CTC models in pytorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import unittest

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('../../../')
from models.pytorch.ctc.ctc import CTC
from models.test.data import generate_data, np2var_pytorch, idx2alpha
from models.test.util import measure_time
from utils.io.tensor import to_np

torch.manual_seed(1)


class TestCTC(unittest.TestCase):

    def test(self):
        print("CTC Working check.")

        # RNNs
        self.check(encoder_type='lstm', bidirectional=False)
        self.check(encoder_type='lstm', bidirectional=True)
        self.check(encoder_type='gru', bidirectional=False)
        self.check(encoder_type='gru', bidirectional=True)
        self.check(encoder_type='rnn', bidirectional=False)
        self.check(encoder_type='rnn', bidirectional=True)
        # self.check(encoder_type='cldnn', bidirectional=True)
        # self.check(encoder_type='cldnn', bidirectional=True)

        # CNNs
        # self.check(encoder_type='resnet')
        # self.check(encoder_type='vgg')

    @measure_time
    def check(self, encoder_type, bidirectional=False):

        print('==================================================')
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('==================================================')

        # Load batch data
        batch_size = 4
        inputs, labels, inputs_seq_len, labels_seq_len = generate_data(
            model='ctc',
            batch_size=batch_size)

        # Wrap by Variable
        inputs = np2var_pytorch(inputs)
        labels = np2var_pytorch(labels, dtype='int')
        # inputs_seq_len = np2var_pytorch(inputs_seq_len, dtype='long')
        # labels_seq_len = np2var_pytorch(labels_seq_len, dtype='long')
        inputs_seq_len = np2var_pytorch(inputs_seq_len, dtype='int')
        labels_seq_len = np2var_pytorch(labels_seq_len, dtype='int')

        # Load model
        model = CTC(
            input_size=inputs.size(-1),
            encoder_type=encoder_type,
            bidirectional=bidirectional,
            num_units=256,
            # num_proj=None,
            num_layers=2,
            dropout=0,
            num_classes=27,  # alphabets + space (excluding a blank class)
            splice=1,
            parameter_init=0.1,
            bottleneck_dim=None)

        # Define optimizer
        optimizer, scheduler = model.set_optimizer(
            'adam', learning_rate_init=1e-3, weight_decay=0,
            lr_schedule=False, factor=0.1, patience_epoch=5)

        # Initialize parameters
        model.init_weights()

        # Count total parameters
        print("Total %.3f M parameters" % (model.total_parameters / 1000000))

        # GPU setting
        use_cuda = torch.cuda.is_available()
        deterministic = False
        if use_cuda and deterministic:
            print('GPU deterministic mode (no cudnn)')
            torch.backends.cudnn.enabled = False
        elif use_cuda:
            print('GPU mode (faster than the deterministic mode)')
        else:
            print('CPU mode')
        if use_cuda:
            model = model.cuda()
            inputs = inputs.cuda()
            # labels = labels.cuda()
            # inputs_seq_len = inputs_seq_len.cuda()
            # labels_seq_len = labels_seq_len.cuda()
            # print(inputs_seq_len)
            # print(labels_seq_len)

        # Train model
        max_step = 1000
        start_time_step = time.time()
        ler_train_pre = 1
        save_flag = False
        for step in range(max_step):

            # Clear gradients before
            optimizer.zero_grad()

            # Make prediction
            logits = model(inputs)

            # Compute loss
            loss = model.loss(logits, labels, inputs_seq_len, labels_seq_len)

            # Compute gradient
            optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm
            nn.utils.clip_grad_norm(model.parameters(), 10)

            ######
            loss_sum = loss.data.sum()
            inf = float("inf")
            if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss.data[0]

            if (step + 1) % 10 == 0:
                # Change to evaluation mode

                # Decode
                # outputs_infer, _ = model.decode_infer(inputs, labels,
                #                                       beam_width=1)

                # Compute accuracy

                duration_step = time.time() - start_time_step
                print('Step %d: loss = %.3f / ler = %.3f (%.3f sec) / lr = %.5f' %
                      (step + 1, to_np(loss), 1, duration_step, 1e-3))
                start_time_step = time.time()

                # Visualize
                # print('Ref: %s' % idx2alpha(to_np(labels)[0][1:-1]))
                # print('Hyp: %s' % idx2alpha(outputs_infer[0][0:-1]))

                # if to_np(loss) <1.:
                #     print('Modle is Converged.')
                #     break
                # ler_train_pre = ler_train


if __name__ == "__main__":
    unittest.main()
