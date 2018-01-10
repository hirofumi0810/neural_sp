#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test CTC models in chainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import unittest

import chainer

sys.path.append('../../')
from models.chainer.ctc.ctc import CTC
from models.chainer.ctc.ctc_decoder import GreedyDecoder
from models.test.data import generate_data, np2var_chainer, np2varlist_chainer, idx2alpha
from utils.measure_time_func import measure_time


class TestCTC(unittest.TestCase):

    def test(self):
        print("CTC Working check.")

        # RNNs
        self.check(encoder_type='lstm', bidirectional=True)
        self.check(encoder_type='lstm', bidirectional=False)
        self.check(encoder_type='gru', bidirectional=True)
        self.check(encoder_type='gru', bidirectional=False)
        self.check(encoder_type='rnn_tanh', bidirectional=True)
        self.check(encoder_type='rnn_tanh', bidirectional=False)
        self.check(encoder_type='rnn_relu', bidirectional=True)
        self.check(encoder_type='rnn_relu', bidirectional=False)

        # self.check(encoder_type='conv_lstm')
        # self.check('vgg_lstm')

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
        batch_size = 1
        inputs, labels, inputs_seq_len, labels_seq_len = generate_data(
            model='ctc', batch_size=batch_size)

        # Wrap by Variable
        inputs = np2varlist_chainer(inputs)
        labels = np2var_chainer(labels)
        inputs_seq_len = np2var_chainer(inputs_seq_len)

        # Load model
        model = CTC(encoder_type=encoder_type,
                    input_size=inputs[0].shape[-1],
                    num_units=256,
                    num_layers=2,
                    num_classes=27,  # alphabets + space (excluding a blank)
                    bidirectional=bidirectional,
                    use_peephole=True,
                    splice=1,
                    parameter_init=0.1,
                    clip_grad=5.,
                    clip_activation=None,
                    num_proj=None,
                    weight_decay=1e-8,
                    bottleneck_dim=None)

        # Load CTC decoder
        decoder = GreedyDecoder(blank_index=27)
        # decoder = BeamSearchDecoder(blank_index=27)

        # Initialize parameters
        # model.init_weights()

        # Count total parameters
        # print("Total %s M parameters" %
        #       ("{:,}".format(model.total_parameters / 1000000)))

        # Define optimizer
        optimizer = model.set_optimizer('sgd', learning_rate_init=1e-3)
        optimizer.setup(model)

        # Add hook
        if model.clip_grad is not None:
            optimizer.add_hook(
                chainer.optimizer.GradientClipping(model.clip_grad))
        if model.weight_decay != 0:
            optimizer.add_hook(
                chainer.optimizer.WeightDecay(model.weight_decay))

        # Train model
        max_step = 1000
        start_time_global = time.time()
        start_time_step = time.time()
        ler_train_pre = 1
        not_improved_count = 0
        for step in range(max_step):

            # Make prediction
            logits = model(inputs)

            # Clear the parameter gradients
            model.cleargrads()

            # Compute loss
            ctc_loss = model.compute_loss(logits, labels,
                                          blank_index=27,
                                          inputs_seq_len=inputs_seq_len,
                                          labels_seq_len=labels_seq_len)

            # Backprop gradients
            ctc_loss.backward()

            # Update parameters
            optimizer.update()

            if (step + 1) % 10 == 0:
                # Change to evaluation mode
                # ex.) dropout

                # Compute accuracy
                indices_pred = decoder(logits[:1])
                # TODO: char に戻す前に編集距離計算したい

                labels_pred = idx2alpha(indices_pred[0])
                # print(labels_pred)
                # ler_train =

                duration_step = time.time() - start_time_step
                print('Step %d: loss = %.3f / ler = %.4f (%.3f sec) / lr = %.5f' %
                      (step + 1, ctc_loss.data, 1, duration_step, 1e-3))
                # print('Step %d: loss = %.3f / ler = %.4f (%.3f sec) / lr = %.5f' %
                #       (step + 1, loss_train, ler_train, duration_step, learning_rate))
                start_time_step = time.time()

                # Visualize
                # print('True: %s' % num2alpha(labels_true[0]))
                # print('Pred: %s' % num2alpha(labels_pred[0]))

                # if ler_train >= ler_train_pre:
                #     not_improved_count += 1
                # else:
                #     not_improved_count = 0
                # if ler_train < 0.05:
                #     print('Modle is Converged.')
                #     break
                # ler_train_pre = ler_train

                # Update learning rate
                # scheduler.step(ler_train)
                # TODO: confirm learning rate

        duration_global = time.time() - start_time_global
        print('Total time: %.3f sec' % (duration_global))


if __name__ == "__main__":
    unittest.main()
