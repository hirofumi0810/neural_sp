#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test RNN language models (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import unittest
import math

import torch
import torch.nn as nn
torch.manual_seed(1623)
torch.cuda.manual_seed_all(1623)

sys.path.append('../../../../')
from models.pytorch_v3.lm.rnnlm import RNNLM
from models.test.data import generate_data
from utils.measure_time_func import measure_time
from utils.training.learning_rate_controller import Controller


class TestRNNLM(unittest.TestCase):

    def test(self):
        print("RNNLM Working check.")

        # unidirectional & bidirectional
        self.check(rnn_type='lstm', bidirectional=True)
        self.check(rnn_type='lstm', bidirectional=False)
        self.check(rnn_type='gru', bidirectional=True)
        self.check(rnn_type='gru', bidirectional=False)

        # Tie weights
        # self.check(rnn_type='lstm', bidirectional=False,
        #            tie_weights=True)

        # word-level LM
        self.check(rnn_type='lstm', bidirectional=True,
                   label_type='word')

    @measure_time
    def check(self, rnn_type, bidirectional=False,
              label_type='char', tie_weights=False):

        print('==================================================')
        print('  label_type: %s' % label_type)
        print('  rnn_type: %s' % rnn_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  tie_weights: %s' % str(tie_weights))
        print('==================================================')

        # Load batch data
        _, ys, _, y_lens = generate_data(label_type=label_type,
                                         batch_size=2)

        if label_type == 'char':
            num_classes = 27
        elif label_type == 'word':
            num_classes = 11

        # Load model
        model = RNNLM(
            embedding_dim=128,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            num_units=256,
            num_layers=1,
            dropout_embedding=0.1,
            dropout_hidden=0.1,
            dropout_output=0.1,
            num_classes=num_classes,
            parameter_init_distribution='uniform',
            parameter_init=0.1,
            recurrent_weight_orthogonal=True,
            init_forget_gate_bias_with_one=True,
            tie_weights=tie_weights)

        # Count total parameters
        for name in sorted(list(model.num_params_dict.keys())):
            num_params = model.num_params_dict[name]
            print("%s %d" % (name, num_params))
        print("Total %.3f M parameters" % (model.total_parameters / 1000000))

        # Define optimizer
        learning_rate = 1e-3
        model.set_optimizer('adam',
                            learning_rate_init=learning_rate,
                            weight_decay=1e-8,
                            lr_schedule=False,
                            factor=0.1,
                            patience_epoch=5)

        # Define learning rate controller
        lr_controller = Controller(learning_rate_init=learning_rate,
                                   backend='pytorch',
                                   decay_type='per_epoch',
                                   decay_start_epoch=20,
                                   decay_rate=0.9,
                                   decay_patient_epoch=10,
                                   lower_better=True)

        # GPU setting
        model.set_cuda(deterministic=False, benchmark=True)

        # Train model
        max_step = 300
        start_time_step = time.time()
        for step in range(max_step):

            # Step for parameter update
            model.optimizer.zero_grad()
            loss = model(ys, y_lens)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 5)
            model.optimizer.step()

            # Inject Gaussian noise to all parameters
            if loss.data[0] < 50:
                model.weight_noise_injection = True

            if (step + 1) % 10 == 0:
                # Compute loss
                loss = model(ys, y_lens, is_eval=True)

                # Compute PPL
                ppl = math.exp(loss)

                duration_step = time.time() - start_time_step
                print('Step %d: loss=%.3f / ppl=%.3f / lr=%.5f (%.3f sec)' %
                      (step + 1, loss, ppl, learning_rate, duration_step))
                start_time_step = time.time()

                if ppl == 0:
                    print('Modle is Converged.')
                    break

                # Update learning rate
                model.optimizer, learning_rate = lr_controller.decay_lr(
                    optimizer=model.optimizer,
                    learning_rate=learning_rate,
                    epoch=step,
                    value=ppl)


if __name__ == "__main__":
    unittest.main()
