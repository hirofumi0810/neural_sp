#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test attention-besed models (pytorch)."""

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
from models.pytorch.lm.rnnlm import RNNLM
from models.test.data import generate_data, idx2char, idx2word
from utils.measure_time_func import measure_time
from utils.evaluation.edit_distance import compute_cer, compute_wer
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
        self.check(rnn_type='lstm', bidirectional=False,
                   tie_weights=True)

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
        _, ys, _, y_lens = generate_data(model_type='lm',
                                         label_type=label_type,
                                         batch_size=2)

        if label_type == 'char':
            num_classes = 27
            map_fn = idx2char
        elif label_type == 'word':
            num_classes = 11
            map_fn = idx2word

        # Load model
        model = RNNLM(
            num_classes,
            embedding_dim=128,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            num_units=1024,
            num_layers=1,
            dropout_embedding=0.1,
            dropout_hidden=0.1,
            dropout_output=0.1,
            parameter_init_distribution='uniform',
            parameter_init=0.1,
            tie_weights=False)

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
                                   decay_start_epoch=20,
                                   decay_rate=0.9,
                                   decay_patient_epoch=10,
                                   lower_better=True)

        # GPU setting
        model.set_cuda(deterministic=False, benchmark=True)

        # Train model
        max_step = 1000
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

                # Decode
                # best_hyps, perm_idx = model.decode(
                #     xs, x_lens,
                #     # beam_width=1,
                #     beam_width=2,
                #     max_decode_len=60)

                # Compute accuracy
                # if label_type == 'char':
                #     str_true = map_fn(ys[0, :y_lens[0]][1:-1])
                #     str_pred = map_fn(best_hyps[0][0:-1]).split('>')[0]
                #     ler = compute_cer(ref=str_true.replace('_', ''),
                #                       hyp=str_pred.replace('_', ''),
                #                       normalize=True)
                # elif label_type == 'word':
                #     str_true = map_fn(ys[0, : y_lens[0]][1: -1])
                #     str_pred = map_fn(best_hyps[0][0: -1]).split('>')[0]
                #     ler, _, _, _ = compute_wer(ref=str_true.split('_'),
                #                                hyp=str_pred.split('_'),
                #                                normalize=True)

                duration_step = time.time() - start_time_step
                print('Step %d: loss=%.3f / ppl=%.3f / lr=%.5f (%.3f sec)' %
                      (step + 1, loss, ppl, learning_rate, duration_step))
                start_time_step = time.time()

                # Visualize
                # print('Ref: %s' % str_true)
                # print('Hyp: %s' % str_pred)

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
