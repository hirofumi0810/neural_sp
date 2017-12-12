#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test hierarchical CTC models in pytorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import unittest

import torch
import torch.nn as nn

sys.path.append('../../../')
from models.pytorch.ctc.hierarchical_ctc import HierarchicalCTC
from models.test.data import generate_data, idx2char, idx2word
from utils.io.variable import np2var, var2np
from utils.measure_time_func import measure_time
from utils.evaluation.edit_distance import compute_cer, compute_wer
from utils.training.learning_rate_controller import Controller

torch.manual_seed(2017)


class TestCTC(unittest.TestCase):

    def test(self):
        print("Hierarchical CTC Working check.")

        # Pyramidal encoder
        self.check(encoder_type='lstm', bidirectional=True,
                   subsample=True)

        # CLDNN-CTC
        self.check(encoder_type='lstm', bidirectional=True,
                   conv=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   conv=True, batch_norm=True)

        self.check(encoder_type='lstm', bidirectional=True)

    @measure_time
    def check(self, encoder_type, bidirectional=False,
              subsample=False, conv=False, batch_norm=False):

        print('==================================================')
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  subsample: %s' % str(subsample))
        print('  conv: %s' % str(conv))
        print('  batch_norm: %s' % str(batch_norm))
        print('==================================================')

        if conv:
            splice = 5
            conv_channels = [32, 32]
            conv_kernel_sizes = [[41, 11], [21, 11]]
            conv_strides = [[2, 2], [2, 1]]  # freq * time
            poolings = [[2, 2], [2, 2]]
            fc_list = [786, 786]
        else:
            splice = 1
            conv_channels = []
            conv_kernel_sizes = []
            conv_strides = []
            poolings = []
            fc_list = []

        # Load batch data
        num_stack = 1 if subsample or conv else 2
        inputs, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub = generate_data(
            model_type='ctc',
            label_type='word_char',
            batch_size=2,
            num_stack=num_stack,
            splice=splice)

        num_classes = 11
        num_classes_sub = 27

        # Load model
        model = HierarchicalCTC(
            input_size=inputs.shape[-1] // splice // num_stack,   # 120
            encoder_type=encoder_type,
            bidirectional=bidirectional,
            num_units=256,
            num_proj=None,
            num_layers=3,
            num_layers_sub=2,
            fc_list=fc_list,
            dropout=0.1,
            main_loss_weight=0.5,
            num_classes=num_classes,
            num_classes_sub=num_classes_sub,
            parameter_init=0.1,
            subsample_list=[] if not subsample else [True] * 3,
            num_stack=num_stack,
            splice=splice,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            poolings=poolings,
            batch_norm=batch_norm)

        # Count total parameters
        for name, num_params in model.num_params_dict.items():
            print("%s %d" % (name, num_params))
        print("Total %.3f M parameters" % (model.total_parameters / 1000000))

        # Define optimizer
        optimizer, scheduler = model.set_optimizer(
            'adam',
            learning_rate_init=1e-3,
            weight_decay=1e-6,
            lr_schedule=False,
            factor=0.1,
            patience_epoch=5)

        # Define learning rate controller
        learning_rate = 1e-3
        lr_controller = Controller(
            learning_rate_init=learning_rate,
            decay_start_epoch=20,
            decay_rate=0.9,
            decay_patient_epoch=10,
            lower_better=True)

        # Initialize parameters
        model.init_weights()

        # GPU setting
        model.set_cuda(deterministic=False)

        # Train model
        max_step = 1000
        start_time_step = time.time()
        ler_pre = 1
        for step in range(max_step):

            # Clear gradients before
            optimizer.zero_grad()

            # Compute loss
            loss, loss_main, loss_sub = model(
                inputs, labels, labels_sub,
                inputs_seq_len, labels_seq_len, labels_seq_len_sub)

            # Compute gradient
            optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm
            nn.utils.clip_grad_norm(model.parameters(), 10)

            # Update parameters
            if scheduler is not None:
                scheduler.step(ler_pre)
            else:
                optimizer.step()

            if (step + 1) % 10 == 0:
                # ***Change to evaluation mode***
                model.eval()

                # Decode
                labels_pred = model.decode(
                    inputs, inputs_seq_len, beam_width=1)
                labels_pred_sub = model.decode(
                    inputs, inputs_seq_len, beam_width=1, is_sub_task=True)

                # Compute accuracy
                str_true = idx2word(labels[0, :labels_seq_len[0]])
                str_pred = idx2word(labels_pred[0])
                ler = compute_wer(ref=str_true.split('_'),
                                  hyp=str_pred.split('_'),
                                  normalize=True)
                str_true_sub = idx2char(labels_sub[0, :labels_seq_len_sub[0]])
                str_pred_sub = idx2char(labels_pred_sub[0])
                ler_sub = compute_cer(ref=str_true_sub.replace('_', ''),
                                      hyp=str_pred_sub.replace('_', ''),
                                      normalize=True)

                # ***Change to training mode***
                model.train()

                duration_step = time.time() - start_time_step
                print('Step %d: loss = %.3f (%.3f/%.3f) / ler = %.3f (%.3f) / lr = %.5f (%.3f sec)' %
                      (step + 1, loss.data[0], loss_main.data[0], loss_sub.data[0],
                       ler, ler_sub, learning_rate, duration_step))
                start_time_step = time.time()

                # Visualize
                print('Ref: %s' % str_true)
                print('Hyp (word): %s' % str_pred)
                print('Hyp (char): %s' % str_pred_sub)

                if ler_sub < 0.1:
                    print('Modle is Converged.')
                    break
                ler_pre = ler

                # Update learning rate
                optimizer, learning_rate = lr_controller.decay_lr(
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    epoch=step,
                    value=ler)


if __name__ == "__main__":
    unittest.main()
