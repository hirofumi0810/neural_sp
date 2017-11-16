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

        self.check(encoder_type='lstm', bidirectional=True)

    @measure_time
    def check(self, encoder_type, bidirectional=False):

        print('==================================================')
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('==================================================')

        # Load batch data
        inputs, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub = generate_data(
            model='ctc',
            label_type='word_char',
            batch_size=2,
            num_stack=1,
            splice=1)
        labels += 1
        labels_sub += 1
        # NOTE: index 0 is reserved for blank

        # Wrap by Variable
        inputs = np2var(inputs)
        labels = np2var(labels, dtype='int')
        labels_sub = np2var(labels_sub, dtype='int')
        inputs_seq_len = np2var(inputs_seq_len, dtype='int')
        labels_seq_len = np2var(labels_seq_len, dtype='int')
        labels_seq_len_sub = np2var(labels_seq_len_sub, dtype='int')

        num_classes = 11
        num_classes_sub = 27

        # Load model
        model = HierarchicalCTC(
            input_size=inputs.size(-1),
            encoder_type=encoder_type,
            bidirectional=bidirectional,
            num_units=256,
            num_proj=None,
            num_layers=3,
            num_layers_sub=2,
            dropout=0.1,
            num_classes=num_classes,
            num_classes_sub=num_classes_sub,
            splice=1,
            parameter_init=0.1,
            bottleneck_dim=None)

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
        use_cuda = model.use_cuda
        model.set_cuda(deterministic=False)
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
            labels_sub = labels_sub.cuda()
            inputs_seq_len = inputs_seq_len.cuda()
            labels_seq_len = labels_seq_len.cuda()
            labels_seq_len_sub = labels_seq_len_sub.cuda()

        # Train model
        max_step = 1000
        start_time_step = time.time()
        ler_pre = 1
        for step in range(max_step):

            # Clear gradients before
            optimizer.zero_grad()

            # Compute loss
            logits, logits_sub, perm_indices = model(inputs, inputs_seq_len)
            loss = model.compute_loss(
                logits,
                labels[perm_indices],
                inputs_seq_len[perm_indices],
                labels_seq_len[perm_indices])
            loss += model.compute_loss(
                logits_sub,
                labels_sub[perm_indices],
                inputs_seq_len[perm_indices],
                labels_seq_len_sub[perm_indices])

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
                    logits, inputs_seq_len[perm_indices], beam_width=5)
                labels_pred_sub = model.decode(
                    logits_sub, inputs_seq_len[perm_indices], beam_width=5)

                # Compute accuracy
                str_true = idx2word(
                    var2np(labels[perm_indices][0, :var2np(labels_seq_len[perm_indices])[0]] - 1))
                str_pred = idx2word(labels_pred[0] - 1)
                ler = compute_wer(ref=str_true.split('_'),
                                  hyp=str_pred.split('_'),
                                  normalize=True)
                str_true_sub = idx2char(
                    var2np(labels_sub[perm_indices][0, :var2np(labels_seq_len_sub[perm_indices])[0]] - 1))
                str_pred_sub = idx2char(labels_pred_sub[0] - 1)
                ler_sub = compute_cer(ref=str_true_sub.replace('_', ''),
                                      hyp=str_pred_sub.replace('_', ''),
                                      normalize=True)

                # ***Change to training mode***
                model.train()

                duration_step = time.time() - start_time_step
                print('Step %d: loss = %.3f / ler (main) = %.3f / ler (sub) = %.3f / lr = %.5f (%.3f sec)' %
                      (step + 1, var2np(loss), ler, ler_sub, learning_rate, duration_step))
                start_time_step = time.time()

                # Visualize
                print('Ref (word): %s' % str_true)
                print('Hyp (word): %s' % str_pred)
                print('Ref (char): %s' % str_true_sub)
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
