#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test hierarchical attention-besed models in pytorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import unittest

import torch
import torch.nn as nn

sys.path.append('../../../')
from models.pytorch.attention.hierarchical_attention_seq2seq import HierarchicalAttentionSeq2seq
from models.test.data import generate_data, idx2char, idx2word
from utils.measure_time_func import measure_time
from utils.io.variable import np2var, var2np
from utils.evaluation.edit_distance import compute_cer, compute_wer
from utils.training.learning_rate_controller import Controller

torch.manual_seed(2017)


class TestHierarchicalAttention(unittest.TestCase):

    def test(self):
        print("Hierarchical Attention Working check.")

        # Label smoothing
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', label_smoothing=True)

        # Residual LSTM encoder
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', residual=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', dense_residual=True)

        # CLDNN encoder
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', conv=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', conv=True, batch_norm=True)

        # Word attention + char CTC
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', subsample=True,
                   ctc_loss_weight_sub=0.5)
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', subsample=True,
                   ctc_loss_weight_sub=0.5,
                   conv=True, batch_norm=True)

        # Pyramidal encoder
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', subsample=True)

        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm')

    @measure_time
    def check(self, encoder_type, bidirectional, decoder_type,
              attention_type='location', subsample=False,
              ctc_loss_weight_sub=0,
              conv=False, batch_norm=False,
              residual=False, dense_residual=False, label_smoothing=False):

        print('==================================================')
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  decoder_type: %s' % decoder_type)
        print('  attention_type: %s' % attention_type)
        print('  subsample: %s' % str(subsample))
        print('  ctc_loss_weight_sub: %s' % str(ctc_loss_weight_sub))
        print('  conv: %s' % str(conv))
        print('  batch_norm: %s' % str(batch_norm))
        print('  residual: %s' % str(residual))
        print('  dense_residual: %s' % str(dense_residual))
        print('  label_smoothing: %s' % str(label_smoothing))
        print('==================================================')

        if conv:
            conv_channels = [32, 32]
            conv_kernel_sizes = [[41, 11], [21, 11]]
            conv_strides = [[2, 2], [2, 1]]
            poolings = [[], []]
        else:
            conv_channels = []
            conv_kernel_sizes = []
            conv_strides = []
            poolings = []

        # Load batch data
        splice = 1
        num_stack = 1 if subsample or conv else 2
        inputs, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub = generate_data(
            model_type='attention',
            label_type='word_char',
            batch_size=2,
            num_stack=num_stack,
            splice=splice)

        num_classes = 11
        num_classes_sub = 27

        # Load model
        model = HierarchicalAttentionSeq2seq(
            input_size=inputs.shape[-1] // splice // num_stack,  # 120
            encoder_type=encoder_type,
            encoder_bidirectional=bidirectional,
            encoder_num_units=256,
            encoder_num_proj=0,
            encoder_num_layers=3,
            encoder_num_layers_sub=2,
            encoder_dropout=0.1,
            attention_type=attention_type,
            attention_dim=128,
            decoder_type=decoder_type,
            decoder_num_units=256,
            decoder_num_layers=2,
            decoder_num_units_sub=256,
            decoder_num_layers_sub=1,
            decoder_dropout=0.1,
            embedding_dim=64,
            embedding_dim_sub=32,
            main_loss_weight=0.5,
            num_classes=num_classes,
            num_classes_sub=num_classes_sub,
            parameter_init=0.1,
            subsample_list=[] if not subsample else [False, True, False],
            init_dec_state_with_enc_state=True,
            sharpening_factor=1,
            logits_temperature=1,
            sigmoid_smoothing=False,
            ctc_loss_weight_sub=ctc_loss_weight_sub,
            attention_conv_num_channels=10,
            attention_conv_width=101,
            num_stack=num_stack,
            splice=splice,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            poolings=poolings,
            batch_norm=batch_norm,
            scheduled_sampling_prob=0.1,
            scheduled_sampling_ramp_max_step=100,
            label_smoothing_prob=0.1 if label_smoothing else 0,
            weight_noise_std=0,
            encoder_residual=residual,
            encoder_dense_residual=dense_residual,
            decoder_residual=residual,
            decoder_dense_residual=dense_residual)

        # Count total parameters
        for name in sorted(list(model.num_params_dict.keys())):
            num_params = model.num_params_dict[name]
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
                # Decode
                labels_pred = model.decode(
                    inputs, inputs_seq_len, beam_width=1, max_decode_len=30)
                labels_pred_sub = model.decode(
                    inputs, inputs_seq_len, beam_width=1, max_decode_len=60,
                    is_sub_task=True)

                # Compute accuracy
                str_pred = idx2word(labels_pred[0][0:-1]).split('>')[0]
                str_true = idx2word(labels[0][1:-1])
                ler = compute_wer(ref=str_true.split('_'),
                                  hyp=str_pred.split('_'),
                                  normalize=True)
                str_pred_sub = idx2char(labels_pred_sub[0][0:-1]).split('>')[0]
                str_true_sub = idx2char(labels_sub[0][1:-1])
                ler_sub = compute_cer(ref=str_true_sub.replace('_', ''),
                                      hyp=str_pred_sub.replace('_', ''),
                                      normalize=True)

                duration_step = time.time() - start_time_step
                print('Step %d: loss = %.3f (%.3f/%.3f) / ler (main) = %.3f / ler (sub) = %.3f / lr = %.5f (%.3f sec)' %
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
