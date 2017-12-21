#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test attention-besed models in pytorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import unittest

import torch
import torch.nn as nn

sys.path.append('../../../')
from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq
from models.test.data import generate_data, idx2char, idx2word
from utils.measure_time_func import measure_time
from utils.io.variable import np2var, var2np
from utils.evaluation.edit_distance import compute_cer, compute_wer
from utils.training.learning_rate_controller import Controller

torch.manual_seed(2017)


class TestAttention(unittest.TestCase):

    def test(self):
        print("Attention Working check.")

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

        # Joint CTC-Attention
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', ctc_loss_weight=0.1)

        # multiple layer decoder
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', decoder_num_layers=2)

        # word-level attention
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='dot_product',
                   label_type='word')

        # unidirectional & bidirectional
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm')
        self.check(encoder_type='lstm', bidirectional=False,
                   decoder_type='lstm')
        self.check(encoder_type='gru', bidirectional=True,
                   decoder_type='gru')
        self.check(encoder_type='gru', bidirectional=False,
                   decoder_type='gru')

        # Pyramidal encoder
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', subsample=True)
        self.check(encoder_type='gru', bidirectional=True,
                   decoder_type='gru', subsample=True)

        # Attention type
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='content')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='location')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='dot_product')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='luong_dot')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='luong_general')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='luong_concat')
        # self.check(encoder_type='lstm', bidirectional=True,
        #            decoder_type='lstm', attention_type='scaled_luong_dot')
        # self.check(encoder_type='lstm', bidirectional=True,
        #            decoder_type='lstm', attention_type='normed_content')

    @measure_time
    def check(self, encoder_type, bidirectional, decoder_type,
              attention_type='location', label_type='char',
              subsample=False, ctc_loss_weight=0, decoder_num_layers=1,
              conv=False, batch_norm=False,
              residual=False, dense_residual=False):

        print('==================================================')
        print('  label_type: %s' % label_type)
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  decoder_type: %s' % decoder_type)
        print('  attention_type: %s' % attention_type)
        print('  subsample: %s' % str(subsample))
        print('  ctc_loss_weight: %s' % str(ctc_loss_weight))
        print('  decoder_num_layers: %s' % str(decoder_num_layers))
        print('  conv: %s' % str(conv))
        print('  batch_norm: %s' % str(batch_norm))
        print('  residual: %s' % str(residual))
        print('  dense_residual: %s' % str(dense_residual))
        print('==================================================')

        if conv:
            conv_channels = [32, 32]
            # pattern 1
            conv_kernel_sizes = [[41, 11], [21, 11]]
            conv_strides = [[2, 2], [2, 1]]

            # pattern 2
            # conv_kernel_sizes = [[8, 5], [8, 5]]
            # conv_strides = [[2, 2], [1, 1]]

            # poolings = [[], []]
            poolings = [[2, 2], [2, 2]]
            # poolings = [[2, 2], []]
            # poolings = [[], [2, 2]]
        else:
            conv_channels = []
            conv_kernel_sizes = []
            conv_strides = []
            poolings = []

        # Load batch data
        splice = 1
        num_stack = 1 if subsample or conv else 2
        inputs, labels, inputs_seq_len, labels_seq_len = generate_data(
            model_type='attention',
            label_type=label_type,
            batch_size=2,
            num_stack=num_stack,
            splice=splice)

        if label_type == 'char':
            num_classes = 27
            map_fn = idx2char
        elif label_type == 'word':
            num_classes = 11
            map_fn = idx2word

        # Load model
        model = AttentionSeq2seq(
            input_size=inputs.shape[-1] // splice // num_stack,  # 120
            encoder_type=encoder_type,
            encoder_bidirectional=bidirectional,
            encoder_num_units=256,
            encoder_num_proj=0,
            encoder_num_layers=2,
            encoder_dropout=0.1,
            attention_type=attention_type,
            attention_dim=128,
            decoder_type=decoder_type,
            decoder_num_units=256,
            decoder_num_layers=decoder_num_layers,
            decoder_dropout=0.1,
            embedding_dim=32,
            num_classes=num_classes,
            ctc_loss_weight=ctc_loss_weight,
            parameter_init=0.1,
            subsample_list=[] if not subsample else [True] * 2,
            init_dec_state_with_enc_state=True,
            sharpening_factor=1,
            logits_temperature=1,
            sigmoid_smoothing=False,
            coverage_weight=0.5,
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
            label_smoothing_prob=0.1,
            weight_noise_std=0,
            residual=residual,
            dense_residual=dense_residual)

        # Count total parameters
        for name in sorted(list(model.num_params_dict.keys())):
            num_params = model.num_params_dict[name]
            print("%s %d" % (name, num_params))
        print("Total %.3f M parameters" % (model.total_parameters / 1000000))

        # Define optimizer
        optimizer, scheduler = model.set_optimizer(
            'adam',
            learning_rate_init=1e-3,
            weight_decay=1e-8,
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
            loss = model(inputs, labels, inputs_seq_len, labels_seq_len)

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

            # Inject Gaussian noise to all parameters
            if loss.data[0] < 50:
                model.weight_noise_injection = True

            if (step + 1) % 10 == 0:
                # ***Change to evaluation mode***
                model.eval()

                # Decode
                labels_pred = model.decode(
                    inputs, inputs_seq_len,
                    # beam_width=1,
                    beam_width=2,
                    max_decode_length=60)

                # Compute accuracy
                if label_type == 'char':
                    str_true = map_fn(labels[0, :labels_seq_len[0]][1:-1])
                    str_pred = map_fn(labels_pred[0][0:-1]).split('>')[0]
                    ler = compute_cer(ref=str_true.replace('_', ''),
                                      hyp=str_pred.replace('_', ''),
                                      normalize=True)
                elif label_type == 'word':
                    str_true = map_fn(labels[0, : labels_seq_len[0]][1: -1])
                    str_pred = map_fn(labels_pred[0][0: -1]).split('>')[0]
                    ler = compute_wer(ref=str_true.split('_'),
                                      hyp=str_pred.split('_'),
                                      normalize=True)

                # ***Change to training mode***
                model.train()

                duration_step = time.time() - start_time_step
                print('Step %d: loss = %.3f / ler = %.3f / lr = %.5f (%.3f sec)' %
                      (step + 1, loss.data[0], ler, learning_rate, duration_step))
                start_time_step = time.time()

                # Visualize
                print('Ref: %s' % str_true)
                print('Hyp: %s' % str_pred)

                # Decode by theCTC decoder
                if model.ctc_loss_weight >= 0.1:
                    labels_pred_ctc = model.decode_ctc(
                        inputs, inputs_seq_len, beam_width=1)
                    str_pred_ctc = map_fn(labels_pred_ctc[0])
                    print('Hyp (CTC): %s' % str_pred_ctc)

                if ler < 0.1:
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
