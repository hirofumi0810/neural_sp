#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test Joint CTC-Attention models in pytorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import unittest

import torch
import torch.nn as nn

sys.path.append('../../../')
from models.pytorch.attention.joint_ctc_attention import JointCTCAttention
from models.test.data import generate_data, idx2char, idx2word
from utils.measure_time_func import measure_time
from utils.io.variable import np2var, var2np
from utils.evaluation.edit_distance import compute_cer, compute_wer
from utils.training.learning_rate_controller import Controller

torch.manual_seed(2017)


class TestAttention(unittest.TestCase):

    def test(self):
        print("Attention Working check.")

        # self.check(encoder_type='lstm', bidirectional=True,
        #            decoder_type='lstm', label_type='word')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', label_type='char')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', label_type='word_char')

    @measure_time
    def check(self, encoder_type, bidirectional, decoder_type,
              attention_type='dot_product', label_type='char',
              downsample=False, input_feeding_approach=False):

        print('==================================================')
        print('  label_type: %s' % label_type)
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  decoder_type: %s' % decoder_type)
        print('  attention_type: %s' % attention_type)
        print('  downsample: %s' % str(downsample))
        print('  input_feeding_approach: %s' % str(input_feeding_approach))
        print('==================================================')

        # Load batch data
        inputs, labels, labels_ctc, inputs_seq_len, labels_seq_len, labels_seq_len_ctc = generate_data(
            model_type='joint_ctc_attention',
            label_type=label_type,
            batch_size=2,
            num_stack=1,
            splice=1)
        labels_ctc += 1
        # NOTE: index 0 is reserved for blank

        if label_type == 'char':
            num_classes = 27
            num_classes_sub = 27
        elif label_type == 'word':
            num_classes = 11
            num_classes_sub = 11
        elif label_type == 'word_char':
            num_classes = 27
            num_classes_sub = 11

        # Load model
        model = JointCTCAttention(
            input_size=inputs.size(-1),
            encoder_type=encoder_type,
            encoder_bidirectional=bidirectional,
            encoder_num_units=256,
            encoder_num_proj=0,
            encoder_num_layers=3,
            encoder_dropout=0.1,
            attention_type=attention_type,
            attention_dim=128,
            decoder_type=decoder_type,
            decoder_num_units=256,
            decoder_num_proj=128,
            decoder_num_layers=1,
            decoder_dropout=0.1,
            embedding_dim=64,
            embedding_dropout=0.1,
            num_classes=num_classes,
            ctc_num_layers=2,
            ctc_loss_weight=0.1,
            ctc_num_classes=num_classes_sub,
            max_decode_length=100,
            splice=1,
            parameter_init=0.1,
            # downsample_list=[] if not downsample else [True] * 2,
            downsample_list=[],
            init_dec_state_with_enc_state=True,
            sharpening_factor=1,
            logits_temperature=1,
            sigmoid_smoothing=False,
            input_feeding_approach=input_feeding_approach)

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

        # Wrap by Variable
        inputs = np2var(inputs, use_cuda=use_cuda)
        # labels must be long
        labels = np2var(labels, dtype='long', use_cuda=use_cuda)
        labels_ctc = np2var(labels_ctc, dtype='int', use_cuda=use_cuda)
        inputs_seq_len = np2var(inputs_seq_len, dtype='int', use_cuda=use_cuda)
        labels_seq_len = np2var(labels_seq_len, dtype='int', use_cuda=use_cuda)
        labels_seq_len_ctc = np2var(
            labels_seq_len_ctc, dtype='int', use_cuda=use_cuda)

        # Train model
        max_step = 1000
        start_time_step = time.time()
        ler_pre = 1
        for step in range(max_step):

            # Clear gradients before
            optimizer.zero_grad()

            # Compute loss
            logits, att_weights, logits_ctc, perm_indices = model(
                inputs, inputs_seq_len, labels)
            loss = model.compute_loss(
                logits,
                inputs_seq_len[perm_indices],
                labels[perm_indices],
                labels_seq_len[perm_indices],
                logits_ctc,
                labels_ctc,
                labels_seq_len_ctc,
                att_weights, coverage_weight=0.5)

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
                labels_pred, _ = model.decode_infer(
                    inputs, inputs_seq_len, beam_width=5)

                # Compute accuracy
                if label_type == 'char':
                    str_true = idx2char(var2np(labels[perm_indices])[
                                        0, :var2np(labels_seq_len[perm_indices])[0]][1:-1])
                    str_pred = idx2char(labels_pred[0][0:-1]).split('>')[0]
                    ler = compute_cer(ref=str_true.replace('_', ''),
                                      hyp=str_pred.replace('_', ''),
                                      normalize=True)
                elif label_type == 'word':
                    str_true = idx2word(var2np(labels[perm_indices])[
                                        0, :var2np(labels_seq_len[perm_indices])[0]][1:-1])
                    str_pred = idx2word(labels_pred[0][0:-1]).split('>')[0]
                    ler = compute_wer(ref=str_true.split('_'),
                                      hyp=str_pred.split('_'),
                                      normalize=True)

                # ***Change to training mode***
                model.train()

                duration_step = time.time() - start_time_step
                print('Step %d: loss = %.3f / ler = %.3f / ler (ctc) = ? / lr = %.5f (%.3f sec)' %
                      (step + 1, var2np(loss), ler, learning_rate, duration_step))
                start_time_step = time.time()

                # Visualize
                print('Ref: %s' % str_true)
                print('Hyp: %s' % str_pred)

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
