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

        # word-level attention
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='dot_product',
                   label_type='word')

        # unidirectional & bidirectional
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', save_path=None)
        self.check(encoder_type='lstm', bidirectional=False,
                   decoder_type='lstm')
        self.check(encoder_type='gru', bidirectional=True,
                   decoder_type='gru')
        self.check(encoder_type='gru', bidirectional=False,
                   decoder_type='gru')

        # Input-feeding approach
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', input_feeding_approach=True)

        # Pyramidal encoder
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', subsample=True)
        self.check(encoder_type='gru', bidirectional=True,
                   decoder_type='gru', subsample=True)

        # Attention type
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='bahdanau_content')
        # self.check(encoder_type='lstm', bidirectional=True,
        # decoder_type='lstm', attention_type='normed_bahdanau_content')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='location')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='hybrid')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='dot_product')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='luong_dot')
        # self.check(encoder_type='lstm', bidirectional=True,
        #            decoder_type='lstm', attention_type='scaled_luong_dot')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='luong_general')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='luong_concat')

    @measure_time
    def check(self, encoder_type, bidirectional, decoder_type,
              attention_type='dot_product', label_type='char',
              subsample=False, input_feeding_approach=False,
              save_path=None):

        print('==================================================')
        print('  label_type: %s' % label_type)
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  decoder_type: %s' % decoder_type)
        print('  attention_type: %s' % attention_type)
        print('  subsample: %s' % str(subsample))
        print('  input_feeding_approach: %s' % str(input_feeding_approach))
        print('==================================================')

        # Load batch data
        inputs, labels, inputs_seq_len, labels_seq_len = generate_data(
            model_type='attention',
            label_type=label_type,
            batch_size=2,
            num_stack=1,
            splice=1)

        if label_type == 'char':
            num_classes = 27
        elif label_type == 'word':
            num_classes = 11

        # Load model
        model = AttentionSeq2seq(
            input_size=inputs.shape[-1],
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
            decoder_num_proj=128,
            decoder_num_layers=1,
            decoder_dropout=0.1,
            embedding_dim=64,
            embedding_dropout=0.1,
            num_classes=num_classes,
            splice=1,
            parameter_init=0.1,
            subsample_list=[] if not subsample else [True] * 2,
            init_dec_state_with_enc_state=True,
            sharpening_factor=1,
            logits_temperature=1,
            sigmoid_smoothing=False,
            input_feeding_approach=input_feeding_approach,
            coverage_weight=0.5,
            ctc_loss_weight=0.1)

        # Count total parameters
        for name, num_params in model.num_params_dict.items():
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
        use_cuda = model.use_cuda
        model.set_cuda(deterministic=False)

        # Wrap by Variable
        inputs = np2var(inputs, use_cuda=use_cuda)
        # labels must be long
        labels = np2var(labels, dtype='long', use_cuda=use_cuda)
        inputs_seq_len = np2var(inputs_seq_len, dtype='int', use_cuda=use_cuda)
        labels_seq_len = np2var(labels_seq_len, dtype='int', use_cuda=use_cuda)

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

            if (step + 1) % 10 == 0:
                # ***Change to evaluation mode***
                model.eval()

                # Decode
                labels_pred, _ = model.decode_infer(
                    inputs, inputs_seq_len, beam_width=5, max_decode_length=100)

                # Compute accuracy
                if label_type == 'char':
                    str_true = idx2char(var2np(labels)[
                                        0, :var2np(labels_seq_len)[0]][1:-1])
                    str_pred = idx2char(labels_pred[0][0:-1]).split('>')[0]
                    ler = compute_cer(ref=str_true.replace('_', ''),
                                      hyp=str_pred.replace('_', ''),
                                      normalize=True)
                elif label_type == 'word':
                    str_true = idx2word(var2np(labels)[
                                        0, :var2np(labels_seq_len)[0]][1:-1])
                    str_pred = idx2word(labels_pred[0][0:-1]).split('>')[0]
                    ler = compute_wer(ref=str_true.split('_'),
                                      hyp=str_pred.split('_'),
                                      normalize=True)

                # ***Change to training mode***
                model.train()

                duration_step = time.time() - start_time_step
                print('Step %d: loss = %.3f / ler = %.3f / lr = %.5f (%.3f sec)' %
                      (step + 1, var2np(loss), ler, learning_rate, duration_step))
                start_time_step = time.time()

                # Visualize
                print('Ref: %s' % str_true)
                print('Hyp: %s' % str_pred)

                if ler < 0.1:
                    print('Modle is Converged.')
                    # Save the model
                    if save_path is not None:
                        saved_path = model.save_checkpoint(save_path, epoch=1)
                        print("=> Saved checkpoint (epoch:%d): %s" %
                              (1, saved_path))
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
