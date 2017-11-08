#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test Attention-besed models in pytorch."""

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
from models.test.data import generate_data, idx2alpha
from utils.measure_time_func import measure_time
from utils.io.tensor import to_np
from utils.io.variable import np2var_pytorch
from utils.evaluation.edit_distance import compute_cer
from utils.training.learning_rate_controller import Controller

torch.manual_seed(1)


class TestAttention(unittest.TestCase):

    def test(self):
        print("Attention Working check.")

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
                   decoder_type='lstm', downsample=True)
        self.check(encoder_type='gru', bidirectional=True,
                   decoder_type='gru', downsample=True)

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
              attention_type='bahdanau_content',
              downsample=False, input_feeding_approach=False,
              save_path=None):

        print('==================================================')
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  decoder_type: %s' % decoder_type)
        print('  attention_type: %s' % attention_type)
        print('  downsample: %s' % str(downsample))
        print('  input_feeding_approach: %s' % str(input_feeding_approach))
        print('==================================================')

        # Load batch data
        inputs, labels, _, _ = generate_data(
            model='attention',
            batch_size=2,
            num_stack=1,
            splice=1)

        # Wrap by Variable
        inputs = np2var_pytorch(inputs)
        labels = np2var_pytorch(labels, dtype='long')

        # Load model
        model = AttentionSeq2seq(
            input_size=inputs.size(-1),
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
            decdoder_num_layers=1,
            decoder_dropout=0.1,
            embedding_dim=64,
            embedding_dropout=0.1,
            num_classes=27,  # excluding <SOS> and <EOS>
            sos_index=27,
            eos_index=28,
            max_decode_length=100,
            splice=1,
            parameter_init=0.1,
            downsample_list=[] if not downsample else [True] * 2,
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
        if use_cuda:
            model = model.cuda()
            inputs = inputs.cuda()
            labels = labels.cuda()

        # Train model
        max_step = 1000
        start_time_step = time.time()
        cer_train_pre = 1
        for step in range(max_step):

            # Clear gradients before
            optimizer.zero_grad()

            # Make prediction
            outputs_train, att_weights = model(inputs, labels)

            # Compute loss
            loss = model.compute_loss(outputs_train, labels, att_weights,
                                      coverage_weight=0.5)

            # Compute gradient
            optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm
            nn.utils.clip_grad_norm(model.parameters(), 10)

            # Update parameters
            if scheduler is not None:
                scheduler.step(cer_train_pre)
            else:
                optimizer.step()

            if (step + 1) % 10 == 0:
                # TODO: Change to evaluation mode

                # Decode
                labels_pred, _ = model.decode_infer(inputs, beam_width=5)

                str_pred = idx2alpha(labels_pred[0][0:-1]).split('>')[0]
                str_true = idx2alpha(to_np(labels)[0][1:-1])

                # Compute accuracy
                cer_train = compute_cer(str_pred=str_pred.replace('_', ''),
                                        str_true=str_true.replace('_', ''),
                                        normalize=True)

                duration_step = time.time() - start_time_step
                print('Step %d: loss = %.3f / ler = %.3f (%.3f sec) / lr = %.5f' %
                      (step + 1, to_np(loss), cer_train, duration_step, 1e-3))
                start_time_step = time.time()

                # Visualize
                print('Ref: %s' % str_true)
                print('Hyp: %s' % str_pred)

                if cer_train < 0.1:
                    print('Modle is Converged.')
                    # Save the model
                    if save_path is not None:
                        saved_path = model.save_checkpoint(save_path, epoch=1)
                        print("=> Saved checkpoint (epoch:%d): %s" %
                              (1, saved_path))
                    break
                cer_train_pre = cer_train

                # Update learning rate
                optimizer, learning_rate = lr_controller.decay_lr(
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    epoch=step,
                    value=cer_train)


if __name__ == "__main__":
    unittest.main()
