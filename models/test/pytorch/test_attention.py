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
from models.test.data import generate_data, np2var_pytorch, idx2alpha
from models.test.util import measure_time
from utils.io.tensor import to_np

torch.manual_seed(1)


class TestAttention(unittest.TestCase):

    def test(self):
        print("Attention Working check.")

        # self.check(encoder_type='lstm', bidirectional=False,
        #            decoder_type='lstm')
        # self.check(encoder_type='lstm', bidirectional=True,
        #            decoder_type='lstm')

        # Pyramidal encoder
        self.check(encoder_type='gru', bidirectional=True,
                   decoder_type='gru', attention_type='content',
                   downsample=True)
        self.check(encoder_type='gru', bidirectional=False,
                   decoder_type='gru', attention_type='content',
                   downsample=True)
        # NOTE: Pyramidal encoder does not work on GPU

        # Attention type
        self.check(encoder_type='gru', bidirectional=True,
                   decoder_type='gru', attention_type='content',
                   save_path='./')
        self.check(encoder_type='gru', bidirectional=False,
                   decoder_type='gru', attention_type='content')
        # self.check(encoder_type='gru', bidirectional=True,
        #            decoder_type='gru', attention_type='location')
        # self.check(encoder_type='gru', bidirectional=False,
        #            decoder_type='gru', attention_type='location')
        # self.check(encoder_type='gru', bidirectional=True,
        #            decoder_type='gru', attention_type='hybrid')
        # self.check(encoder_type='gru', bidirectional=False,
        #            decoder_type='gru', attention_type='hybrid')
        self.check(encoder_type='gru', bidirectional=True,
                   decoder_type='gru', attention_type='MLP_dot')
        self.check(encoder_type='gru', bidirectional=False,
                   decoder_type='gru', attention_type='MLP_dot')
        self.check(encoder_type='gru', bidirectional=True,
                   decoder_type='gru', attention_type='luong_dot')
        self.check(encoder_type='gru', bidirectional=False,
                   decoder_type='gru', attention_type='luong_dot')
        self.check(encoder_type='gru', bidirectional=True,
                   decoder_type='gru', attention_type='luong_general')
        self.check(encoder_type='gru', bidirectional=False,
                   decoder_type='gru', attention_type='luong_general')
        self.check(encoder_type='gru', bidirectional=True,
                   decoder_type='gru', attention_type='luong_concat')
        self.check(encoder_type='gru', bidirectional=False,
                   decoder_type='gru', attention_type='luong_concat')

    @measure_time
    def check(self, encoder_type, bidirectional, decoder_type, attention_type,
              downsample=False, save_path=None):

        print('==================================================')
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  decoder_type: %s' % decoder_type)
        print('  attention_type: %s' % attention_type)
        print('  downsample: %s' % str(downsample))
        print('==================================================')

        # Load batch data
        batch_size = 4
        inputs, labels, inputs_seq_len, labels_seq_len = generate_data(
            model='attention',
            batch_size=batch_size)

        # Wrap by Variable
        inputs = np2var_pytorch(inputs)
        labels = np2var_pytorch(labels, dtype='long')
        inputs_seq_len = np2var_pytorch(inputs_seq_len, dtype='long')
        labels_seq_len = np2var_pytorch(labels_seq_len, dtype='long')

        # Load model
        model = AttentionSeq2seq(
            input_size=inputs.size(-1),
            encoder_type=encoder_type,
            encoder_bidirectional=bidirectional,
            encoder_num_units=128 if bidirectional else 256,
            #  encoder_num_proj,
            encoder_num_layers=2,
            encoder_dropout=0.1,
            attention_type=attention_type,
            attention_dim=128,
            decoder_type=decoder_type,
            decoder_num_units=256,
            decoder_num_proj=128,
            #   decdoder_num_layers,
            decoder_dropout=0,
            embedding_dim=64,
            num_classes=27,  # alphabets + space (excluding <SOS> and <EOS>)
            eos_index=28,
            max_decode_length=100,
            splice=1,
            parameter_init=0.1,
            init_dec_state_with_enc_state=True,
            downsample_list=[] if not downsample else [True] * 2,
            sharpening_factor=2,
            logits_softmax_temperature=1)
        model.name = 'att_pytorch'

        # Define optimizer
        optimizer, scheduler = model.set_optimizer(
            'adam', learning_rate_init=1e-3, weight_decay=0,
            lr_schedule=False, factor=0.1, patience_epoch=5)

        # Initialize parameters
        model.init_weights()

        # Count total parameters
        print("Total %.3f M parameters" % (model.total_parameters / 1000000))

        # GPU setting
        use_cuda = torch.cuda.is_available()
        deterministic = False
        if use_cuda and deterministic:
            print('GPU deterministic mode (no cudnn)')
            torch.backends.cudnn.enabled = False
        elif use_cuda:
            print('GPU mode (faster than the deterministic mode)')
        else:
            print('CPU mode')
        if use_cuda:
            model = model.cuda()
            inputs = inputs.cuda()
            labels = labels.cuda()

        # Train model
        max_step = 1000
        start_time_step = time.time()
        ler_train_pre = 1
        save_flag = False
        for step in range(max_step):

            # Clear gradients before
            optimizer.zero_grad()

            # Make prediction
            outputs_train, att_weights = model(inputs, labels)

            # Compute loss
            loss = model.compute_loss(outputs_train, labels,
                                      att_weights, coverage_weight=0.5)

            # Compute gradient
            optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm
            nn.utils.clip_grad_norm(model.parameters(), 10)

            # Update parameters
            if scheduler is not None:
                scheduler.step(ler_train_pre)
            else:
                optimizer.step()

            if (step + 1) % 10 == 0:
                # Change to evaluation mode

                # Decode
                outputs_infer, _ = model.decode_infer(inputs, labels,
                                                      beam_width=1)

                # Compute accuracy

                duration_step = time.time() - start_time_step
                print('Step %d: loss = %.3f / ler = %.3f (%.3f sec) / lr = %.5f' %
                      (step + 1, to_np(loss), 1, duration_step, 1e-3))
                start_time_step = time.time()

                # Visualize
                print('Ref: %s' % idx2alpha(to_np(labels)[0][1:-1]))
                print('Hyp: %s' % idx2alpha(outputs_infer[0][0:-1]))

                if to_np(loss) < 30 and not save_flag:
                    # Save the model
                    if save_path is not None:
                        model.save_checkpoint(save_path, epoch=1)
                        print("=> Saved checkpoint (epoch:%d): %s" %
                              (1, save_path))
                        save_flag = True

                if to_np(loss) < 1.:
                    print('Modle is Converged.')
                    break
                # ler_train_pre = ler_train


if __name__ == "__main__":
    unittest.main()
