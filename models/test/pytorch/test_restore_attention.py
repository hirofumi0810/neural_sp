#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test loading Attention-besed models in pytorch."""

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
from utils.io.variable import np2var
from utils.measure_time_func import measure_time
from utils.io.tensor import tensor2np
from utils.evaluation.edit_distance import compute_cer

torch.manual_seed(1)


class TestRestoreAttention(unittest.TestCase):

    def test(self):
        print("Attention restoring check.")

        self.check()

    @measure_time
    def check(self):

        # Load batch data
        batch_size = 2
        inputs, labels, inputs_seq_len, labels_seq_len = generate_data(
            model='attention',
            batch_size=batch_size)

        # Wrap by Variable
        inputs = np2var(inputs)
        labels = np2var(labels, dtype='long')
        inputs_seq_len = np2var(inputs_seq_len, dtype='long')
        labels_seq_len = np2var(labels_seq_len, dtype='long')

        # Load model
        model = AttentionSeq2seq(
            input_size=inputs.size(-1),
            encoder_type='lstm',
            encoder_bidirectional=True,
            encoder_num_units=256,
            encoder_num_proj=0,
            encoder_num_layers=2,
            encoder_dropout=0.1,
            attention_type='bahdanau_content',
            attention_dim=128,
            decoder_type='lstm',
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
            downsample_list=[],
            init_dec_state_with_enc_state=True,
            sharpening_factor=1,
            logits_temperature=1,
            sigmoid_smoothing=False,
            input_feeding_approach=False)

        # Count total parameters
        print("Total %.3f M parameters" % (model.total_parameters / 1000000))

        # Define optimizer
        optimizer, scheduler = model.set_optimizer(
            'adam', learning_rate_init=1e-3, weight_decay=0,
            lr_schedule=False, factor=0.1, patience_epoch=5)

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

        # Load the saved model
        checkpoint = model.load_checkpoint(save_path='./', epoch=1)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        # Retrain model
        max_step = 200
        start_time_step = time.time()
        cer_train_pre = 1
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
                scheduler.step(cer_train_pre)
            else:
                optimizer.step()

            if (step + 1) % 10 == 0:
                # TODO: Change to evaluation mode

                # Decode
                outputs_infer, _ = model.decode_infer(inputs, beam_width=1)

                str_pred = idx2alpha(outputs_infer[0][0:-1]).split('>')[0]
                str_true = idx2alpha(tensor2np(labels)[0][1:-1])

                # Compute accuracy
                cer_train = compute_cer(str_pred=str_pred.replace('_', ''),
                                        str_true=str_true.replace('_', ''),
                                        normalize=True)

                duration_step = time.time() - start_time_step
                print('Step %d: loss = %.3f / ler = %.3f (%.3f sec) / lr = %.5f' %
                      (step + 1, tensor2np(loss), cer_train, duration_step, 1e-3))
                start_time_step = time.time()

                # Visualize
                print('Ref: %s' % str_true)
                print('Hyp: %s' % str_pred)

                if cer_train < 0.1:
                    print('Modle is Converged.')
                    break
                cer_train_pre = cer_train


if __name__ == "__main__":
    unittest.main()
