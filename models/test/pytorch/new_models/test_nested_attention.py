#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test nested attention-besed models with word-character composition (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import unittest

import torch
import torch.nn as nn
torch.manual_seed(1623)
torch.cuda.manual_seed_all(1623)

sys.path.append('../../../../')
from models.pytorch.attention.nested_attention_seq2seq import NestedAttentionSeq2seq
from models.test.data import generate_data, idx2char, idx2word
from utils.measure_time_func import measure_time
from utils.evaluation.edit_distance import compute_cer, compute_wer
from utils.training.learning_rate_controller import Controller


class TestNestedAttention(unittest.TestCase):

    def test(self):
        print("Nested Attention Working check.")

        # self.check(composition_case='hidden')
        self.check(composition_case='embedding')
        self.check(composition_case='hidden_embedding')
        self.check(composition_case='multiscale')

    @measure_time
    def check(self, composition_case,
              encoder_type='lstm', bidirectional=True, decoder_type='lstm',
              attention_type='location', subsample=False,
              ctc_loss_weight=0, decoder_num_layers=2):

        print('==================================================')
        print('  composition_case: %s' % composition_case)
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  decoder_type: %s' % decoder_type)
        print('  attention_type: %s' % attention_type)
        print('  subsample: %s' % str(subsample))
        print('  ctc_loss_weight: %s' % str(ctc_loss_weight))
        print('  decoder_num_layers: %s' % str(decoder_num_layers))
        print('==================================================')

        # Load batch data
        num_stack = 1 if subsample else 2
        splice = 1
        xs, ys, ys_sub, x_lens, y_lens, y_lens_sub = generate_data(
            model_type='attention',
            label_type='word_char',
            batch_size=2,
            num_stack=num_stack,
            splice=splice)

        num_classes = 11
        num_classes_sub = 27

        # Load model
        model = NestedAttentionSeq2seq(
            input_size=xs.shape[-1] // splice // num_stack,  # 120
            encoder_type=encoder_type,
            encoder_bidirectional=bidirectional,
            encoder_num_units=256,
            encoder_num_proj=256,
            encoder_num_layers=3,
            encoder_num_layers_sub=2,
            attention_type=attention_type,
            attention_dim=128,
            decoder_type=decoder_type,
            decoder_num_units=256,
            decoder_num_layers=decoder_num_layers,
            decoder_num_units_sub=256,
            decoder_num_layers_sub=1,
            embedding_dim=64,
            embedding_dim_sub=32,
            dropout_input=0.1,
            dropout_encoder=0.1,
            dropout_decoder=0.1,
            dropout_embedding=0.1,
            main_loss_weight=0.1,
            num_classes=num_classes,
            num_classes_sub=num_classes_sub,
            parameter_init_distribution='uniform',
            parameter_init=0.1,
            recurrent_weight_orthogonal=False,
            init_forget_gate_bias_with_one=True,
            subsample_list=[] if not subsample else [True, False, False],
            subsample_type='drop',
            init_dec_state='final',
            sharpening_factor=1,
            logits_temperature=1,
            sigmoid_smoothing=False,
            ctc_loss_weight_sub=0.1,
            attention_conv_num_channels=10,
            attention_conv_width=101,
            num_stack=num_stack,
            splice=splice,
            conv_channels=[],
            conv_kernel_sizes=[],
            conv_strides=[],
            poolings=[],
            batch_norm=False,
            scheduled_sampling_prob=0.1,
            scheduled_sampling_ramp_max_step=200,
            # label_smoothing_prob=0.1,
            label_smoothing_prob=0,
            weight_noise_std=0,
            composition_case=composition_case,
            space_index=0)

        # Count total parameters
        for name in sorted(list(model.num_params_dict.keys())):
            num_params = model.num_params_dict[name]
            print("%s %d" % (name, num_params))
        print("Total %.3f M parameters" % (model.total_parameters / 1000000))

        # Define optimizer
        learning_rate = 1e-3
        model.set_optimizer('adam',
                            learning_rate_init=learning_rate,
                            weight_decay=1e-6,
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
            loss, loss_main, loss_sub = model(
                xs, ys, ys_sub, x_lens, y_lens, y_lens_sub)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 5)
            model.optimizer.step()

            if (step + 1) % 10 == 0:
                model.main_loss_weight = 0.1 + (1 - 0.1) / max_step * step

                # Decode
                best_hyps, perm_idx = model.decode(
                    xs, x_lens, beam_width=1, max_decode_len=30)
                best_hyps_sub, perm_idx_sub = model.decode(
                    xs, x_lens, beam_width=1, max_decode_len=60,
                    is_sub_task=True)

                # Compute accuracy
                str_pred = idx2word(best_hyps[0][0:-1]).split('>')[0]
                str_true = idx2word(ys[0][1:-1])
                ler, _, _, _ = compute_wer(ref=str_true.split('_'),
                                           hyp=str_pred.split('_'),
                                           normalize=True)
                str_pred_sub = idx2char(best_hyps_sub[0][0:-1]).split('>')[0]
                str_true_sub = idx2char(ys_sub[0][1:-1])
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
                # Update learning rate
                model.optimizer, learning_rate = lr_controller.decay_lr(
                    optimizer=model.optimizer,
                    learning_rate=learning_rate,
                    epoch=step,
                    value=ler)


if __name__ == "__main__":
    unittest.main()
