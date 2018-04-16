#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test nested attention models (pytorch)."""

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
from utils.evaluation.edit_distance import compute_wer
from utils.training.learning_rate_controller import Controller


class TestCharseqAttention(unittest.TestCase):

    def test(self):
        print("Hierarchical Attention Working check.")

        # char initialization
        # self.check(main_loss_weight=0)

        # how to use character-level decoder states
        self.check(usage_dec_sub='no_use')
        self.check(usage_dec_sub='update_decoder')
        self.check(usage_dec_sub='all')

        # attention_regularization
        self.check(attention_regularization=True)

        # smoothing
        self.check(dec_out_sub_attend_temperature=2)
        self.check(dec_out_sub_attend_temperature=3)
        self.check(dec_out_sub_sigmoid_smoothing=True)

        # gating mechanism
        self.check(gating_mechanism='scalar')
        self.check(gating_mechanism='elementwise')

    @measure_time
    def check(self, usage_dec_sub='update_decoder', gating_mechanism='no_gate',
              attention_regularization=True,
              main_loss_weight=0.5, ctc_loss_weight_sub=0,
              dec_out_sub_attend_temperature=1,
              dec_out_sub_sigmoid_smoothing=False):

        print('==================================================')
        print('  usage_dec_sub: %s' % usage_dec_sub)
        print('  gating_mechanism: %s' % gating_mechanism)
        print('  attention_regularization: %s' % str(attention_regularization))
        print('  main_loss_weight: %s' % str(main_loss_weight))
        print('  ctc_loss_weight_sub: %s' % str(ctc_loss_weight_sub))
        print('  dec_out_sub_attend_temperature: %s' %
              str(dec_out_sub_attend_temperature))
        print('  dec_out_sub_sigmoid_smoothing: %s' %
              str(dec_out_sub_sigmoid_smoothing))
        print('==================================================')

        # Load batch data
        splice = 1
        num_stack = 1
        xs, ys, ys_sub, x_lens, y_lens, y_lens_sub = generate_data(
            label_type='word_char',
            batch_size=2,
            num_stack=num_stack,
            splice=splice)

        # Load model
        model = NestedAttentionSeq2seq(
            input_size=xs.shape[-1] // splice // num_stack,  # 120
            encoder_type='lstm',
            encoder_bidirectional=True,
            encoder_num_units=256,
            encoder_num_proj=0,
            encoder_num_layers=3,
            encoder_num_layers_sub=2,
            attention_type='location',
            attention_dim=128,
            decoder_type='lstm',
            decoder_num_units=256,
            decoder_num_layers=2,
            decoder_num_units_sub=256,
            decoder_num_layers_sub=1,
            embedding_dim=64,
            embedding_dim_sub=32,
            dropout_input=0.1,
            dropout_encoder=0.1,
            dropout_decoder=0.1,
            dropout_embedding=0.1,
            main_loss_weight=main_loss_weight,
            num_classes=11,
            num_classes_sub=27,
            parameter_init_distribution='uniform',
            parameter_init=0.1,
            recurrent_weight_orthogonal=False,
            init_forget_gate_bias_with_one=True,
            subsample_list=[True, False, False],
            subsample_type='drop',
            init_dec_state='first',
            sharpening_factor=1,
            logits_temperature=1,
            sigmoid_smoothing=False,
            ctc_loss_weight_sub=ctc_loss_weight_sub,
            attention_conv_num_channels=10,
            attention_conv_width=201,
            num_stack=num_stack,
            splice=1,
            conv_channels=[],
            conv_kernel_sizes=[],
            conv_strides=[],
            poolings=[],
            batch_norm=False,
            scheduled_sampling_prob=0.1,
            scheduled_sampling_ramp_max_step=200,
            label_smoothing_prob=0.1,
            weight_noise_std=0,
            encoder_residual=False,
            encoder_dense_residual=False,
            decoder_residual=False,
            decoder_dense_residual=False,
            decoding_order='attend_generate_update',
            usage_dec_sub=usage_dec_sub,
            gating_mechanism=gating_mechanism,
            attention_regularization=attention_regularization,
            dec_out_sub_attend_temperature=dec_out_sub_attend_temperature,
            dec_out_sub_sigmoid_smoothing=dec_out_sub_attend_temperature)

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
        max_step = 300
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
                # Compute loss
                loss, loss_main, loss_sub = model(
                    xs, ys, ys_sub, x_lens, y_lens, y_lens_sub, is_eval=True)

                best_hyps, best_hyps_sub, perm_idx = model.decode(
                    xs, x_lens, beam_width=1,
                    max_decode_len=30,
                    max_decode_len_sub=60)

                str_hyp = idx2word(best_hyps[0][:-1]).split('>')[0]
                str_ref = idx2word(ys[0])
                str_hyp_sub = idx2char(best_hyps_sub[0][:-1]).split('>')[0]
                str_ref_sub = idx2char(ys_sub[0])

                # Compute accuracy
                try:
                    wer, _, _, _ = compute_wer(ref=str_ref.split('_'),
                                               hyp=str_hyp.split('_'),
                                               normalize=True)
                    cer, _, _, _ = compute_wer(
                        ref=list(str_ref_sub.replace('_', '')),
                        hyp=list(str_hyp_sub.replace('_', '')),
                        normalize=True)
                except:
                    wer = 1
                    cer = 1

                duration_step = time.time() - start_time_step
                print('Step %d: loss=%.3f(%.3f/%.3f) / wer=%.3f / cer=%.3f / lr=%.5f (%.3f sec)' %
                      (step + 1, loss, loss_main, loss_sub,
                       wer, cer, learning_rate, duration_step))
                start_time_step = time.time()

                # Visualize
                print('Ref: %s' % str_ref)
                print('Hyp (word): %s' % str_hyp)
                print('Hyp (char): %s' % str_hyp_sub)

                if cer < 0.1:
                    print('Modle is Converged.')
                    break

                # Update learning rate
                model.optimizer, learning_rate = lr_controller.decay_lr(
                    optimizer=model.optimizer,
                    learning_rate=learning_rate,
                    epoch=step,
                    value=wer)


if __name__ == "__main__":
    unittest.main()
