#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test nested attention models v2 (pytorch)."""

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
from models.pytorch.attention.nested_attention_seq2seq_v2 import NestedAttentionSeq2seq
from models.test.data import generate_data, idx2char, idx2word
from utils.measure_time_func import measure_time
from utils.evaluation.edit_distance import compute_cer, compute_wer
from utils.training.learning_rate_controller import Controller


class TestCharseqAttention(unittest.TestCase):

    def test(self):
        print("Hierarchical Attention Working check.")

        # Curriculum training
        # self.check(composition_case='fine_grained_gating',
        #            ctc_loss_weight_sub=0.2, curriculum_training=True)

        # Composition case (word attn + char attn)
        self.check(composition_case='fine_grained_gating')
        self.check(composition_case='scalar_gating')
        self.check(composition_case='concat')

        # Composition case (word attn + char attn + char CTC)
        self.check(composition_case='fine_grained_gating',
                   ctc_loss_weight_sub=0.1)
        self.check(composition_case='scalar_gating',
                   ctc_loss_weight_sub=0.1)
        self.check(composition_case='concat',
                   ctc_loss_weight_sub=0.1)

    @measure_time
    def check(self, composition_case, ctc_loss_weight_sub=0,
              curriculum_training=False):

        print('==================================================')
        print('  ctc_loss_weight_sub: %s' % str(ctc_loss_weight_sub))
        print('  curriculum_training: %s' % str(curriculum_training))
        print('  composition_case: %s' % str(composition_case))
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
            main_loss_weight=0.8,
            num_classes=11,
            num_classes_sub=27,
            parameter_init_distribution='uniform',
            parameter_init=0.1,
            recurrent_weight_orthogonal=False,
            init_forget_gate_bias_with_one=True,
            subsample_list=[False, True, False],
            subsample_type='drop',
            init_dec_state='zero',
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
            curriculum_training=curriculum_training,
            composition_case=composition_case)

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
                # Compute loss
                loss, loss_main, loss_sub = model(
                    xs, ys, ys_sub, x_lens, y_lens, y_lens_sub, is_eval=True)

                # Decode
                best_hyps, best_hyps_sub, perm_idx = model.decode(
                    xs, x_lens, beam_width=1, max_decode_len=30,
                    max_decode_len_sub=60)
                best_hyps_sub, perm_idx_sub = model.decode(
                    xs, x_lens, beam_width=1, max_decode_len=60,
                    is_sub_task=True)

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
