#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test hierarchical attention-besed models (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import unittest

sys.path.append('../../../../')
from models.chainer.attention.hierarchical_attention_seq2seq import HierarchicalAttentionSeq2seq
from models.test.data import generate_data, idx2char, idx2word
from utils.measure_time_func import measure_time
from utils.evaluation.edit_distance import compute_cer, compute_wer
from utils.training.learning_rate_controller import Controller


class TestHierarchicalAttention(unittest.TestCase):

    def test(self):
        print("Hierarchical Attention Working check.")

        # Curriculum training
        # self.check(encoder_type='lstm', bidirectional=True,
        #            decoder_type='lstm', curriculum_training=True)

        # Word attention + char CTC
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', ctc_loss_weight_sub=0.2)

        # Pyramidal encoder
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', subsample='drop')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', subsample='concat')

        # Projection layer
        self.check(encoder_type='lstm', bidirectional=True, projection=True,
                   decoder_type='lstm')

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

        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm')

    @measure_time
    def check(self, encoder_type, bidirectional, decoder_type,
              attention_type='location', subsample=False, projection=False,
              ctc_loss_weight_sub=0, conv=False, batch_norm=False,
              residual=False, dense_residual=False, curriculum_training=False):

        print('==================================================')
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  projection: %s' % str(projection))
        print('  decoder_type: %s' % decoder_type)
        print('  attention_type: %s' % attention_type)
        print('  subsample: %s' % str(subsample))
        print('  ctc_loss_weight_sub: %s' % str(ctc_loss_weight_sub))
        print('  conv: %s' % str(conv))
        print('  batch_norm: %s' % str(batch_norm))
        print('  residual: %s' % str(residual))
        print('  dense_residual: %s' % str(dense_residual))
        print('  curriculum_training: %s' % str(curriculum_training))
        print('==================================================')

        if conv or encoder_type == 'cnn':
            # pattern 1
            # conv_channels = [32, 32]
            # conv_kernel_sizes = [[41, 11], [21, 11]]
            # conv_strides = [[2, 2], [2, 1]]
            # poolings = [[], []]

            # pattern 2 (VGG like)
            conv_channels = [64, 64]
            conv_kernel_sizes = [[3, 3], [3, 3]]
            conv_strides = [[1, 1], [1, 1]]
            poolings = [[2, 2], [2, 2]]
        else:
            conv_channels = []
            conv_kernel_sizes = []
            conv_strides = []
            poolings = []

        # Load batch data
        splice = 1
        num_stack = 1 if subsample or conv else 2
        xs, ys, ys_sub, x_lens, y_lens, y_lens_sub = generate_data(
            label_type='word_char',
            batch_size=2,
            num_stack=num_stack,
            splice=splice,
            backend='chainer')

        num_classes = 11
        num_classes_sub = 27

        # Load model
        model = HierarchicalAttentionSeq2seq(
            input_size=xs[0].shape[-1] // splice // num_stack,  # 120
            encoder_type=encoder_type,
            encoder_bidirectional=bidirectional,
            encoder_num_units=320,
            encoder_num_proj=320 if projection else 0,
            encoder_num_layers=3,
            encoder_num_layers_sub=2,
            attention_type=attention_type,
            attention_dim=128,
            decoder_type=decoder_type,
            decoder_num_units=320,
            decoder_num_layers=1,
            decoder_num_units_sub=320,
            decoder_num_layers_sub=1,
            embedding_dim=64,
            embedding_dim_sub=32,
            dropout_input=0.1,
            dropout_encoder=0.1,
            dropout_decoder=0.1,
            dropout_embedding=0.1,
            main_loss_weight=0.8,
            num_classes=num_classes,
            num_classes_sub=num_classes_sub,
            parameter_init_distribution='uniform',
            parameter_init=0.1,
            recurrent_weight_orthogonal=False,
            init_forget_gate_bias_with_one=True,
            subsample_list=[] if not subsample else [True, True, False],
            subsample_type='concat' if subsample is False else subsample,
            init_dec_state='final',
            sharpening_factor=1,
            logits_temperature=1,
            sigmoid_smoothing=False,
            ctc_loss_weight_sub=ctc_loss_weight_sub,
            attention_conv_num_channels=10,
            attention_conv_width=201,
            num_stack=num_stack,
            splice=splice,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            poolings=poolings,
            batch_norm=batch_norm,
            scheduled_sampling_prob=0.1,
            scheduled_sampling_ramp_max_step=200,
            label_smoothing_prob=0.1,  # default
            weight_noise_std=0,
            encoder_residual=residual,
            encoder_dense_residual=dense_residual,
            decoder_residual=residual,
            decoder_dense_residual=dense_residual,
            curriculum_training=curriculum_training)

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
        # lr_controller = Controller(learning_rate_init=learning_rate,
        #                            backend='chainer',
        #                            decay_start_epoch=20,
        #                            decay_rate=0.9,
        #                            decay_patient_epoch=10,
        #                            lower_better=True)

        # GPU setting
        model.set_cuda(deterministic=False, benchmark=True)

        # Train model
        max_step = 1000
        start_time_step = time.time()
        for step in range(max_step):

            # Step for parameter update
            loss, loss_main, loss_sub = model(
                xs, ys, ys_sub, x_lens, y_lens, y_lens_sub)
            model.optimizer.target.cleargrads()
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            model.optimizer.update()

            if (step + 1) % 10 == 0:
                # Compute loss
                loss, loss_main, loss_sub = model(
                    xs, ys, ys_sub, x_lens, y_lens, y_lens_sub, is_eval=True)

                # Decode
                best_hyps, _ = model.decode(
                    xs, x_lens, beam_width=1, max_decode_len=30)
                best_hyps_sub, _ = model.decode(
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
                # model.optimizer, learning_rate = lr_controller.decay_lr(
                #     optimizer=model.optimizer,
                #     learning_rate=learning_rate,
                #     epoch=step,
                #     value=wer)


if __name__ == "__main__":
    unittest.main()
