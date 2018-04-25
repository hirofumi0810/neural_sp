#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test attention-besed models (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import unittest

sys.path.append('../../../../')
from models.chainer.attention.attention_seq2seq import AttentionSeq2seq
from models.test.data import generate_data, idx2char, idx2word
from utils.measure_time_func import measure_time
from utils.evaluation.edit_distance import compute_wer
from utils.training.learning_rate_controller import Controller


class TestAttention(unittest.TestCase):

    def test(self):
        print("Attention Working check.")

        # Backward decoder
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', backward_loss_weight=1)
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', backward_loss_weight=0.8)
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', backward_loss_weight=0.5)
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', backward_loss_weight=0.2)

        # CNN encoder
        self.check(encoder_type='cnn',
                   decoder_type='lstm', batch_norm=True)

        # CLDNN encoder
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', conv=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', conv=True, batch_norm=True)

        # Multi-head attention
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='content', num_heads=2)
        # self.check(encoder_type='lstm', bidirectional=True,
        #            decoder_type='lstm', attention_type='location', num_heads=2) # TODO
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='dot_product', num_heads=2)

        # Decoding order
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', decoding_order='conditional')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', decoding_order='attend_update_generate')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', decoding_order='attend_generate_update')

        # Decoder type
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm')

        # Joint CTC-Attention
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', ctc_loss_weight=0.2)

        # Initialize decoder state
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', init_dec_state='first')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', init_dec_state='final')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', init_dec_state='mean')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', init_dec_state='zero')

        # Pyramidal encoder
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', subsample='drop')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', subsample='concat')

        # Projection layer
        self.check(encoder_type='lstm', bidirectional=True, projection=True,
                   decoder_type='lstm')

        # Residual connection
        # self.check(encoder_type='lstm', bidirectional=True,
        #            decoder_type='lstm', residual=True)  # TODO
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', dense_residual=True)

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

        # Attention type
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='content')
        # # self.check(encoder_type='lstm', bidirectional=True,
        #            decoder_type='lstm', attention_type='location') # TODO
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='dot_product')

    @measure_time
    def check(self, encoder_type, decoder_type, bidirectional=False,
              attention_type='content', label_type='char',
              subsample=False, projection=False, init_dec_state='first',
              ctc_loss_weight=0, conv=False, batch_norm=False,
              residual=False, dense_residual=False,
              decoding_order='attend_generate_update',
              backward_loss_weight=False, num_heads=1):

        print('==================================================')
        print('  label_type: %s' % label_type)
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  projection: %s' % str(projection))
        print('  decoder_type: %s' % decoder_type)
        print('  init_dec_state: %s' % init_dec_state)
        print('  attention_type: %s' % attention_type)
        print('  subsample: %s' % str(subsample))
        print('  ctc_loss_weight: %s' % str(ctc_loss_weight))
        print('  conv: %s' % str(conv))
        print('  batch_norm: %s' % str(batch_norm))
        print('  residual: %s' % str(residual))
        print('  dense_residual: %s' % str(dense_residual))
        print('  decoding_order: %s' % decoding_order)
        print('  backward_loss_weight: %s' % str(backward_loss_weight))
        print('  num_heads: %s' % str(num_heads))
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
        num_stack = 1 if subsample or conv or encoder_type == 'cnn' else 2
        xs, ys, x_lens, y_lens = generate_data(
            label_type=label_type,
            batch_size=2,
            num_stack=num_stack,
            splice=splice,
            backend='chainer')

        if label_type == 'char':
            num_classes = 27
            map_fn = idx2char
        elif label_type == 'word':
            num_classes = 11
            map_fn = idx2word

        # Load model
        model = AttentionSeq2seq(
            input_size=xs[0].shape[-1] // splice // num_stack,  # 120
            encoder_type=encoder_type,
            encoder_bidirectional=bidirectional,
            encoder_num_units=320,
            encoder_num_proj=320 if projection else 0,
            encoder_num_layers=1 if not subsample else 2,
            attention_type=attention_type,
            attention_dim=320,
            decoder_type=decoder_type,
            decoder_num_units=320,
            decoder_num_layers=2,
            embedding_dim=32,
            dropout_input=0.1,
            dropout_encoder=0.1,
            dropout_decoder=0.1,
            dropout_embedding=0.1,
            num_classes=num_classes,
            parameter_init_distribution='uniform',
            parameter_init=0.1,
            recurrent_weight_orthogonal=False,
            # recurrent_weight_orthogonal=True,
            init_forget_gate_bias_with_one=True,
            subsample_list=[] if not subsample else [True, False],
            subsample_type='concat' if not subsample else subsample,
            bridge_layer=True,
            init_dec_state=init_dec_state,
            sharpening_factor=1,
            logits_temperature=1,
            sigmoid_smoothing=False,
            coverage_weight=0,
            ctc_loss_weight=ctc_loss_weight,
            attention_conv_num_channels=10,
            attention_conv_width=201,
            num_stack=num_stack,
            splice=splice,
            input_channel=3,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            poolings=poolings,
            activation='relu',
            batch_norm=batch_norm,
            scheduled_sampling_prob=0.1,
            scheduled_sampling_ramp_max_step=200,
            label_smoothing_prob=0.1,
            weight_noise_std=0,
            encoder_residual=residual,
            encoder_dense_residual=dense_residual,
            decoder_residual=residual,
            decoder_dense_residual=dense_residual,
            decoding_order=decoding_order,
            bottleneck_dim=256,
            backward_loss_weight=backward_loss_weight,
            num_heads=num_heads)

        # Count total parameters
        for name in sorted(list(model.num_params_dict.keys())):
            num_params = model.num_params_dict[name]
            print("%s %d" % (name, num_params))
        print("Total %.3f M parameters" % (model.total_parameters / 1000000))

        # Define optimizer
        learning_rate = 1e-3
        model.set_optimizer('adam',
                            learning_rate_init=learning_rate,
                            weight_decay=1e-8,
                            lr_schedule=False,
                            factor=0.1,
                            patience_epoch=5)

        # Define learning rate controller
        lr_controller = Controller(learning_rate_init=learning_rate,
                                   backend='chainer',
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
            loss = model(xs, ys, x_lens, y_lens)
            model.optimizer.target.cleargrads()
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            model.optimizer.update()

            # Inject Gaussian noise to all parameters

            if (step + 1) % 10 == 0:
                # Compute loss
                loss = model(xs, ys, x_lens, y_lens, is_eval=True)

                # Decode
                best_hyps, _ = model.decode(xs, x_lens,
                                            beam_width=1,
                                            # beam_width=2,  # TODO: fix bugs
                                            max_decode_len=60)

                str_ref = map_fn(ys[0])
                str_hyp = map_fn(best_hyps[0][:-1])

                # Compute accuracy
                try:
                    if label_type == 'char':
                        ler, _, _, _ = compute_wer(
                            ref=list(str_ref.replace('_', '')),
                            hyp=list(str_hyp.replace('_', '')),
                            normalize=True)
                    elif label_type == 'word':
                        ler, _, _, _ = compute_wer(ref=str_ref.split('_'),
                                                   hyp=str_hyp.split('_'),
                                                   normalize=True)
                except:
                    ler = 1

                duration_step = time.time() - start_time_step
                print('Step %d: loss=%.3f / ler=%.3f / lr=%.5f (%.3f sec)' %
                      (step + 1, loss, ler, learning_rate, duration_step))
                start_time_step = time.time()

                # Visualize
                print('Ref: %s' % str_ref)
                print('Hyp: %s' % str_hyp)

                # Decode by theCTC decoder
                if model.ctc_loss_weight >= 0.1:
                    best_hyps_ctc, _ = model.decode_ctc(
                        xs, x_lens, beam_width=1)
                    str_pred_ctc = map_fn(best_hyps_ctc[0])
                    print('Hyp (CTC): %s' % str_pred_ctc)

                if ler < 0.1:
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
