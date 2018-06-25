#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test attention-besed models (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import unittest
import argparse

import torch
torch.manual_seed(1623)
torch.cuda.manual_seed_all(1623)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_style("white")
blue = '#4682B4'
orange = '#D2691E'
green = '#006400'

sys.path.append('../../../../../')
from src.models.pytorch_v3.attention.attention_seq2seq import AttentionSeq2seq
from src.models.pytorch_v3.data_parallel import CustomDataParallel
from src.models.test.data import generate_data, idx2char, idx2word
from src.utils.measure_time_func import measure_time
from src.utils.evaluation.edit_distance import compute_wer
from src.bin.training.utils.learning_rate_controller import Controller

parser = argparse.ArgumentParser()
parser.add_argument('--ngpus', type=int, default=0,
                    help='the number of GPUs (negative value indicates CPU)')
args = parser.parse_args()


class TestAttention(unittest.TestCase):

    def test(self):
        print("Attention Working check.")

        # Multi-head attention
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='content', num_heads=2)
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='location', num_heads=2)
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='location', num_heads=2, beam_width=2)
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='dot_product', num_heads=2)

        # CNN encoder
        self.check(encoder_type='cnn', decoder_type='lstm', batch_norm=True)

        # Decoding order
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', decoding_order='bahdanau')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', decoding_order='luong')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', decoding_order='conditional')

        # Beam search
        self.check(encoder_type='lstm', bidirectional=True, beam_width=2,
                   decoder_type='lstm', decoding_order='bahdanau')
        self.check(encoder_type='lstm', bidirectional=True, beam_width=2,
                   decoder_type='lstm', decoding_order='luong')
        self.check(encoder_type='lstm', bidirectional=True, beam_width=2,
                   decoder_type='lstm', decoding_order='conditional')

        # Backward decoder
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', backward_loss_weight=1)
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', backward_loss_weight=0.8)
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', backward_loss_weight=0.5)
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', backward_loss_weight=0.2)
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', backward_loss_weight=0.8, beam_width=2)

        # CLDNN encoder
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', conv=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', conv=True, batch_norm=True)

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
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', residual=True)
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
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='location')
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', attention_type='dot_product')

    @measure_time
    def check(self, encoder_type, decoder_type, bidirectional=False,
              attention_type='location', label_type='char',
              subsample=False, projection=False, init_dec_state='first',
              ctc_loss_weight=0, conv=False, batch_norm=False,
              residual=False, dense_residual=False,
              decoding_order='bahdanau', beam_width=1,
              backward_loss_weight=0, num_heads=1):

        print('==================================================')
        print('  label_type: %s' % label_type)
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  projection: %d' % projection)
        print('  decoder_type: %s' % decoder_type)
        print('  init_dec_state: %s' % init_dec_state)
        print('  attention_type: %s' % attention_type)
        print('  subsample: %s' % str(subsample))
        print('  ctc_loss_weight: %.2f' % ctc_loss_weight)
        print('  conv: %s' % str(conv))
        print('  batch_norm: %s' % str(batch_norm))
        print('  residual: %s' % str(residual))
        print('  dense_residual: %s' % str(dense_residual))
        print('  decoding_order: %s' % decoding_order)
        print('  beam_width: %d' % beam_width)
        print('  backward_loss_weight: %.2f' % backward_loss_weight)
        print('  num_heads: %d' % num_heads)
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
        xs, ys = generate_data(label_type=label_type,
                               batch_size=2 * args.ngpus)

        if label_type == 'char':
            num_classes = 27
            map_fn = idx2char
        elif label_type == 'word':
            num_classes = 11
            map_fn = idx2word

        # Load model
        num_stack = 1 if subsample or conv or encoder_type == 'cnn' else 3
        model = AttentionSeq2seq(
            input_size=xs[0].shape[-1],
            encoder_type=encoder_type,
            encoder_bidirectional=bidirectional,
            encoder_num_units=256,
            encoder_num_proj=256 if projection else 0,
            encoder_num_layers=1 if not subsample else 2,
            attention_type=attention_type,
            attention_dim=128,
            decoder_type=decoder_type,
            decoder_num_units=256,
            decoder_num_layers=1,
            embedding_dim=32,
            dropout_input=0.1,
            dropout_encoder=0.1,
            dropout_decoder=0.1,
            dropout_embedding=0.1,
            num_classes=num_classes,
            parameter_init_distribution='uniform',
            parameter_init=0.1,
            recurrent_weight_orthogonal=False,
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
            num_skip=num_stack,
            splice=1,
            input_channel=3,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            poolings=poolings,
            activation='relu',
            batch_norm=batch_norm,
            scheduled_sampling_prob=0.1,
            scheduled_sampling_max_step=200,
            label_smoothing_prob=0.1,
            weight_noise_std=1e-9,
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
        print("Total %.2f M parameters" % (model.total_parameters / 1000000))

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
                                   decay_type='compare_metric',
                                   decay_start_epoch=20,
                                   decay_rate=0.9,
                                   decay_patient_epoch=10,
                                   lower_better=True)

        # GPU setting
        if args.ngpus >= 1:
            model = CustomDataParallel(
                model, device_ids=list(range(0, args.ngpus, 1)),
                benchmark=True)
            model.cuda()

        # Train model
        max_step = 200
        start_time_step = time.time()
        for step in range(max_step):

            # Step for parameter update
            model.module.optimizer.zero_grad()
            if args.ngpus > 1:
                torch.cuda.empty_cache()
            loss, acc = model(xs, ys)
            if args.ngpus > 1:
                loss.backward(torch.ones(args.ngpus))
            else:
                loss.backward()
            loss.detach()
            if model.module.torch_version < 0.4:
                torch.nn.utils.clip_grad_norm(model.module.parameters(), 5)
                loss = loss.data[0]
            else:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), 5)
                loss = loss.item()
            model.module.optimizer.step()

            # Inject Gaussian noise to all parameters
            if loss < 50:
                model.module.weight_noise_injection = True

            if (step + 1) % 10 == 0:
                # Compute loss
                loss, acc = model(xs, ys, is_eval=True)
                loss = loss.data[0] if model.module.torch_version < 0.4 else loss.item(
                )

                # Decode
                best_hyps, _, perm_idx = model.module.decode(
                    xs, beam_width, max_decode_len=60)

                str_ref = map_fn(ys[0])
                str_hyp = map_fn(best_hyps[0])

                # Compute accuracy
                try:
                    if label_type == 'char':
                        ler = compute_wer(ref=list(str_ref.replace('_', '')),
                                          hyp=list(str_hyp.replace('_', '')),
                                          normalize=True)[0]
                    elif label_type == 'word':
                        ler = compute_wer(ref=str_ref.split('_'),
                                          hyp=str_hyp.split('_'),
                                          normalize=True)[0]
                except:
                    ler = 100

                duration_step = time.time() - start_time_step
                print('Step %d: loss=%.2f/acc=%.2f/ler=%.2f%%/lr=%.5f (%.2f sec)' %
                      (step + 1, loss, acc, ler, learning_rate, duration_step))
                start_time_step = time.time()

                # Visualize
                print('Ref: %s' % str_ref)
                print('Hyp: %s' % str_hyp)

                # Decode by the CTC decoder
                if model.module.ctc_loss_weight >= 0.1:
                    best_hyps_ctc, perm_idx = model.module.decode_ctc(
                        xs, beam_width)
                    str_pred_ctc = map_fn(best_hyps_ctc[0])
                    print('Hyp (CTC): %s' % str_pred_ctc)

                if ler < 5:
                    print('Modle is Converged.')
                    break

                # Update learning rate
                model.module.optimizer, learning_rate = lr_controller.decay_lr(
                    optimizer=model.module.optimizer,
                    learning_rate=learning_rate,
                    epoch=step,
                    value=ler)


if __name__ == "__main__":
    if sys.argv:
        del sys.argv[1:]

    unittest.main()
