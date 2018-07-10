#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test hierarchical attention-besed models (pytorch)."""

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

sys.path.append('../../../../../')
from src.models.pytorch_v3.attention.hierarchical_attention_seq2seq import HierarchicalAttentionSeq2seq
from src.models.pytorch_v3.data_parallel import CustomDataParallel
from src.models.test.data import generate_data, idx2char, idx2word
from src.utils.measure_time_func import measure_time
from src.utils.evaluation.edit_distance import compute_wer
from src.bin.training.utils.learning_rate_controller import Controller

parser = argparse.ArgumentParser()
parser.add_argument('--ngpus', type=int, default=0,
                    help='the number of GPUs (negative value indicates CPU)')
args = parser.parse_args()


class TestHierarchicalAttention(unittest.TestCase):

    def test(self):
        print("Hierarchical Attention Working check.")

        # Sharing attention
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', share_attention=True)

        # Multi-head attention
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', num_heads=2)

        # CNN encoder
        self.check(encoder_type='cnn', decoder_type='lstm', batch_norm=True)

        # Forward word decoder + backward char decoder
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', backward_sub=True)

        # CLDNN encoder
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', conv=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', conv=True, batch_norm=True)

        # Word attention + char CTC
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm', ctc_loss_weight_sub=0.5)

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

        # BLSTM encoder
        self.check(encoder_type='lstm', bidirectional=True,
                   decoder_type='lstm')

    @measure_time
    def check(self, encoder_type, decoder_type, bidirectional=False,
              attention_type='location', subsample=False, projection=False,
              ctc_loss_weight_sub=0, conv=False, batch_norm=False,
              residual=False, dense_residual=False,
              decoding_order='bahdanau', beam_width=1,
              num_heads=1, backward_sub=False, share_attention=True):

        print('==================================================')
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  projection: %d' % projection)
        print('  decoder_type: %s' % decoder_type)
        print('  attention_type: %s' % attention_type)
        print('  subsample: %s' % str(subsample))
        print('  ctc_loss_weight_sub: %.2f' % ctc_loss_weight_sub)
        print('  conv: %s' % str(conv))
        print('  batch_norm: %s' % str(batch_norm))
        print('  residual: %s' % str(residual))
        print('  dense_residual: %s' % str(dense_residual))
        print('  beam_width: %d' % beam_width)
        print('  backward_sub: %s' % str(backward_sub))
        print('  num_heads: %s' % str(num_heads))
        print('  share_attention: %s' % str(share_attention))
        print('==================================================')

        if conv or encoder_type == 'cnn':
            # pattern 1
            conv_channels = [32, 32]
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
        xs, ys, ys_sub = generate_data(label_type='word_char', batch_size=2)

        # Load model
        num_stack = 1 if subsample or conv or encoder_type == 'cnn' else 2
        model = HierarchicalAttentionSeq2seq(
            input_type='speech',
            input_size=xs[0].shape[-1],
            encoder_type=encoder_type,
            encoder_bidirectional=bidirectional,
            encoder_num_units=256,
            encoder_num_proj=256 if projection else 0,
            encoder_num_layers=2,
            encoder_num_layers_sub=1,
            attention_type=attention_type,
            attention_dim=128,
            decoder_type=decoder_type,
            decoder_num_units=256,
            decoder_num_layers=1,
            decoder_num_units_sub=256,
            decoder_num_layers_sub=1,
            embedding_dim=64,
            embedding_dim_sub=32,
            dropout_input=0.1,
            dropout_encoder=0.1,
            dropout_decoder=0.1,
            dropout_embedding=0.1,
            main_loss_weight=0.5,
            sub_loss_weight=0.5 if ctc_loss_weight_sub == 0 else 0,
            num_classes=11,
            num_classes_sub=27,
            parameter_init_distribution='uniform',
            parameter_init=0.1,
            recurrent_weight_orthogonal=False,
            init_forget_gate_bias_with_one=True,
            subsample_list=[] if not subsample else [True, False],
            subsample_type='drop' if subsample is False else subsample,
            bridge_layer=True,
            init_dec_state='first',
            sharpening_factor=1,
            logits_temperature=1,
            sigmoid_smoothing=False,
            ctc_loss_weight_sub=ctc_loss_weight_sub,
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
            weight_noise_std=0,
            encoder_residual=residual,
            encoder_dense_residual=dense_residual,
            decoder_residual=residual,
            decoder_dense_residual=dense_residual,
            decoding_order='bahdanau',
            bottleneck_dim=256,
            bottleneck_dim_sub=256,
            backward_sub=backward_sub,
            num_heads=num_heads,
            num_heads_sub=num_heads,
            share_attention=share_attention)

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
        max_step = 300
        start_time_step = time.time()
        for step in range(max_step):

            # Step for parameter update
            model.module.optimizer.zero_grad()
            if args.ngpus > 1:
                torch.cuda.empty_cache()
            loss, loss_main, loss_sub, acc_main, acc_sub = model(
                xs, ys, ys_sub)
            if args.ngpus > 1:
                loss.backward(torch.ones(args.ngpus))
            else:
                loss.backward()
            loss.detach()
            if model.module.torch_version < 0.4:
                torch.nn.utils.clip_grad_norm(model.module.parameters(), 5)
            else:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), 5)
            model.module.optimizer.step()

            if (step + 1) % 10 == 0:
                # Compute loss
                loss, loss_main, loss_sub, acc_main, acc_sub = model(
                    xs, ys, ys_sub, is_eval=True)
                loss = loss.data[0] if model.module.torch_version < 0.4 else loss.item(
                )

                # Decode
                best_hyps, _, perm_idx = model.module.decode(
                    xs, beam_width, max_decode_len=30)
                best_hyps_sub, _, _ = model.module.decode(
                    xs, beam_width, max_decode_len=60, task_index=1)

                str_hyp = idx2word(best_hyps[0][:-1])
                str_ref = idx2word(ys[0])
                str_hyp_sub = idx2char(best_hyps_sub[0][:-1])
                str_ref_sub = idx2char(ys_sub[0])

                # Compute accuracy
                try:
                    wer = compute_wer(ref=str_ref.split('_'),
                                      hyp=str_hyp.split('_'),
                                      normalize=True)[0]
                    cer = compute_wer(ref=list(str_ref_sub.replace('_', '')),
                                      hyp=list(str_hyp_sub.replace('_', '')),
                                      normalize=True)[0]
                except:
                    wer = 100
                    cer = 100

                duration_step = time.time() - start_time_step
                print('Step %d: loss=%.2f(%.2f/%.2f)/acc=%.2f/%.2f/wer=%.2f%%/cer=%.2f%%/lr=%.5f (%.2f sec)' %
                      (step + 1, loss, loss_main, loss_sub, acc_main, acc_sub,
                       wer, cer, learning_rate, duration_step))
                start_time_step = time.time()

                # Visualize
                print('Ref: %s' % str_ref)
                print('Hyp (word): %s' % str_hyp)
                print('Hyp (char): %s' % str_hyp_sub)

                if cer < 1:
                    print('Modle is Converged.')
                    break

                # Update learning rate
                model.module.optimizer, learning_rate = lr_controller.decay_lr(
                    optimizer=model.module.optimizer,
                    learning_rate=learning_rate,
                    epoch=step,
                    value=wer)


if __name__ == "__main__":
    if sys.argv:
        del sys.argv[1:]

    unittest.main()
