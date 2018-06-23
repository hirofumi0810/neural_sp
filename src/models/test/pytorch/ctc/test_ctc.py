#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test CTC models (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import unittest

import torch
torch.manual_seed(1623)
torch.cuda.manual_seed_all(1623)

sys.path.append('../../../../../')
# from models.pytorch.ctc.ctc import CTC
from src.models.pytorch_v3.ctc.ctc import CTC
from src.models.test.data import generate_data, idx2char, idx2word
from src.utils.measure_time_func import measure_time
from src.utils.evaluation.edit_distance import compute_wer
from src.bin.training.utils.learning_rate_controller import Controller


class TestCTC(unittest.TestCase):

    def test(self):
        print("CTC Working check.")

        self.check(encoder_type='lstm', bidirectional=True, subsample=True)

        # TODO: CNN-CTC
        self.check(encoder_type='cnn', batch_norm=True, activation='relu')
        self.check(encoder_type='cnn', batch_norm=True, activation='prelu')
        self.check(encoder_type='cnn', batch_norm=True, activation='hard_tanh')
        # self.check(encoder_type='cnn', batch_norm=True, activation='maxout') # TODO

        # Beam search
        self.check(encoder_type='lstm', bidirectional=True, beam_width=2)

        # CLDNN-CTC
        self.check(encoder_type='lstm', bidirectional=True,
                   conv=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   conv=True, batch_norm=True)

        # TODO: Label smoothing
        # self.check(encoder_type='lstm', bidirectional=True,
        #            label_smoothing=True)

        # Pyramidal encoder
        self.check(encoder_type='lstm', bidirectional=True, subsample=True)

        # Projection layer
        self.check(encoder_type='lstm', bidirectional=True, projection=True)
        self.check(encoder_type='lstm', bidirectional=False, projection=True)

        # Residual LSTM-CTC
        self.check(encoder_type='lstm', bidirectional=True,
                   encoder_residual=True)
        self.check(encoder_type='lstm', bidirectional=True,
                   encoder_dense_residual=True)

        # word-level CTC
        self.check(encoder_type='lstm', bidirectional=True,
                   label_type='word')

        # RNNs
        self.check(encoder_type='lstm', bidirectional=True)
        self.check(encoder_type='lstm', bidirectional=False)
        self.check(encoder_type='gru', bidirectional=True)
        self.check(encoder_type='gru', bidirectional=False)

    @measure_time
    def check(self, encoder_type, bidirectional=False, label_type='char',
              subsample=False,  projection=False,
              conv=False, batch_norm=False, activation='relu',
              encoder_residual=False, encoder_dense_residual=False,
              label_smoothing=False, beam_width=1):

        print('==================================================')
        print('  label_type: %s' % label_type)
        print('  encoder_type: %s' % encoder_type)
        print('  bidirectional: %s' % str(bidirectional))
        print('  projection: %s' % str(projection))
        print('  subsample: %s' % str(subsample))
        print('  conv: %s' % str(conv))
        print('  batch_norm: %s' % str(batch_norm))
        print('  activation: %s' % activation)
        print('  encoder_residual: %s' % str(encoder_residual))
        print('  encoder_dense_residual: %s' % str(encoder_dense_residual))
        print('  label_smoothing: %s' % str(label_smoothing))
        print('  beam_width: %d' % beam_width)
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

            fc_list = [786, 786]
        else:
            conv_channels = []
            conv_kernel_sizes = []
            conv_strides = []
            poolings = []
            fc_list = []

        # Load batch data
        xs, ys = generate_data(label_type=label_type, batch_size=2)

        if label_type == 'char':
            num_classes = 27
            map_fn = idx2char
        elif label_type == 'word':
            num_classes = 11
            map_fn = idx2word

        # Load model
        num_stack = 1 if subsample or conv or encoder_type == 'cnn' else 2
        model = CTC(
            input_size=xs[0].shape[-1],
            encoder_type=encoder_type,
            encoder_bidirectional=bidirectional,
            encoder_num_units=256,
            encoder_num_proj=256 if projection else 0,
            encoder_num_layers=1 if not subsample else 2,
            fc_list=fc_list,
            dropout_input=0.1,
            dropout_encoder=0.1,
            num_classes=num_classes,
            parameter_init_distribution='uniform',
            parameter_init=0.1,
            recurrent_weight_orthogonal=False,
            init_forget_gate_bias_with_one=True,
            subsample_list=[] if not subsample else [True] * 2,
            num_stack=num_stack,
            num_skip=num_stack,
            splice=1,
            input_channel=3,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            poolings=poolings,
            activation=activation,
            batch_norm=batch_norm,
            label_smoothing_prob=0.1 if label_smoothing else 0,
            weight_noise_std=0,
            encoder_residual=encoder_residual,
            encoder_dense_residual=encoder_dense_residual)

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
        model.set_cuda(deterministic=False, benchmark=True)

        # Train model
        max_step = 300
        start_time_step = time.time()
        for step in range(max_step):

            # Step for parameter update
            model.optimizer.zero_grad()
            if model.device_id >= 0:
                torch.cuda.empty_cache()
            loss, _ = model(xs, ys)
            loss.backward()
            loss.detach()
            if model.torch_version < 0.4:
                torch.nn.utils.clip_grad_norm(model.parameters(), 5)
                loss = loss.data[0]
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                loss = loss.item()
            model.optimizer.step()

            # Inject Gaussian noise to all parameters
            if loss < 50:
                model.weight_noise_injection = True

            if (step + 1) % 10 == 0:
                # Compute loss
                loss, _ = model(xs, ys, is_eval=True)

                # Decode
                best_hyps, _, _ = model.decode(xs, beam_width)

                str_ref = map_fn(ys[0])
                str_hyp = map_fn(best_hyps[0])

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
                    ler = 100

                duration_step = time.time() - start_time_step
                print('Step %d: loss=%.2f/ler=%.2f%%/lr=%.5f (%.2f sec)' %
                      (step + 1, loss.data, ler, learning_rate, duration_step))
                start_time_step = time.time()

                # Visualize
                print('Ref: %s' % str_ref)
                print('Hyp: %s' % str_hyp)

                if ler < 5:
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
