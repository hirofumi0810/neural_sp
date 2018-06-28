#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test attention-besed phone-to-word models (pytorch)."""

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
from src.models.test.data import generate_data_p2w, idx2char, idx2word
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

        self.check(label_type_in='phone', label_type_out='word')

    @measure_time
    def check(self, label_type_in, label_type_out):

        print('==================================================')
        print('  label_type_in: %s' % label_type_in)
        print('  label_type_out: %s' % label_type_out)
        print('==================================================')

        # Load batch data
        xs, ys = generate_data_p2w(label_type_in=label_type_in,
                                   batch_size=2 * args.ngpus)

        if label_type_in == 'char':
            num_classes_in = 27
        elif label_type_in == 'phone':
            num_classes_in = 27

        if label_type_out == 'char':
            num_classes_out = 27
            map_fn_out = idx2char
        elif label_type_out == 'word':
            num_classes_out = 11
            map_fn_out = idx2word

        # Load model
        model = AttentionSeq2seq(
            input_type='text',
            input_size=32,
            encoder_type='lstm',
            encoder_bidirectional=True,
            encoder_num_units=256,
            encoder_num_proj=0,
            encoder_num_layers=2,
            attention_type='content',
            attention_dim=128,
            decoder_type='lstm',
            decoder_num_units=256,
            decoder_num_layers=1,
            embedding_dim=32,
            dropout_input=0.1,
            dropout_encoder=0.1,
            dropout_decoder=0.1,
            dropout_embedding=0.1,
            num_classes=num_classes_out,
            num_classes_input=num_classes_in)

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
                    xs, beam_width=1, max_decode_len=60)

                str_ref = map_fn_out(ys[0])
                str_hyp = map_fn_out(best_hyps[0]).replace(
                    '_>', '').replace('>', '')

                # Compute accuracy
                try:
                    if label_type_out == 'char':
                        ler = compute_wer(ref=list(str_ref.replace('_', '')),
                                          hyp=list(str_hyp.replace('_', '')),
                                          normalize=True)[0]
                    elif label_type_out == 'word':
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
