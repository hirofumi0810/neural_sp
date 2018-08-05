#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Test attention-besed models (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import unittest

import torch

sys.path.append('../../../../')
from src.bin.training.utils.learning_rate_controller import Controller
from src.models.pytorch_v3.attention.attention_seq2seq import AttentionSeq2seq
from src.models.pytorch_v3.data_parallel import CustomDataParallel
from src.test.data import generate_data
from src.test.data import idx2char
from src.test.data import idx2word
from src.utils.evaluation.edit_distance import compute_wer
from src.utils.measure_time_func import measure_time

parser = argparse.ArgumentParser()
parser.add_argument('--ngpus', type=int, default=0,
                    help='the number of GPUs (negative value indicates CPU)')
args = parser.parse_args()

torch.manual_seed(1623)
torch.cuda.manual_seed_all(1623)


class TestAttention(unittest.TestCase):

    def test(self):
        print("Attention Working check.")

        # Internal LM
        self.check(internal_lm=True)

        # RNNLM objective
        self.check(internal_lm=True, lm_weight=1)
        self.check(internal_lm=True, lm_weight=1, share_softmax=True)

        # Multi-head attention
        self.check(att_type='content', n_heads=2)
        self.check(att_type='location', n_heads=2)
        self.check(att_type='dot_product', n_heads=2)
        self.check(att_type='location', n_heads=2, beam_width=2)

        # CNN encoder
        self.check(enc_type='cnn', batch_norm=True)

        # Beam search
        self.check(beam_width=2)

        # Backward decoder
        self.check(bwd_weight=1)
        self.check(bwd_weight=0.8)
        self.check(bwd_weight=0.5)
        self.check(bwd_weight=0.2)
        self.check(bwd_weight=0.8, beam_width=2)

        # CLDNN encoder
        self.check(conv=True)
        self.check(conv=True, batch_norm=True)

        # Joint CTC-Attention
        self.check(ctc_weight=0.2)

        # Pyramidal encoder
        self.check(subsample='drop')
        self.check(subsample='concat')

        # Projection layer
        self.check(enc_proj=True)

        # Residual connection
        self.check(residual=True)

        # word-level attention
        self.check(label_type='word')
        # self.check(label_type='phone')

        # unidirectional & enc_bidirectional
        self.check(enc_bidirectional=True)
        self.check(enc_bidirectional=False)
        self.check(enc_type='gru', dec_type='gru')
        self.check(enc_type='gru', enc_bidirectional=False, dec_type='gru')

        # Attention type
        self.check(att_type='content')
        self.check(att_type='location')
        self.check(att_type='dot_product')

    @measure_time
    def check(self, enc_type='lstm', dec_type='lstm', enc_bidirectional=True,
              att_type='location', label_type='char',
              subsample=False, enc_proj=False, ctc_weight=0,
              conv=False, batch_norm=False, residual=False,
              beam_width=1, bwd_weight=0, n_heads=1,
              internal_lm=False, lm_weight=0, share_softmax=False):

        print('==================================================')
        print('  label_type: %s' % label_type)
        print('  enc_type: %s' % enc_type)
        print('  enc_bidirectional: %s' % str(enc_bidirectional))
        print('  enc_proj: %d' % enc_proj)
        print('  dec_type: %s' % dec_type)
        print('  att_type: %s' % att_type)
        print('  subsample: %s' % str(subsample))
        print('  ctc_weight: %.2f' % ctc_weight)
        print('  conv: %s' % str(conv))
        print('  batch_norm: %s' % str(batch_norm))
        print('  residual: %s' % str(residual))
        print('  beam_width: %d' % beam_width)
        print('  bwd_weight: %.2f' % bwd_weight)
        print('  n_heads: %d' % n_heads)
        print('  internal_lm: %s' % str(internal_lm))
        print('  lm_weight: %.2f' % lm_weight)
        print('  share_softmax: %s' % str(share_softmax))
        print('==================================================')

        if conv or enc_type == 'cnn':
            # pattern 1
            # conv_channels = [32, 32]
            # conv_kernel_sizes = [[41, 11], [21, 11]]
            # conv_strides = [[2, 2], [2, 1]]
            # conv_poolings = [[], []]

            # pattern 2 (VGG like)
            conv_channels = [64, 64]
            conv_kernel_sizes = [[3, 3], [3, 3]]
            conv_strides = [[1, 1], [1, 1]]
            conv_poolings = [[2, 2], [2, 2]]
        else:
            conv_channels = []
            conv_kernel_sizes = []
            conv_strides = []
            conv_poolings = []

        # Load batch data
        xs, ys = generate_data(label_type=label_type,
                               batch_size=2 * args.ngpus)

        if label_type == 'char':
            n_classes = 27
            map_fn = idx2char
        elif label_type == 'word':
            n_classes = 11
            map_fn = idx2word

        # Load model
        n_stack = 1 if subsample or conv or enc_type == 'cnn' else 3
        model = AttentionSeq2seq(
            enc_in_type='speech',
            enc_in_size=xs[0].shape[-1],
            n_stack=n_stack,
            n_skip=n_stack,
            n_splice=1,
            conv_in_channel=3,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            conv_poolings=conv_poolings,
            conv_batch_norm=batch_norm,
            enc_type=enc_type,
            enc_bidirectional=enc_bidirectional,
            enc_n_units=256,
            enc_n_projs=256 if enc_proj else 0,
            enc_n_layers=1 if not subsample else 2,
            enc_residual=residual,
            subsample_list=[] if not subsample else [True, False],
            subsample_type='concat' if not subsample else subsample,
            att_type=att_type,
            att_dim=128,
            att_conv_n_channels=10,
            att_conv_width=201,
            att_n_heads=n_heads,
            sharpening_factor=1,
            sigmoid_smoothing=False,
            bridge_layer=True,
            dec_type=dec_type,
            dec_n_units=256,
            dec_n_layers=1,
            dec_residual=residual,
            emb_dim=256,
            generate_feat='sc',
            n_classes=n_classes,
            logits_temp=1,
            param_init_dist='uniform',
            param_init=0.1,
            rec_weight_orthogonal=False,
            dropout_in=0.1,
            dropout_enc=0.1,
            dropout_dec=0.1,
            dropout_emb=0.1,
            ss_prob=0.1,
            lsm_prob=0.1,
            lsm_type='uniform',
            ctc_weight=ctc_weight,
            bwd_weight=bwd_weight,
            internal_lm=internal_lm,
            lm_weight=lm_weight,
            share_softmax=share_softmax,
        )

        # Count total parameters
        for name in sorted(list(model.num_params_dict.keys())):
            n_params = model.num_params_dict[name]
            print("%s %d" % (name, n_params))
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
            model = CustomDataParallel(model,
                                       device_ids=list(range(0, args.ngpus, 1)),
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
                loss = loss.data[0] if model.module.torch_version < 0.4 else loss.item()

                # Decode
                best_hyps, _, perm_idx = model.module.decode(xs, beam_width, max_len_ratio=0.7)

                str_ref = map_fn(ys[0])
                str_hyp = map_fn(best_hyps[0]).replace('_>', '').replace('>', '')

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
                if ctc_weight >= 0.1:
                    best_hyps_ctc, perm_idx = model.module.decode_ctc(xs, beam_width)
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
