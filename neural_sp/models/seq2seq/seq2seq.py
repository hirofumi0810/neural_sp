#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""General sequence-to-sequence model (including CTC)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import six
from torch.autograd import Variable

from neural_sp.models.base import ModelBase
from neural_sp.models.linear import Embedding
from neural_sp.models.linear import LinearND
from neural_sp.models.rnnlm.rnnlm import RNNLM
from neural_sp.models.seq2seq.attention.attention import AttentionMechanism
from neural_sp.models.seq2seq.attention.multihead_attention import MultiheadAttentionMechanism
from neural_sp.models.seq2seq.decoders.decoder import Decoder
from neural_sp.models.seq2seq.encoders.cnn import CNNEncoder
from neural_sp.models.seq2seq.encoders.frame_stacking import stack_frame
from neural_sp.models.seq2seq.encoders.rnn import RNNEncoder
from neural_sp.models.seq2seq.encoders.splicing import do_splice
from neural_sp.models.utils import np2var
from neural_sp.models.utils import pad_list


class Seq2seq(ModelBase):
    """Attention-based sequence-to-sequence model (including CTC)."""

    def __init__(self, args):

        super(ModelBase, self).__init__()

        # for encoder
        self.input_type = args.input_type
        assert args.input_type in ['speech', 'text']
        self.input_dim = args.input_dim
        self.num_stack = args.num_stack
        self.num_skip = args.num_skip
        self.num_splice = args.num_splice
        self.enc_type = args.enc_type
        self.enc_num_units = args.enc_num_units
        if args.enc_type in ['blstm', 'bgru']:
            self.enc_num_units *= 2

        # for attention layer
        self.att_num_heads_0 = args.att_num_heads
        self.att_num_heads_1 = args.att_num_heads_sub
        self.share_attention = False

        # for decoder
        self.num_classes = args.num_classes
        self.num_classes_sub = args.num_classes_sub
        self.blank = 0
        self.unk = 1
        self.sos = 2
        self.eos = 3
        self.pad = 4
        # NOTE: these are reserved in advance

        # for CTC
        self.ctc_weight_0 = args.ctc_weight
        self.ctc_weight_1 = args.ctc_weight_sub

        # for backward decoder
        assert 0 <= args.bwd_weight <= 1
        assert 0 <= args.bwd_weight_sub <= 1
        self.fwd_weight_0 = 1 - args.bwd_weight
        self.bwd_weight_0 = args.bwd_weight
        self.fwd_weight_1 = 1 - args.bwd_weight
        self.bwd_weight_1 = args.bwd_weight

        # for the sub task
        self.main_task_weight = args.main_task_weight

        # Encoder
        if args.enc_type in ['blstm', 'lstm', 'bgru', 'gru']:
            self.enc = RNNEncoder(input_dim=args.input_dim if args.input_type == 'speech' else args.emb_dim,
                                  rnn_type=args.enc_type,
                                  num_units=args.enc_num_units,
                                  num_projs=args.enc_num_projs,
                                  num_layers=args.enc_num_layers,
                                  num_layers_sub=args.enc_num_layers_sub,
                                  dropout_in=args.dropout_in,
                                  dropout_hidden=args.dropout_enc,
                                  subsample=args.subsample,
                                  subsample_type=args.subsample_type,
                                  batch_first=True,
                                  num_stack=args.num_stack,
                                  num_splice=args.num_splice,
                                  conv_in_channel=args.conv_in_channel,
                                  conv_channels=args.conv_channels,
                                  conv_kernel_sizes=args.conv_kernel_sizes,
                                  conv_strides=args.conv_strides,
                                  conv_poolings=args.conv_poolings,
                                  conv_batch_norm=args.conv_batch_norm,
                                  residual=args.enc_residual,
                                  nin=0,
                                  num_projs_final=args.dec_num_units if args.bridge_layer else 0)
        elif args.enc_type == 'cnn':
            assert args.num_stack == 1 and args.num_splice == 1
            self.enc = CNNEncoder(input_dim=args.input_dim if args.input_type == 'speech' else args.emb_dim,
                                  in_channel=args.conv_in_channel,
                                  channels=args.conv_channels,
                                  kernel_sizes=args.conv_kernel_sizes,
                                  strides=args.conv_strides,
                                  poolings=args.conv_poolings,
                                  dropout_in=args.dropout_in,
                                  dropout_hidden=args.dropout_enc,
                                  num_projs_final=args.dec_num_units,
                                  batch_norm=args.conv_batch_norm)
        else:
            raise NotImplementedError()

        # Bridge layer between the encoder and decoder
        if args.enc_type == 'cnn':
            self.enc_num_units = args.dec_num_units
        elif args.bridge_layer:
            self.bridge_0 = LinearND(self.enc_num_units, args.dec_num_units)
            self.enc_num_units = args.dec_num_units
        else:
            self.bridge_0 = lambda x: x

        directions = []
        if self.fwd_weight_0 > 0:
            directions.append('fwd')
        if self.bwd_weight_0 > 0:
            directions.append('bwd')
        for dir in directions:
            if args.ctc_weight < 1:
                # Attention layer
                if args.att_num_heads > 1:
                    attention = MultiheadAttentionMechanism(
                        enc_num_units=self.enc_num_units,
                        dec_num_units=args.dec_num_units,
                        att_type=args.att_type,
                        att_dim=args.att_dim,
                        sharpening_factor=args.att_sharpening_factor,
                        sigmoid_smoothing=args.att_sigmoid_smoothing,
                        conv_out_channels=args.att_conv_num_channels,
                        conv_kernel_size=args.att_conv_width,
                        num_heads=args.att_num_heads)
                else:
                    attention = AttentionMechanism(
                        enc_num_units=self.enc_num_units,
                        dec_num_units=args.dec_num_units,
                        att_type=args.att_type,
                        att_dim=args.att_dim,
                        sharpening_factor=args.att_sharpening_factor,
                        sigmoid_smoothing=args.att_sigmoid_smoothing,
                        conv_out_channels=args.att_conv_num_channels,
                        conv_kernel_size=args.att_conv_width)

                # Cold fusion
                # if args.rnnlm_cf is not None and dir == 'fwd':
                #     raise NotImplementedError()
                #     # TODO(hirofumi): cold fusion for backward RNNLM
                # else:
                #     args.rnnlm_cf = None
                #
                # # RNNLM initialization
                # if args.rnnlm_config_init is not None and dir == 'fwd':
                #     raise NotImplementedError()
                #     # TODO(hirofumi): RNNLM initialization for backward RNNLM
                # else:
                #     args.rnnlm_init = None
            else:
                attention = None

            # Decoder
            decoder = Decoder(attention=attention,
                              sos=self.sos,
                              eos=self.eos,
                              pad=self.pad,
                              enc_num_units=self.enc_num_units,
                              rnn_type=args.dec_type,
                              num_units=args.dec_num_units,
                              num_layers=args.dec_num_layers,
                              residual=args.dec_residual,
                              emb_dim=args.emb_dim,
                              num_classes=self.num_classes,
                              logits_temp=args.logits_temp,
                              dropout_dec=args.dropout_dec,
                              dropout_emb=args.dropout_emb,
                              ss_prob=args.ss_prob,
                              lsm_prob=args.lsm_prob,
                              lsm_type=args.lsm_type,
                              init_with_enc=args.init_with_enc,
                              ctc_weight=args.ctc_weight if dir == 'fwd' else 0,
                              ctc_fc_list=args.ctc_fc_list,
                              backward=(dir == 'bwd'),
                              rnnlm_cf=args.rnnlm_cf,
                              cold_fusion_type=args.cold_fusion_type,
                              internal_lm=args.internal_lm,
                              rnnlm_init=args.rnnlm_init,
                              # rnnlm_weight=args.rnnlm_weight,
                              share_softmax=args.share_softmax)
            setattr(self, 'dec_' + dir + '_0', decoder)

        # NOTE: fwd only for the sub task
        if args.main_task_weight < 1:
            if args.ctc_weight_sub < 1:
                # Attention layer
                if args.att_num_heads_sub > 1:
                    attention_sub = MultiheadAttentionMechanism(
                        enc_num_units=self.enc_num_units,
                        dec_num_units=args.dec_num_units,
                        att_type=args.att_type,
                        att_dim=args.att_dim,
                        sharpening_factor=args.att_sharpening_factor,
                        sigmoid_smoothing=args.att_sigmoid_smoothing,
                        conv_out_channels=args.att_conv_num_channels,
                        conv_kernel_size=args.att_conv_width,
                        num_heads=args.att_num_heads_sub)
                else:
                    attention_sub = AttentionMechanism(
                        enc_num_units=self.enc_num_units,
                        dec_num_units=args.dec_num_units,
                        att_type=args.att_type,
                        att_dim=args.att_dim,
                        sharpening_factor=args.att_sharpening_factor,
                        sigmoid_smoothing=args.att_sigmoid_smoothing,
                        conv_out_channels=args.att_conv_num_channels,
                        conv_kernel_size=args.att_conv_width)
            else:
                attention_sub = None

            # Decoder
            self.dec_fwd_1 = Decoder(attention=attention_sub,
                                     sos=self.sos,
                                     eos=self.eos,
                                     pad=self.pad,
                                     enc_num_units=self.enc_num_units,
                                     rnn_type=args.dec_type,
                                     num_units=args.dec_num_units,
                                     num_layers=args.dec_num_layers,
                                     residual=args.dec_residual,
                                     emb_dim=args.emb_dim,
                                     num_classes=self.num_classes_sub,
                                     logits_temp=args.logits_temp,
                                     dropout_dec=args.dropout_dec,
                                     dropout_emb=args.dropout_emb,
                                     ss_prob=args.ss_prob,
                                     lsm_prob=args.lsm_prob,
                                     lsm_type=args.lsm_type,
                                     init_with_enc=args.init_with_enc,
                                     ctc_weight=args.ctc_weight_sub,
                                     ctc_fc_list=args.ctc_fc_list)  # sub??

        if args.input_type == 'text':
            if args.num_classes == args.num_classes_sub:
                # Share the embedding layer between input and output
                self.embed_in = decoder.emb
            else:
                self.embed_in = Embedding(num_classes=args.num_classes_sub,
                                          emb_dim=args.emb_dim,
                                          dropout=args.dropout_emb,
                                          ignore_index=self.pad)

        # Initialize weight matrices
        self.init_weights(args.param_init, dist=args.param_init_dist, ignore_keys=['bias'])

        # Initialize all biases with 0
        self.init_weights(0, dist='constant', keys=['bias'])

        # Recurrent weights are orthogonalized
        if args.rec_weight_orthogonal:
            # encoder
            if args.enc_type != 'cnn':
                self.init_weights(args.param_init, dist='orthogonal',
                                  keys=[args.enc_type, 'weight'], ignore_keys=['bias'])
            # TODO(hirofumi): in case of CNN + LSTM
            # decoder
            self.init_weights(args.param_init, dist='orthogonal',
                              keys=[args.dec_type, 'weight'], ignore_keys=['bias'])

        # Initialize bias in forget gate with 1
        self.init_forget_gate_bias_with_one()

        # Initialize bias in gating with -1
        if args.rnnlm_cf is not None:
            self.init_weights(-1, dist='constant', keys=['cf_fc_lm_gate.fc.bias'])

    def forward(self, xs, ys, ys_sub=None, is_eval=False):
        """Forward computation.

        Args:
            xs (list): A list of length `[B]`, which contains arrays of size `[T, input_dim]`
            ys (list): A list of length `[B]`, which contains arrays of size `[L]`
            ys_sub (list): A list of lenght `[B]`, which contains arrays of size `[L_sub]`
            is_eval (bool): the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (torch.autograd.Variable(float)): A tensor of size `[1]`
            acc (float): Token-level accuracy in teacher-forcing

        """
        if is_eval:
            self.eval()
        else:
            self.train()

        # Encode input features
        if self.input_type == 'speech':
            # Sort by lenghts in the descending order
            # if self.enc_type != 'cnn':
            perm_idx = sorted(list(six.moves.range(0, len(xs), 1)),
                              key=lambda i: len(xs[i]), reverse=True)
            xs = [xs[i] for i in perm_idx]
            ys = [ys[i] for i in perm_idx]
            # NOTE: must be descending order for pack_padded_sequence
            xs, x_lens, xs_sub, x_lens_sub = self._encode(xs)
        else:
            # Sort by lenghts in the descending order
            perm_idx = sorted(list(six.moves.range(0, len(ys_sub), 1)),
                              key=lambda i: len(ys_sub[i]), reverse=True)
            ys = [ys[i] for i in perm_idx]
            ys_sub = [ys_sub[i] for i in perm_idx]
            # NOTE: must be descending order for pack_padded_sequence
            xs, x_lens, xs_sub, x_lens_sub = self._encode(ys_sub)

        # Compute XE loss for the forward decoder
        if self.fwd_weight_0 > 0:
            ys_fwd = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long() for y in ys]
            loss_acc_fwd = self.dec_fwd_0(xs, x_lens, ys_fwd)
            loss = loss_acc_fwd['loss'] * self.fwd_weight_0
        else:
            loss_acc_fwd = {}
            loss = Variable(xs.data.new(1,).fill_(0.))

        # Compute XE loss for the backward decoder
        if self.bwd_weight_0 > 0:
            ys_bwd = [np2var(np.fromiter(y[::-1], dtype=np.int64), self.device_id).long() for y in ys]
            loss_acc_bwd = self.dec_bwd_0(xs, x_lens, ys_bwd)
            loss += loss_acc_bwd['loss'] * self.bwd_weight_0
        else:
            loss_acc_bwd = {}

        if self.main_task_weight < 1:
            ys_sub = [ys_sub[i] for i in perm_idx]
            ys_sub = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long()
                      for y in ys_sub]
            loss_acc_sub = self.dec_fwd_1(xs_sub, x_lens_sub, ys_sub)
            loss = loss * self.main_task_weight + loss_acc_sub['loss'] * (1 - self.main_task_weight)
        else:
            loss_acc_sub = {}

        # TODO(hirofumi): report here

        return loss, loss_acc_fwd, loss_acc_bwd, loss_acc_sub

    def _encode(self, xs):
        """Encode acoustic features.

        Args:
            xs (list): A list of length `[B]`, which contains Variables of size `[T, input_dim]`
        Returns:
            xs (torch.autograd.Variable, float): A tensor of size
                `[B, T, enc_num_units]`
            x_lens (list): A tensor of size `[B]`
            OPTION:
                xs_sub (torch.autograd.Variable, float): A tensor of size
                    `[B, T, enc_num_units]`
                x_lens_sub (list): A tensor of size `[B]`

        """
        if self.input_type == 'speech':
            # Frame stacking
            if self.num_stack > 1:
                xs = [stack_frame(x, self.num_stack, self.num_skip)for x in xs]

            # Splicing
            if self.num_splice > 1:
                xs = [do_splice(x, self.num_splice, self.num_stack) for x in xs]

            x_lens = [len(x) for x in xs]
            xs = [np2var(x, self.device_id).float() for x in xs]
            xs = pad_list(xs)

        elif self.input_type == 'text':
            x_lens = [len(x) for x in xs]
            xs = [np2var(np.fromiter(x, dtype=np.int64), self.device_id).long() for x in xs]
            xs = pad_list(xs, self.pad)
            xs = self.embed_in(xs)

        if self.main_task_weight < 1:
            if self.enc_type == 'cnn':
                xs, x_lens = self.enc(xs, x_lens)
                xs_sub = xs.clone()
                x_lens_sub = copy.deepcopy(x_lens)
            else:
                xs, x_lens, xs_sub, x_lens_sub = self.enc(xs, x_lens)
        else:
            xs, x_lens = self.enc(xs, x_lens)
            xs_sub = None
            x_lens_sub = None

        # Bridge between the encoder and decoder in the main task
        xs = self.bridge_0(xs)
        # if self.main_task_weight < 1:
        #     xs_sub = self.bridge_1(xs_sub)
        # TODO:

        return xs, x_lens, xs_sub, x_lens_sub

    def decode(self, xs, decode_params, exclude_eos=False, task_index=0):
        """Decoding in the inference stage.

        Args:
            xs (list): A list of length `[B]`, which contains arrays of size `[T, input_dim]`
            decode_params (dict):
                beam_width (int): the size of beam
                min_len_ratio (float):
                max_len_ratio (float):
                len_penalty (float): length penalty
                cov_penalty (float): coverage penalty
                cov_threshold (float): threshold for coverage penalty
                rnnlm_weight (float): the weight of RNNLM score
                resolving_unk (bool): not used (to make compatible)

            exclude_eos (bool): exclude <eos> from best_hyps
            task_index (int): not used (to make compatible)
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`
            perm_idx (list): A list of length `[B]`

        """
        self.eval()

        # Sort by lenghts in the descending order
        # if self.enc_type != 'cnn':
        perm_idx = sorted(list(six.moves.range(0, len(xs), 1)),
                          key=lambda i: len(xs[i]), reverse=True)
        xs = [xs[i] for i in perm_idx]
        # NOTE: must be descending order for pack_padded_sequence

        # Encode input features
        enc_out, x_lens, enc_out_sub, x_lens_sub = self._encode(xs)

        dir = 'fwd' if self.fwd_weight_0 >= self.bwd_weight_0 else 'bwd'

        if self.ctc_weight_0 == 1:
            best_hyps = getattr(self, 'dec_' + dir + '_0').decode_ctc(
                enc_out, x_lens, decode_params['beam_width'], decode_params['rnnlm'])
            return best_hyps, None, perm_idx
        else:
            if decode_params['beam_width'] == 1:
                best_hyps, aw = getattr(self, 'dec_' + dir + '_0').greedy(
                    enc_out, x_lens, decode_params['max_len_ratio'], exclude_eos)
            else:
                # Set RNNLM
                if decode_params['rnnlm_weight'] > 0:
                    assert hasattr(self, 'rnnlm_' + dir + '_0')
                    rnnlm = getattr(self, 'rnnlm_' + dir + '_0')
                else:
                    rnnlm = None

                best_hyps, aw = getattr(self, 'dec_' + dir + '_0').beam_search(
                    enc_out, x_lens, decode_params, rnnlm, exclude_eos)
            return best_hyps, aw, perm_idx
