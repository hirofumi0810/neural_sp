#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""General sequence-to-sequence model (including CTC)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import numpy as np
import six
from torch.autograd import Variable

from neural_sp.models.base import ModelBase
from neural_sp.models.linear import Embedding
from neural_sp.models.linear import LinearND
from neural_sp.models.rnnlm.rnnlm import RNNLM
from neural_sp.models.seq2seq.decoders.attention import AttentionMechanism
from neural_sp.models.seq2seq.decoders.decoder import Decoder
from neural_sp.models.seq2seq.decoders.multihead_attention import MultiheadAttentionMechanism
from neural_sp.models.seq2seq.encoders.cnn import CNNEncoder
from neural_sp.models.seq2seq.encoders.frame_stacking import stack_frame
from neural_sp.models.seq2seq.encoders.rnn import RNNEncoder
from neural_sp.models.seq2seq.encoders.splicing import do_splice
from neural_sp.models.utils import np2var
from neural_sp.models.utils import pad_list


logger = logging.getLogger("training")


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
        self.bridge_layer = args.bridge_layer

        # for attention layer
        self.att_num_heads = args.att_num_heads
        self.share_attention = False  # TODO(hirofumi): between fwd and bwd

        # for decoder
        self.num_classes = args.num_classes
        self.num_classes_sub = args.num_classes_sub
        self.blank = 0
        self.sos = 2
        self.eos = 2
        self.pad = 3
        # NOTE: reserved in advance

        # for CTC
        self.ctc_weight = args.ctc_weight

        # for backward decoder
        assert 0 <= args.bwd_weight <= 1
        self.fwd_weight = 1 - args.bwd_weight
        self.bwd_weight = args.bwd_weight

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
                                  nin=0)
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
                                  batch_norm=args.conv_batch_norm)
        else:
            raise NotImplementedError(args.enc_type)

        # Bridge layer between the encoder and decoder
        if args.enc_type == 'cnn':
            self.bridge = LinearND(self.encoder.output_dim, args.dec_num_units)
            self.enc_num_units = args.dec_num_units
        elif args.bridge_layer:
            self.bridge = LinearND(self.enc_num_units, args.dec_num_units)
            self.enc_num_units = args.dec_num_units

        # MAIN TASK
        directions = []
        if self.fwd_weight > 0:
            directions.append('fwd')
        if self.bwd_weight > 0:
            directions.append('bwd')
        for dir in directions:
            if (dir == 'fwd' and args.ctc_weight < 1) or dir == 'bwd':
                # Attention layer
                if args.att_num_heads > 1:
                    att = MultiheadAttentionMechanism(
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
                    att = AttentionMechanism(
                        enc_num_units=self.enc_num_units,
                        dec_num_units=args.dec_num_units,
                        att_type=args.att_type,
                        att_dim=args.att_dim,
                        sharpening_factor=args.att_sharpening_factor,
                        sigmoid_smoothing=args.att_sigmoid_smoothing,
                        conv_out_channels=args.att_conv_num_channels,
                        conv_kernel_size=args.att_conv_width)

                # Cold fusion
                if args.rnnlm_cold_fusion and dir == 'fwd':
                    raise NotImplementedError()
                    # TODO(hirofumi): cold fusion for backward RNNLM
                else:
                    args.rnnlm_cold_fusion = False
            else:
                att = None

            # Decoder
            dec = Decoder(attention=att,
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
                          dropout_hidden=args.dropout_dec,
                          dropout_emb=args.dropout_emb,
                          ss_prob=args.ss_prob,
                          lsm_prob=args.lsm_prob,
                          init_with_enc=args.init_with_enc,
                          ctc_weight=args.ctc_weight if dir == 'fwd' or (
                              dir == 'bwd' and self.fwd_weight == 0) else 0,
                          ctc_fc_list=args.ctc_fc_list,
                          backward=(dir == 'bwd'),
                          rnnlm_cold_fusion=args.rnnlm_cold_fusion,
                          cold_fusion=args.cold_fusion,
                          internal_lm=args.internal_lm,
                          rnnlm_init=args.rnnlm_init,
                          rnnlm_task_weight=args.rnnlm_task_weight,
                          share_lm_softmax=args.share_lm_softmax,
                          global_weight=self.fwd_weight * args.main_task_weight if dir == 'fwd' else self.bwd_weight)
            setattr(self, 'dec_' + dir, dec)

        # SUB TASK
        # NOTE: only forward direction for the sub task
        if args.main_task_weight < 1:
            if args.ctc_weight_sub < 1:
                # Attention layer
                if args.att_num_heads_sub > 1:
                    att_sub = MultiheadAttentionMechanism(
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
                    att_sub = AttentionMechanism(
                        enc_num_units=self.enc_num_units,
                        dec_num_units=args.dec_num_units,
                        att_type=args.att_type,
                        att_dim=args.att_dim,
                        sharpening_factor=args.att_sharpening_factor,
                        sigmoid_smoothing=args.att_sigmoid_smoothing,
                        conv_out_channels=args.att_conv_num_channels,
                        conv_kernel_size=args.att_conv_width)
            else:
                att_sub = None

            # Decoder
            self.dec_fwd_sub1 = Decoder(attention=att_sub,
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
                                        dropout_hidden=args.dropout_dec,
                                        dropout_emb=args.dropout_emb,
                                        ss_prob=args.ss_prob,
                                        lsm_prob=args.lsm_prob,
                                        init_with_enc=args.init_with_enc,
                                        ctc_weight=args.ctc_weight_sub,
                                        ctc_fc_list=args.ctc_fc_list_sub,
                                        global_weight=1 - args.main_task_weight)

        if args.input_type == 'text':
            if args.num_classes == args.num_classes_sub:
                # Share the embedding layer between input and output
                self.embed_in = dec.embed
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
        if args.rnnlm_cold_fusion:
            self.init_weights(-1, dist='constant', keys=['cf_linear_lm_gate.fc.bias'])

    def forward(self, xs, ys, ys_sub, reporter, is_eval=False):
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

        report = {}

        # Compute XE loss for the forward decoder
        if self.fwd_weight > 0:
            loss_fwd, report_fwd = self.dec_fwd(xs, x_lens, ys)
            loss = loss_fwd
            report['loss.att'] = report_fwd['loss_att']
            report['loss.ctc'] = report_fwd['loss_ctc']
            report['loss.lm'] = report_fwd['loss_lm']
            report['acc.main'] = report_fwd['acc']
        else:
            loss = Variable(xs.new(1,).fill_(0.))

        # Compute XE loss for the backward decoder
        if self.bwd_weight > 0:
            loss_bwd, report_bwd = self.dec_bwd(xs, x_lens, ys)
            loss += loss_bwd
            report['loss.att-bwd'] = report_bwd['loss_att']
            if self.fwd_weight == 0:
                report['loss.ctc'] = report_bwd['loss_ctc']
            report['acc.bwd'] = report_bwd['acc']

        if self.main_task_weight < 1:
            ys_sub = [ys_sub[i] for i in perm_idx]
            loss_sub, report_sub = self.dec_fwd_sub1(xs_sub, x_lens_sub, ys_sub)
            loss += loss_sub
            report['loss.att-sub'] = report_sub['loss_att']
            report['loss.ctc-sub'] = report_sub['loss_ctc']
            report['acc.sub'] = report_sub['acc']

        # TODO(hirofumi): report here
        if reporter is not None:
            reporter.step(observation=report, is_eval=is_eval)

        return loss, reporter

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

        # Bridge between the encoder and decoder
        if self.enc_type == 'cnn' or self.bridge_layer:
            xs = self.bridge(xs)

            # if self.main_task_weight < 1:
            #     xs_sub = self.bridge_sub(xs_sub)
            # TODO(hirofumi):

        return xs, x_lens, xs_sub, x_lens_sub

    def decode(self, xs, decode_params, nbest=1, exclude_eos=False, task='',
               idx2token=None, refs=None):
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
            nbest (int):
            exclude_eos (bool): exclude <eos> from best_hyps
            task (str): sub1
            idx2token (): converter from index to token
            refs (list): gold transcriptions to compute log likelihood
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aws (list): A list of length `[B]`, which contains arrays of size `[L, T]`
            perm_idx (list): A list of length `[B]`

        """
        self.eval()

        # Sort by lenghts in the descending order
        perm_idx = sorted(list(six.moves.range(0, len(xs), 1)),
                          key=lambda i: len(xs[i]), reverse=True)
        xs = [xs[i] for i in perm_idx]
        # NOTE: must be descending order for pack_padded_sequence

        # Encode input features
        enc_out, x_lens, enc_out_sub, x_lens_sub = self._encode(xs)

        dir = 'fwd' if self.fwd_weight >= self.bwd_weight else 'bwd'

        if self.ctc_weight == 1:
            # Set RNNLM
            if decode_params['rnnlm_weight'] > 0:
                assert hasattr(self, 'rnnlm_' + dir)
                rnnlm = getattr(self, 'rnnlm_' + dir)
            else:
                rnnlm = None

            best_hyps = getattr(self, 'dec_' + dir).decode_ctc(
                enc_out, x_lens, decode_params['beam_width'], rnnlm)
            return best_hyps, None, perm_idx
        else:
            if decode_params['beam_width'] == 1:
                best_hyps, aws = getattr(self, 'dec_' + dir).greedy(
                    enc_out, x_lens, decode_params['max_len_ratio'], exclude_eos)
            else:
                # Set RNNLM
                if decode_params['rnnlm_weight'] > 0:
                    assert hasattr(self, 'rnnlm_' + dir)
                    rnnlm = getattr(self, 'rnnlm_' + dir)
                else:
                    rnnlm = None

                nbest_hyps, aws, scores = getattr(self, 'dec_' + dir).beam_search(
                    enc_out, x_lens, decode_params, rnnlm,
                    nbest, exclude_eos, idx2token, refs)

                if nbest == 1:
                    best_hyps = [hyp[0] for hyp in nbest_hyps]
                    aws = [aw[0] for aw in aws]
                else:
                    return nbest_hyps, aws, scores, perm_idx
                # NOTE: nbest >= 2 is used for MWER training only

            return best_hyps, aws, perm_idx
