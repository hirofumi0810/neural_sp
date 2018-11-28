#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Attention-based RNN sequence-to-sequence model (including CTC)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import numpy as np
import six
import torch
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
from neural_sp.models.seq2seq.encoders.splicing import splice
from neural_sp.models.torch_utils import np2var
from neural_sp.models.torch_utils import pad_list


logger = logging.getLogger("training")


class Seq2seq(ModelBase):
    """Attention-based RNN sequence-to-sequence model (including CTC)."""

    def __init__(self, args):

        super(ModelBase, self).__init__()

        # for encoder
        self.input_type = args.input_type
        self.input_dim = args.input_dim
        self.nstacks = args.nstacks
        self.nskips = args.nskips
        self.nsplices = args.nsplices
        self.enc_type = args.enc_type
        self.enc_nunits = args.enc_nunits
        if args.enc_type in ['blstm', 'bgru']:
            self.enc_nunits *= 2
        self.bridge_layer = args.bridge_layer

        # for attention layer
        self.attn_nheads = args.attn_nheads

        # for decoder
        self.vocab = args.vocab
        self.vocab_sub = args.vocab_sub
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
                                  nunits=args.enc_nunits,
                                  nprojs=args.enc_nprojs,
                                  nlayers=args.enc_nlayers,
                                  nlayers_sub=args.enc_nlayers_sub,
                                  dropout_in=args.dropout_in,
                                  dropout_hidden=args.dropout_enc,
                                  subsample=[s == '1' for s in args.subsample.split('_')],
                                  subsample_type=args.subsample_type,
                                  batch_first=True,
                                  nstacks=args.nstacks,
                                  nsplices=args.nsplices,
                                  conv_in_channel=args.conv_in_channel,
                                  conv_channels=args.conv_channels,
                                  conv_kernel_sizes=args.conv_kernel_sizes,
                                  conv_strides=args.conv_strides,
                                  conv_poolings=args.conv_poolings,
                                  conv_batch_norm=args.conv_batch_norm,
                                  residual=args.enc_residual,
                                  nin=0,
                                  layer_norm=args.layer_norm)
        elif args.enc_type == 'cnn':
            assert args.nstacks == 1 and args.nsplices == 1
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
            logger.info('insert a bridge layer')
            self.bridge = LinearND(self.encoder.output_dim, args.dec_nunits,
                                   dropout=args.dropout_enc)
            self.enc_nunits = args.dec_nunits
        elif args.bridge_layer:
            logger.info('insert a bridge layer')
            self.bridge = LinearND(self.enc_nunits, args.dec_nunits,
                                   dropout=args.dropout_enc)
            self.enc_nunits = args.dec_nunits

        # MAIN TASK
        directions = []
        if self.fwd_weight > 0:
            directions.append('fwd')
        if self.bwd_weight > 0:
            directions.append('bwd')
        for dir in directions:
            if (dir == 'fwd' and args.ctc_weight < 1) or dir == 'bwd':
                # Attention layer
                if args.attn_nheads > 1:
                    logger.info('multi-head attention')
                    attn = MultiheadAttentionMechanism(
                        enc_nunits=self.enc_nunits,
                        dec_nunits=args.dec_nunits,
                        attn_type=args.attn_type,
                        attn_dim=args.attn_dim,
                        sharpening_factor=args.attn_sharpening,
                        sigmoid_smoothing=args.attn_sigmoid,
                        conv_out_channels=args.attn_conv_nchannels,
                        conv_kernel_size=args.attn_conv_width,
                        nheads=args.attn_nheads)
                else:
                    logger.info('single-head attention')
                    attn = AttentionMechanism(
                        enc_nunits=self.enc_nunits,
                        dec_nunits=args.dec_nunits,
                        attn_type=args.attn_type,
                        attn_dim=args.attn_dim,
                        sharpening_factor=args.attn_sharpening,
                        sigmoid_smoothing=args.attn_sigmoid,
                        conv_out_channels=args.attn_conv_nchannels,
                        conv_kernel_size=args.attn_conv_width,
                        dropout=args.dropout_att)
            else:
                attn = None

            # Cold fusion
            if args.rnnlm_cold_fusion and dir == 'fwd':
                logger.inof('cold fusion')
                raise NotImplementedError()
                # TODO(hirofumi): cold fusion for backward RNNLM
            else:
                args.rnnlm_cold_fusion = False

            # Decoder
            dec = Decoder(attention=attn,
                          sos=self.sos,
                          eos=self.eos,
                          pad=self.pad,
                          enc_nunits=self.enc_nunits,
                          rnn_type=args.dec_type,
                          nunits=args.dec_nunits,
                          nlayers=args.dec_nlayers,
                          residual=args.dec_residual,
                          emb_dim=args.emb_dim,
                          vocab=self.vocab,
                          logits_temp=args.logits_temp,
                          dropout_hidden=args.dropout_dec,
                          dropout_emb=args.dropout_emb,
                          ss_prob=args.ss_prob,
                          lsm_prob=args.lsm_prob,
                          layer_norm=args.layer_norm,
                          init_with_enc=args.init_with_enc,
                          ctc_weight=args.ctc_weight if dir == 'fwd' or (
                              dir == 'bwd' and self.fwd_weight == 0) else 0,
                          ctc_fc_list=[int(fc) for fc in args.ctc_fc_list.split('_')],
                          input_feeding=args.input_feeding,
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
                if args.attn_nheads_sub > 1:
                    logger.info('multi-head attention (sub)')
                    attn_sub = MultiheadAttentionMechanism(
                        enc_nunits=self.enc_nunits,
                        dec_nunits=args.dec_nunits,
                        attn_type=args.attn_type,
                        attn_dim=args.attn_dim,
                        sharpening_factor=args.attn_sharpening,
                        sigmoid_smoothing=args.attn_sigmoid,
                        conv_out_channels=args.attn_conv_nchannels,
                        conv_kernel_size=args.attn_conv_width,
                        nheads=args.attn_nheads_sub)
                else:
                    logger.info('single-head attention (sub)')
                    attn_sub = AttentionMechanism(
                        enc_nunits=self.enc_nunits,
                        dec_nunits=args.dec_nunits,
                        attn_type=args.attn_type,
                        attn_dim=args.attn_dim,
                        sharpening_factor=args.attn_sharpening,
                        sigmoid_smoothing=args.attn_sigmoid,
                        conv_out_channels=args.attn_conv_nchannels,
                        conv_kernel_size=args.attn_conv_width,
                        dropout=args.dropout_att)
            else:
                attn_sub = None

            # Decoder
            self.dec_fwd_sub1 = Decoder(attention=attn_sub,
                                        sos=self.sos,
                                        eos=self.eos,
                                        pad=self.pad,
                                        enc_nunits=self.enc_nunits,
                                        rnn_type=args.dec_type,
                                        nunits=args.dec_nunits,
                                        nlayers=args.dec_nlayers,
                                        residual=args.dec_residual,
                                        emb_dim=args.emb_dim,
                                        vocab=self.vocab_sub,
                                        logits_temp=args.logits_temp,
                                        dropout_hidden=args.dropout_dec,
                                        dropout_emb=args.dropout_emb,
                                        ss_prob=args.ss_prob,
                                        lsm_prob=args.lsm_prob,
                                        layer_norm=args.layer_norm,
                                        init_with_enc=args.init_with_enc,
                                        ctc_weight=args.ctc_weight_sub,
                                        ctc_fc_list=[int(fc) for fc in args.ctc_fc_list_sub.split('_')],
                                        global_weight=1 - args.main_task_weight)

        if args.input_type == 'text':
            if args.vocab == args.vocab_sub:
                # Share the embedding layer between input and output
                self.embed_in = dec.embed
            else:
                self.embed_in = Embedding(vocab=args.vocab_sub,
                                          emb_dim=args.emb_dim,
                                          dropout=args.dropout_emb,
                                          ignore_index=self.pad)

        # Initialize weight matrices
        self.init_weights(args.param_init, dist=args.param_init_dist, ignore_keys=['bias'])

        # Initialize CNN layers like chainer
        self.init_weights(args.param_init, dist='lecun', keys=['conv'], ignore_keys=['score'])

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
            reporter ():
            is_eval (bool): the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (torch.autograd.Variable(float)): `[1]`
            acc (float): Token-level accuracy in teacher-forcing

        """
        if is_eval:
            self.eval()
            with torch.no_grad():
                loss, report = self._forward(xs, ys, ys_sub)
        else:
            self.train()
            loss, report = self._forward(xs, ys, ys_sub)

        # Report here
        if reporter is not None:
            reporter.step(observation=report, is_eval=is_eval)

        return loss, reporter

    def _forward(self, xs, ys, ys_sub):
        # Encode input features
        if self.input_type == 'speech':
            # Sort by lenghts in the descending order
            perm_idx = sorted(list(six.moves.range(0, len(xs), 1)),
                              key=lambda i: len(xs[i]), reverse=True)
            xs = [xs[i] for i in perm_idx]
            ys = [ys[i] for i in perm_idx]
            # NOTE: must be descending order for pack_padded_sequence
            xs, x_lens, xs_sub, x_lens_sub = self.encode(xs)
        else:
            # Sort by lenghts in the descending order
            perm_idx = sorted(list(six.moves.range(0, len(ys_sub), 1)),
                              key=lambda i: len(ys_sub[i]), reverse=True)
            ys = [ys[i] for i in perm_idx]
            ys_sub = [ys_sub[i] for i in perm_idx]
            # NOTE: must be descending order for pack_padded_sequence
            xs, x_lens, xs_sub, x_lens_sub = self.encode(ys_sub)

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

        return loss, report

    def encode(self, xs):
        """Encode acoustic or text features.

        Args:
            xs (list): A list of length `[B]`, which contains Variables of size `[T, input_dim]`
        Returns:
            xs (torch.autograd.Variable, float): `[B, T, enc_units]`
            x_lens (list): `[B]`
            OPTION:
                xs_sub (torch.autograd.Variable, float): `[B, T, enc_units]`
                x_lens_sub (list): `[B]`

        """
        if self.input_type == 'speech':
            # Frame stacking
            if self.nstacks > 1:
                xs = [stack_frame(x, self.nstacks, self.nskips)for x in xs]

            # Splicing
            if self.nsplices > 1:
                xs = [splice(x, self.nsplices, self.nstacks) for x in xs]

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
                fwd_bwd_attention (bool):
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
        with torch.no_grad():
            # Sort by lenghts in the descending order
            perm_idx = sorted(list(six.moves.range(0, len(xs), 1)),
                              key=lambda i: len(xs[i]), reverse=True)
            xs = [xs[i] for i in perm_idx]
            # NOTE: must be descending order for pack_padded_sequence

            # Encode input features
            enc_out, x_lens, enc_out_sub, x_lens_sub = self.encode(xs)

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
                if decode_params['beam_width'] == 1 and not decode_params['fwd_bwd_attention']:
                    best_hyps, aws = getattr(self, 'dec_' + dir).greedy(
                        enc_out, x_lens, decode_params['max_len_ratio'], exclude_eos)
                else:
                    if decode_params['fwd_bwd_attention']:
                        rnnlm_fwd = None
                        nbest_hyps_fwd, aws_fwd, scores_fwd = self.dec_fwd.beam_search(
                            enc_out, x_lens, decode_params, rnnlm_fwd,
                            decode_params['beam_width'], False, idx2token, refs)

                        rnnlm_bwd = None
                        nbest_hyps_bwd, aws_bwd, scores_bwd = self.dec_bwd.beam_search(
                            enc_out, x_lens, decode_params, rnnlm_bwd,
                            decode_params['beam_width'], False, idx2token, refs)
                        best_hyps = fwd_bwd_attention(nbest_hyps_fwd, aws_fwd, scores_fwd,
                                                      nbest_hyps_bwd, aws_bwd, scores_bwd,
                                                      idx2token, refs)
                        aws = None
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


def fwd_bwd_attention(nbest_hyps_fwd, aws_fwd, scores_fwd,
                      nbest_hyps_bwd, aws_bwd, scores_bwd,
                      idx2token=None, refs=None):
    """Forward-backward joint decoding.
    Args:
        nbest_hyps_fwd (list): A list of length `[B]`, which contains list of n hypotheses
        aws_fwd (list): A list of length `[B]`, which contains arrays of size `[L, T]`
        scores_fwd (list):
        nbest_hyps_bwd (list):
        aws_bwd (list):
        scores_bwd (list):
        idx2token (): converter from index to token
        refs ():
    Returns:

    """
    logger = logging.getLogger("decoding")
    batch_size = len(nbest_hyps_fwd)
    nbest = len(nbest_hyps_fwd[0])
    eos = 2

    best_hyps = []
    for b in range(batch_size):
        merged = []
        for n in range(nbest):
            # forward
            if len(nbest_hyps_fwd[b][n]) > 1:
                if nbest_hyps_fwd[b][n][-1] == eos:
                    merged.append({'hyp': nbest_hyps_fwd[b][n][:-1],
                                   'score': scores_fwd[b][n][-2]})
                   # NOTE: remove eos probability
                else:
                    merged.append({'hyp': nbest_hyps_fwd[b][n],
                                   'score': scores_fwd[b][n][-1]})
            else:
                # <eos> only
                logger.info(nbest_hyps_fwd[b][n])

            # backward
            if len(nbest_hyps_bwd[b][n]) > 1:
                if nbest_hyps_bwd[b][n][0] == eos:
                    merged.append({'hyp': nbest_hyps_bwd[b][n][1:],
                                   'score': scores_bwd[b][n][1]})
                   # NOTE: remove eos probability
                else:
                    merged.append({'hyp': nbest_hyps_bwd[b][n],
                                   'score': scores_bwd[b][n][0]})
            else:
                # <eos> only
                logger.info(nbest_hyps_bwd[b][n])

        for n_f in range(nbest):
            for n_b in range(nbest):
                for i_f in range(len(aws_fwd[b][n_f]) - 1):
                    for i_b in range(len(aws_bwd[b][n_b]) - 1):
                        t_prev = aws_bwd[b][n_b][i_b + 1].argmax(-1).item()
                        t_curr = aws_fwd[b][n_f][i_f].argmax(-1).item()
                        t_next = aws_bwd[b][n_b][i_b - 1].argmax(-1).item()

                        # the same token at the same time
                        if t_curr >= t_prev and t_curr <= t_next and nbest_hyps_fwd[b][n_f][i_f] == nbest_hyps_bwd[b][n_b][i_b]:
                            new_hyp = nbest_hyps_fwd[b][n_f][:i_f + 1].tolist() + \
                                nbest_hyps_bwd[b][n_b][i_b + 1:].tolist()
                            score_curr_fwd = scores_fwd[b][n_f][i_f] - scores_fwd[b][n_f][i_f - 1]
                            score_curr_bwd = scores_bwd[b][n_b][i_b] - scores_bwd[b][n_b][i_b + 1]
                            score_curr = max(score_curr_fwd, score_curr_bwd)
                            new_score = scores_fwd[b][n_f][i_f - 1] + scores_bwd[b][n_b][i_b + 1] + score_curr
                            merged.append({'hyp': new_hyp, 'score': new_score})

                            logger.info('time matching')
                            if idx2token is not None:
                                if refs is not None:
                                    logger.info('Ref: %s' % refs[b].lower())
                                logger.info('hyp (fwd): %s' % idx2token(nbest_hyps_fwd[b][n_f]))
                                logger.info('hyp (bwd): %s' % idx2token(nbest_hyps_bwd[b][n_b]))
                                logger.info('hyp (fwd-bwd): %s' % idx2token(new_hyp))
                            logger.info('log prob (fwd): %.3f' % scores_fwd[b][n_f][-1])
                            logger.info('log prob (bwd): %.3f' % scores_bwd[b][n_b][0])
                            logger.info('log prob (fwd-bwd): %.3f' % new_score)

        merged = sorted(merged, key=lambda x: x['score'], reverse=True)
        best_hyps.append(merged[0]['hyp'])

    return best_hyps
