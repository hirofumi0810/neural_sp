#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Hierarchical attention-based sequence-to-sequence model (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import six

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from src.models.pytorch_v3.attention.attention_layer import AttentionMechanism
from src.models.pytorch_v3.attention.attention_layer import MultiheadAttentionMechanism
from src.models.pytorch_v3.attention.attention_seq2seq import AttentionSeq2seq
from src.models.pytorch_v3.attention.decoder import Decoder
from src.models.pytorch_v3.ctc.decoders.beam_search_decoder import BeamSearchDecoder
from src.models.pytorch_v3.ctc.decoders.greedy_decoder import GreedyDecoder
from src.models.pytorch_v3.encoders.load_encoder import load
from src.models.pytorch_v3.linear import Embedding
from src.models.pytorch_v3.linear import LinearND
from src.models.pytorch_v3.lm.rnnlm import RNNLM
from src.models.pytorch_v3.utils import np2var
from src.models.pytorch_v3.utils import var2np


class HierarchicalAttentionSeq2seq(AttentionSeq2seq):

    def __init__(self,
                 enc_in_type,
                 enc_in_size,
                 n_stack,
                 n_skip,
                 n_splice,
                 conv_in_channel,
                 conv_channels,
                 conv_kernel_sizes,
                 conv_strides,
                 conv_poolings,
                 conv_batch_norm,
                 enc_type,
                 enc_bidirectional,
                 enc_n_units,
                 enc_n_projs,
                 enc_n_layers,
                 enc_n_layers_sub,  # ***
                 enc_residual,
                 subsample_list,
                 subsample_type,
                 att_type,
                 att_dim,
                 att_conv_n_channels,
                 att_conv_width,
                 att_n_heads,
                 att_n_heads_sub,  # ***
                 sharpening_factor,
                 sigmoid_smoothing,
                 bridge_layer,
                 dec_type,
                 dec_n_units,
                 dec_n_units_sub,  # ***
                 dec_n_layers,
                 dec_n_layers_sub,  # ***
                 dec_residual,
                 emb_dim,
                 emb_dim_sub,  # ***
                 bottle_dim,
                 bottle_dim_sub,
                 generate_feat,
                 n_classes,
                 n_classes_sub,  # ***
                 logits_temp,
                 param_init,
                 param_init_dist,
                 rec_weight_orthogonal,
                 drop_in,
                 drop_enc,
                 dropout_dec,
                 drop_emb,
                 ss_prob,
                 lsm_prob,
                 lsm_type,
                 main_loss_weight,  # ***
                 sub_loss_weight,  # ***
                 ctc_weight_sub,
                 bwd_sub,  # ***
                 lm_fusion='',
                 lm_fusion_sub='',  # ***
                 lm_config=None,
                 lm_config_sub=None,  # ***
                 lm_loss_weight=0,
                 lm_loss_weight_sub=0,  # ***
                 lm_init=False,
                 lm_init_sub=False,  # ***
                 finetune_gate=False,
                 n_classes_in=-1,
                 share_att=False):

        super(HierarchicalAttentionSeq2seq, self).__init__(
            enc_in_type=enc_in_type,
            enc_in_size=enc_in_size,
            n_stack=n_stack,
            n_skip=n_skip,
            n_splice=n_splice,
            conv_in_channel=conv_in_channel,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            conv_poolings=conv_poolings,
            conv_batch_norm=conv_batch_norm,
            enc_type=enc_type,
            enc_bidirectional=enc_bidirectional,
            enc_n_units=enc_n_units,
            enc_n_projs=enc_n_projs,
            enc_n_layers=enc_n_layers,
            enc_residual=enc_residual,
            subsample_list=subsample_list,
            subsample_type=subsample_type,
            att_type=att_type,
            att_dim=att_dim,
            att_conv_n_channels=att_conv_n_channels,
            att_conv_width=att_conv_width,
            att_n_heads=att_n_heads,
            sharpening_factor=sharpening_factor,
            sigmoid_smoothing=sigmoid_smoothing,
            bridge_layer=bridge_layer,
            dec_type=dec_type,
            dec_n_units=dec_n_units,
            dec_n_layers=dec_n_layers,
            dec_residual=dec_residual,
            emb_dim=emb_dim,
            bottle_dim=bottle_dim,
            generate_feat=generate_feat,
            n_classes=n_classes,
            logits_temp=logits_temp,
            param_init=param_init,
            param_init_dist=param_init_dist,
            rec_weight_orthogonal=rec_weight_orthogonal,
            drop_in=drop_in,
            drop_enc=drop_enc,
            dropout_dec=dropout_dec,
            drop_emb=drop_emb,
            ss_prob=ss_prob,
            lsm_prob=lsm_prob,
            lsm_type=lsm_type,
            weight_noise_std=weight_noise_std,
            ctc_weight=0,
            bwd_weight=0,
            lm_fusion=lm_fusion,
            lm_config=lm_config,
            lm_loss_weight=lm_loss_weight,
            lm_init=False,
            finetune_gate=False,  # not yet turn on here
            n_classes_in=n_classes_in)
        self.model_type = 'hierarchical_attention'

        # Setting for the encoder
        self.enc_n_units_sub = enc_n_units
        if enc_bidirectional:
            self.enc_n_units_sub *= 2
        self.enc_n_layers_sub = enc_n_layers_sub

        # Setting for the attention in the sub task
        self.att_n_heads_1 = att_n_heads_sub
        self.share_att = share_att
        if share_att:
            assert dec_n_units == dec_n_units_sub
            assert att_n_heads == att_n_heads_sub

        # Setting for the decoder in the sub task
        self.n_classes_sub = n_classes_sub + 1  # Add <EOS> class
        self.sos_1 = n_classes_sub
        self.eos_1 = n_classes_sub
        # NOTE: <SOS> and <EOS> have the same index

        # TODO: ここまで

        # Setting for MTL
        self.main_loss_weight = main_loss_weight
        self.sub_loss_weight = sub_loss_weight
        self.ctc_loss_weight_1 = ctc_weight_sub
        if ctc_weight_sub > 0:
            from src.models.pytorch_v3.ctc.ctc import my_warpctc
            self.warp_ctc = my_warpctc
        if bwd_sub:
            self.bwd_weight_1 = sub_loss_weight
        self.backward_1 = bwd_sub

        # Setting for the RNNLM fusion
        self.lm_fusion_sub = lm_fusion_sub
        self.lm_init_1 = lm_init_sub
        if lm_fusion_sub or lm_init_sub:
            assert lm_config is not None

        # Encoder
        # NOTE: overide encoder
        if enc_type in ['lstm', 'gru']:
            self.encoder = load(enc_type=enc_type)(
                input_size=enc_in_size,
                rnn_type=enc_type,
                bidirectional=enc_bidirectional,
                n_units=enc_n_units,
                n_projs=enc_n_projs,
                n_layers=enc_n_layers,
                n_layers_sub=enc_n_layers_sub,
                drop_in=drop_in,
                drop_hidden=drop_enc,
                subsample_list=subsample_list,
                subsample_type=subsample_type,
                batch_first=True,
                merge_bidirectional=False,
                pack_sequence=True,
                n_stack=n_stack,
                n_splice=n_splice,
                conv_in_channel=conv_in_channel,
                conv_channels=conv_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                conv_strides=conv_strides,
                conv_poolings=conv_poolings,
                conv_batch_norm=conv_batch_norm,
                residual=enc_residual,
                nin=0)
        elif enc_type == 'cnn':
            assert n_stack == 1 and n_splice == 1
            self.encoder = load(enc_type='cnn')(
                input_size=enc_in_size,
                in_channel=conv_in_channel,
                conv_channels=conv_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                conv_strides=conv_strides,
                poolings=conv_poolings,
                drop_in=drop_in,
                drop_hidden=drop_enc,
                batch_norm=conv_batch_norm)
        else:
            raise NotImplementedError

        self.is_bridge_sub = False
        if self.sub_loss_weight > 0:
            # Bridge layer between the encoder and decoder
            if enc_type == 'cnn':
                self.bridge_1 = LinearND(self.encoder.output_size, dec_n_units_sub)
                self.enc_n_units_sub = dec_n_units_sub
                self.is_bridge_sub = True
            elif bridge_layer:
                self.bridge_1 = LinearND(self.enc_n_units_sub, dec_n_units_sub)
                self.enc_n_units_sub = dec_n_units_sub
                self.is_bridge_sub = True
            else:
                self.is_bridge_sub = False

            # Attention layer (sub)
            if not share_att:
                if att_n_heads_sub > 1:
                    att = MultiheadAttentionMechanism(
                        enc_n_units=self.enc_n_units_sub,
                        dec_n_units=dec_n_units_sub,
                        att_type=att_type,
                        att_dim=att_dim,
                        sharpening_factor=sharpening_factor,
                        sigmoid_smoothing=sigmoid_smoothing,
                        out_channels=att_conv_n_channels,
                        kernel_size=att_conv_width,
                        n_heads=att_n_heads_sub)
                else:
                    att = AttentionMechanism(
                        enc_n_units=self.enc_n_units_sub,
                        dec_n_units=dec_n_units_sub,
                        att_type=att_type,
                        att_dim=att_dim,
                        sharpening_factor=sharpening_factor,
                        sigmoid_smoothing=sigmoid_smoothing,
                        out_channels=att_conv_n_channels,
                        kernel_size=att_conv_width)

                # RNNLM fusion
                if (lm_fusion_sub or self.lm_init_1) and not bwd_sub:
                    rnnlm_sub = RNNLM(
                        emb_dim=lm_config_sub['emb_dim'],
                        rnn_type=lm_config_sub['rnn_type'],
                        bidirectional=lm_config_sub['bidirectional'],
                        n_units=lm_config_sub['n_units'],
                        n_layers=lm_config_sub['n_layers'],
                        drop_emb=lm_config_sub['drop_emb'],
                        drop_hidden=lm_config_sub['drop_hidden'],
                        drop_out=lm_config_sub['drop_out'],
                        n_classes=lm_config_sub['n_classes'],
                        param_init_dist=lm_config_sub['param_init_dist'],
                        param_init=lm_config_sub['param_init'],
                        rec_weight_orthogonal=lm_config_sub['rec_weight_orthogonal'],
                        lsm_prob=lm_config['lsm_prob'],
                        tie_weights=lm_config_sub['tie_weights'],
                        residual=lm_config_sub['residual'],
                        backward=lm_config['backward'])
                else:
                    rnnlm_sub = None

            # Decoder (sub)
            setattr(self, 'dec_1_' + dir, Decoder(
                score_fn=att,
                sos=self.sos_0,
                eos=self.eos_0,
                enc_n_units=self.enc_n_units,
                rnn_type=dec_type,
                n_units=dec_n_units_sub,
                n_layers=dec_n_layers_sub,
                residual=dec_residual,
                emb_dim=emb_dim,
                bottle_dim=bottle_dim,
                generate_feat=generate_feat,
                n_classes=self.n_classes,
                logits_temp=logits_temp,
                dropout_dec=dropout_dec,
                drop_emb=drop_emb,
                ss_prob=ss_prob,
                lsm_prob=lsm_prob,
                lsm_type=lsm_type,
                backward=(dir == 'bwd'),
                lm_fusion=lm_fusion,
                lm_loss_weight=lm_loss_weight,
                rnnlm=rnnlm_sub))

        # CTC (sub)
        if ctc_weight_sub > 0:
            if self.is_bridge:
                self.fc_ctc_0 = LinearND(dec_n_units_sub, n_classes_sub + 1)
            else:
                self.fc_ctc_1 = LinearND(self.enc_n_units_sub, n_classes_sub + 1)

            # Set CTC decoders
            self._decode_ctc_greedy_np = GreedyDecoder(blank_index=0)
            self._decode_ctc_beam_np = BeamSearchDecoder(blank_index=0)
            # TODO(hirofumi): set space index

        # Fix all parameters except for gate
        if lm_fusion_sub and finetune_gate:
            assert lm_fusion_sub in ['cold_fusion_prob', 'cold_fusion_hidden']
            fix_params = ['fc_dec', 'fc_cv', 'fc_bottle',
                          'fc_cf_lm_logits', 'fc_cf_gate', 'fc_cf_gated_lm']

            for n, p in self.named_parameters():
                if n.split('.')[0] not in fix_params:
                    p.requires_grad = False

        # Initialize weight matricess
        self.init_weights(param_init, distribution=param_init_dist, ignore_keys=['bias'])

        # Initialize all biases with 0
        self.init_weights(0, distribution='constant', keys=['bias'])

        # Recurrent weights are orthogonalized
        if rec_weight_orthogonal:
            # encoder
            if enc_type != 'cnn':
                self.init_weights(param_init, distribution='orthogonal',
                                  keys=[enc_type, 'weight'], ignore_keys=['bias'])
            # decoder
            self.init_weights(param_init, distribution='orthogonal',
                              keys=[dec_type, 'weight'], ignore_keys=['bias'])

        # Initialize bias in forget gate with 1
        if init_forget_gate_bias_with_one:
            self.init_forget_gate_bias_with_one()

        # Initialize bias in gating with -1
        if lm_fusion_sub in ['cold_fusion_prob', 'cold_fusion_hidden']:
            self.init_weights(-1, distribution='constant', keys=['fc_cf_gate.fc.bias'])

    def forward(self, xs, ys, ys_sub, is_eval=False):
        """Forward computation.

        Args:
            xs (list): A list of length `[B]`, which contains arrays of size `[T, input_size]`
            ys (list): A list of lenght `[B]`, which contains arrays of size `[L]`
            ys_sub (list): A list of lenght `[B]`, which contains arrays of size `[L_sub]`
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (torch.autograd.Variable(float)): A tensor of size `[1]`
            loss_main (float):
            loss_sub (float):
            acc_main (float): Token-level accuracy in teacher-forcing in the main task
            acc_sub (float): Token-level accuracy in teacher-forcing in the sub task

        """
        if is_eval:
            self.eval()
        else:
            self.train()

            # Gaussian noise injection
            if self.weight_noise_injection:
                self.inject_weight_noise(mean=0, std=self.weight_noise_std)

        # Sort by lenghts in the descending order
        if is_eval and self.enc_type != 'cnn' or self.enc_in_type == 'text':
            perm_idx = sorted(list(six.moves.range(0, len(xs), 1)),
                              key=lambda i: len(xs[i]), reverse=True)
            xs = [xs[i] for i in perm_idx]
            ys = [ys[i] for i in perm_idx]
            ys_sub = [ys_sub[i] for i in perm_idx]
            # NOTE: must be descending order for pack_padded_sequence
            # NOTE: assumed that xs is already sorted in the training stage

        # Encode input features
        xs, x_lens, xs_sub, x_lens_sub = self._encode(xs, is_multi_task=True)

        # Main task
        if self.main_loss_weight > 0:
            ys = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long()
                  for y in ys]
            loss, acc_main = self.compute_xe_loss(
                xs, ys, x_lens, task=0, dir='fwd')
            loss *= self.main_loss_weight
        else:
            loss = Variable(xs.data.new(1,).fill_(0.))
            acc_main = 0.
        loss_main = loss.data[0]

        # Sub task (attention)
        if self.sub_loss_weight > 0:
            if self.backward_1:
                ys_sub = [np2var(np.fromiter(y[::-1], dtype=np.int64), self.device_id).long()
                          for y in ys_sub]
            else:
                ys_sub = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long()
                          for y in ys_sub]
            loss_sub, acc_sub = self.compute_xe_loss(
                xs_sub, ys_sub, x_lens_sub,
                task=1, dir='bwd' if self.backward_1 else 'fwd')
            loss_sub *= self.sub_loss_weight
        else:
            loss_sub = Variable(xs.data.new(1,).fill_(0.))
            acc_sub = 0.

        # Sub task (CTC)
        if self.ctc_weight_sub > 0:
            ys_sub_ctc = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long()
                          for y in ys_sub]
            ctc_loss_sub = self.compute_ctc_loss(
                xs_sub, ys_sub_ctc, x_lens_sub, task=1)
            loss_sub += ctc_loss_sub * self.ctc_weight_sub
        loss += loss_sub

        if not is_eval:
            # Update the probability of scheduled sampling
            self._step += 1
            if self.ss_prob > 0:
                self._ss_prob = min(
                    self.ss_prob, self.ss_prob / self.ss_max_step * self._step)

        return loss, loss_main, loss_sub.data[0], acc_main, acc_sub

    def decode(self, xs, beam_width, max_decode_len, min_decode_len=0, min_decode_len_ratio=0,
               len_penalty=0, cov_penalty=0, cov_threshold=0, lm_loss_weight=0,
               task_index=0, joint_decoding=False, space_index=-1, oov_index=-1,
               word2char=None, score_sub_weight=0, lm_loss_weight_sub=0,
               idx2word=None, idx2char=None, exclude_eos=True):
        """Decoding in the inference stage.

        Args:
            xs (list): A list of length `[B]`, which contains arrays of size `[T, input_size]`
            beam_width (int): the size of beam
            max_decode_len (int): the maximum sequence length of tokens
            min_decode_len (int): the minimum sequence length of tokens
            min_decode_len_ratio (float):
            len_penalty (float): length penalty
            cov_penalty (float): coverage penalty
            cov_threshold (float): threshold for coverage penalty
            lm_loss_weight (float): the weight of RNNLM score
            task_index (int): the index of a task
            joint_decoding (bool): None or onepass or rescoring
            space_index (int):
            oov_index (int):
            word2char ():
            score_sub_weight (float):
            lm_loss_weight_sub (float): the weight of RNNLM score of the sub task
            idx2word: for debug
            idx2char: for debug
            exclude_eos (bool): if True, exclude <EOS> from best_hyps
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
            aw (np.ndarray): A tensor of size `[B, L, T]`
            perm_idx (list): A list of length `[B]`

        """
        self.eval()

        if task_index > 0 and self.ctc_weight_sub > self.sub_loss_weight:
            # Decode by CTC decoder
            best_hyps, perm_idx = self.decode_ctc(xs, beam_width, task_index)

            return best_hyps, None, perm_idx
            # NOTE: None corresponds to aw in attention-based models
        else:
            # Sort by lenghts in the descending order
            if self.enc_type != 'cnn' or self.enc_in_type == 'text':
                perm_idx = sorted(list(six.moves.range(0, len(xs), 1)),
                                  key=lambda i: len(xs[i]), reverse=True)
                xs = [xs[i] for i in perm_idx]
                # NOTE: must be descending order for pack_padded_sequence
            else:
                perm_idx = list(six.moves.range(0, len(xs), 1))

            dir = 'bwd' if task_index == 1 and self.backward_1 else 'fwd'

            # Encode input features
            if joint_decoding and task_index == 0 and dir == 'fwd':
                enc_out, x_lens, enc_out_sub, x_lens_sub = self._encode(
                    xs, is_multi_task=True)
            elif task_index == 0:
                enc_out, x_lens, _, _ = self._encode(xs, is_multi_task=True)
            elif task_index == 1:
                _, _, enc_out, x_lens = self._encode(xs, is_multi_task=True)
            else:
                raise NotImplementedError

            # Decode by attention decoder
            if joint_decoding and task_index == 0 and dir == 'fwd':
                best_hyps, aw, best_hyps_sub, aw_sub, = self._decode_infer_joint(
                    enc_out, x_lens,
                    enc_out_sub, x_lens_sub,
                    beam_width, max_decode_len, min_decode_len, min_decode_len_ratio,
                    len_penalty, cov_penalty, cov_threshold, lm_loss_weight, lm_loss_weight_sub,
                    space_index, oov_index, word2char, score_sub_weight,
                    idx2word, idx2char, exclude_eos)

                return best_hyps, aw, best_hyps_sub, aw_sub, perm_idx
            else:
                if beam_width == 1:
                    best_hyps, aw = self._decode_infer_greedy(
                        enc_out, x_lens, max_decode_len, task_index, dir, exclude_eos)
                else:
                    best_hyps, aw = self._decode_infer_beam(
                        enc_out, x_lens, beam_width, max_decode_len, min_decode_len,
                        min_decode_len_ratio, len_penalty, cov_penalty, cov_threshold,
                        lm_loss_weight, task_index, dir, exclude_eos)

            return best_hyps, aw, perm_idx

    def _decode_infer_joint(self, enc_out, x_lens, enc_out_sub, x_lens_sub,
                            beam_width, max_decode_len, min_decode_len, min_decode_len_ratio,
                            len_penalty, cov_penalty, cov_threshold,
                            lm_loss_weight, lm_loss_weight_sub,
                            space_index, oov_index, word2char, score_sub_weight,
                            idx2word, idx2char, exclude_eos):
        """Joint decoding (one-pass).

        Args:
            enc_out (torch.FloatTensor): A tensor of size
                `[B, T, enc_n_units]`
            x_lens (list): A list of length `[B]`
            enc_out_sub (torch.FloatTensor): A tensor of size
                `[B, T_in_sub, enc_n_units]`
            x_lens_sub (list): A list of length `[B]`
            beam_width (int): the size of beam in the main task
            max_decode_len (int): the maximum sequence length of tokens
            min_decode_len (int): the minimum sequence length of tokens
            min_decode_len_ratio (float):
            len_penalty (float): length penalty
            cov_penalty (float): coverage penalty
            cov_threshold (float): threshold for converage penalty
            lm_loss_weight (float): the weight of RNNLM score of the main task
            lm_loss_weight_sub (float): the weight of RNNLM score of the sub task
            space_index (int):
            oov_index (int):
            word2char ():
            score_sub_weight (float):
            idx2word (): for debug
            idx2char (): for debug
            exclude_eos (bool): if True, exclude <EOS> from best_hyps
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, L]`
            aw (np.ndarray): A tensor of size `[B, L, T]`
            best_hyps_sub (np.ndarray): A tensor of size `[B, L_sub]`
            aw_sub (np.ndarray): A tensor of size `[B, L_sub, T]`
            aw_dec (np.ndarray): A tensor of size `[B, L, L_sub]`

        """
        debug = False
        # debug = True

        cancel_prev_score_sub = True
        # cancel_prev_score_sub = False

        batch, max_time = enc_out.size()[:2]

        if lm_loss_weight > 0 or self.lm_fusion_0:
            assert self.rnnlm_0_fwd is not None
            assert not self.rnnlm_0_fwd.training
        if lm_loss_weight_sub > 0 and self.lm_fusion_1:
            assert self.rnnlm_1_fwd is not None
            assert not self.rnnlm_1_fwd.training

        if self.lm_fusion_0 or self.lm_fusion_1:
            raise NotImplementedError

        best_hyps, aw = [], []
        best_hyps_sub, aw_sub = [], []
        eos_flags = [False] * batch
        eos_flags_sub = [False] * batch
        for b in six.moves.range(batch):
            # Initialization for the word model per utterance
            dec_out, hx_list, cx_list = self._init_dec_state(
                enc_out[b:b + 1], x_lens[b], task=0, dir='fwd')
            cv = Variable(enc_out.data.new(
                1, 1, enc_out.size(-1)).fill_(0.), volatile=True)
            self.attend_0_fwd.reset()

            dec_out_sub, hx_list_sub, cx_list_sub = self._init_dec_state(
                enc_out_sub[b:b + 1], x_lens_sub[b], task=1, dir='fwd')
            cv_sub = Variable(enc_out.data.new(
                1, 1, enc_out_sub.size(-1)).fill_(0.), volatile=True)
            self.attend_1_fwd.reset()

            complete = []
            beam = [{'hyp': [self.sos_0],
                     'hyp_sub': [self.sos_1],
                     'score': 0,  # log1
                     'score_sub': 0,  # log 1
                     'cx_list': cx_list,
                     'hx_list': hx_list,
                     'dec_out': dec_out,
                     'dec_out_sub': dec_out_sub,
                     'cx_list_sub': cx_list_sub,
                     'hx_list_sub': hx_list_sub,
                     'cv': cv,
                     'cv_sub': cv_sub,
                     'aw_steps': [None],
                     'aw_steps_sub':[None],
                     'rnnlm_state': None,
                     'lm_state_sub': None,
                     'pre_cov': 0}]
            for t in six.moves.range(max_decode_len + 1):
                new_beam = []
                for i_beam in six.moves.range(len(beam)):
                    # Update RNNLM states
                    if lm_loss_weight > 0 or self.lm_fusion_0:
                        y_rnnlm = Variable(enc_out.data.new(
                            1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                        y_rnnlm = self.rnnlm_0_fwd.embed(y_rnnlm)
                        rnnlm_logits_step, rnnlm_out, rnnlm_state = self.rnnlm_0_fwd.predict(
                            y_rnnlm, h=beam[i_beam]['rnnlm_state'])

                    y = Variable(enc_out.data.new(
                        1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                    y = self.embed_0(y)

                    if t == 0:
                        dec_out = beam[i_beam]['dec_out']
                    else:
                        # Recurrency
                        dec_in = torch.cat(
                            [y,  beam[i_beam]['cv']], dim=-1)
                        dec_out, hx_list, cx_list = self.decoder_0_fwd(
                            dec_in, beam[i_beam]['hx_list'], beam[i_beam]['cx_list'])

                    # Score
                    cv, aw_step = self.attend_0_fwd(
                        enc_out[b:b + 1, :x_lens[b]], x_lens[b:b + 1],
                        dec_out, beam[i_beam]['aw_steps'][-1])

                    # Generate
                    out = self.W_d_0_fwd(dec_out) + \
                        self.W_c_0_fwd(cv)
                    logits_step = self.fc_0_fwd(F.tanh(out))

                    # Path through the log-softmax layer
                    log_probs = F.log_softmax(logits_step.squeeze(1), dim=1)

                    # Pick up the top-k scores
                    log_probs_topk, indices_topk = log_probs.topk(
                        beam_width, dim=1, largest=True, sorted=True)

                    for k in six.moves.range(beam_width):
                        # Exclude short hypotheses
                        if indices_topk[0, k].data[0] == self.eos_0 and len(beam[i_beam]['hyp']) < min_decode_len:
                            continue
                        if indices_topk[0, k].data[0] == self.eos_0 and len(beam[i_beam]['hyp']) < x_lens[b] * min_decode_len_ratio:
                            continue

                        # Add length penalty
                        score = beam[i_beam]['score'] + \
                            log_probs_topk.data[0, k] + len_penalty

                        # Add coverage penalty
                        if cov_penalty > 0:
                            # Recompute converage penalty in each step
                            score -= beam[i_beam]['pre_cov'] * \
                                cov_penalty

                            aw_steps = torch.stack(
                                beam[i_beam]['aw_steps'][1:] + [aw_step], dim=1)

                            if self.num_heads_0 > 1:
                                cov_sum = aw_steps.data[0,
                                                        :, :, 0].cpu().numpy()
                                # TODO(hirofumi): fix for MHA
                            else:
                                cov_sum = aw_steps.data[0].cpu().numpy()
                            if cov_threshold == 0:
                                cov_sum = np.sum(cov_sum)
                            else:
                                cov_sum = np.sum(cov_sum[np.where(
                                    cov_sum > cov_threshold)[0]])
                            score += cov_sum * cov_penalty
                        else:
                            cov_sum = 0

                        # Add RNNLM score
                        if lm_loss_weight > 0:
                            rnnlm_log_probs = F.log_softmax(
                                rnnlm_logits_step.squeeze(1), dim=1)
                            assert log_probs.size() == rnnlm_log_probs.size()
                            score += rnnlm_log_probs.data[0,
                                                          indices_topk.data[0, k]] * lm_loss_weight
                        else:
                            rnnlm_state = None

                        # NOTE: Resocre by the second decoder's score
                        oov_flag = indices_topk.data[0, k] == oov_index
                        eos_flag = indices_topk.data[0, k] == self.eos_0
                        score_c2w = 0  # log_1
                        score_c2w_until_prev_space = 0  # log_1
                        word_idx = indices_topk.data[0, k]

                        if oov_flag:
                            # Add previous space score
                            if t > 0:
                                # Score
                                cv_sub, aw_step_sub = self.attend_1_fwd(
                                    enc_out_sub[b:b + 1, :x_lens_sub[b]],
                                    x_lens_sub[b:b + 1],
                                    beam[i_beam]['dec_out_sub'],
                                    beam[i_beam]['aw_steps_sub'][-1])

                                # Generate
                                logits_step_sub = self.fc_1_fwd(F.tanh(
                                    self.W_d_1_fwd(beam[i_beam]['dec_out_sub']) +
                                    self.W_c_1_fwd(cv_sub)))

                                # Recurrency
                                y_sub = Variable(enc_out.data.new(
                                    1, 1).fill_(space_index).long(), volatile=True)
                                y_sub = self.embed_1(y_sub)
                                dec_in_sub = torch.cat(
                                    [y_sub, cv_sub], dim=-1)
                                dec_out_sub, hx_list_sub, cx_list_sub = self.decoder_1_fwd(
                                    dec_in_sub, beam[i_beam]['hx_list_sub'], beam[i_beam]['cx_list_sub'])

                                # Path through the log-softmax layer
                                log_probs_sub = F.log_softmax(
                                    logits_step_sub.squeeze(1), dim=-1)

                                # Add RNNLM score in the sub task
                                if lm_loss_weight_sub > 0:
                                    y_rnnlm_sub = Variable(enc_out.data.new(
                                        1, 1).fill_(beam[i_beam]['hyp_sub'][-1]).long(), volatile=True)
                                    y_rnnlm_sub = self.rnnlm_1_fwd.embed(
                                        y_rnnlm_sub)
                                    lm_logits_step_sub, lm_out_sub, lm_state_sub = self.rnnlm_1_fwd.predict(
                                        y_rnnlm_sub, h=beam[i_beam]['lm_state_sub'])
                                    lm_log_probs_sub = F.log_softmax(
                                        lm_logits_step_sub.squeeze(1), dim=1)
                                    assert log_probs_sub.size() == lm_log_probs_sub.size()
                                    score_c2w_until_prev_space += lm_log_probs_sub.data[0,
                                                                                        space_index] * lm_loss_weight_sub
                                else:
                                    lm_state_sub = None

                                score_c2w_until_prev_space += log_probs_sub.data[0,
                                                                                 space_index] * score_sub_weight
                                charseq_init = [space_index]
                                aw_steps_sub = [
                                    beam[i_beam]['aw_steps_sub'][-1], aw_step_sub]
                            else:
                                dec_out_sub = beam[i_beam]['dec_out_sub']
                                hx_list_sub = beam[i_beam]['hx_list_sub']
                                cx_list_sub = beam[i_beam]['cx_list_sub']
                                cv_sub = beam[i_beam]['cv_sub']
                                charseq_init = []
                                aw_steps_sub = [
                                    beam[i_beam]['aw_steps_sub'][-1]]
                                lm_state_sub = beam[i_beam]['lm_state_sub']

                            # TODO(hirofumi): add max OOV len
                            beam_width_sub = beam_width
                            # Decode until outputting a space or eos
                            complete_sub = []
                            beam_sub = [{'hyp': charseq_init,
                                         'score': 0,  # log 1
                                         'dec_out': dec_out_sub,
                                         'hx_list': hx_list_sub,
                                         'cx_list': cx_list_sub,
                                         'cv': cv_sub,
                                         'aw_steps': aw_steps_sub,
                                         'rnnlm_state': lm_state_sub}]
                            for t_sub in six.moves.range(20):
                                new_beam_sub = []
                                for i_beam_sub in six.moves.range(len(beam_sub)):
                                    if t_sub == 0:
                                        dec_out_sub = beam_sub[i_beam_sub]['dec_out']
                                    else:
                                        y_sub = Variable(enc_out.data.new(
                                            1, 1).fill_(beam_sub[i_beam_sub]['hyp'][-1]).long(), volatile=True)
                                        y_sub = self.embed_1(y_sub)

                                        # Recurrency
                                        dec_in_sub = torch.cat(
                                            [y_sub, beam_sub[i_beam_sub]['cv']], dim=-1)
                                        dec_out_sub, hx_list_sub, cx_list_sub = self.decoder_1_fwd(
                                            dec_in_sub, beam_sub[i_beam_sub]['hx_list'], beam_sub[i_beam_sub]['cx_list'])

                                    # Score
                                    cv_sub, aw_step_sub = self.attend_1_fwd(
                                        enc_out_sub[b:b + 1, :x_lens_sub[b]],
                                        x_lens_sub[b:b + 1],
                                        dec_out_sub,
                                        beam_sub[i_beam_sub]['aw_steps'][-1])

                                    # Generate
                                    logits_step_sub = self.fc_1_fwd(F.tanh(
                                        self.W_d_1_fwd(beam_sub[i_beam_sub]['dec_out']) +
                                        self.W_c_1_fwd(cv_sub)))

                                    # Path through the log-softmax layer
                                    log_probs_sub = F.log_softmax(
                                        logits_step_sub.squeeze(1), dim=-1)

                                    # Pick up the top-k scores
                                    log_probs_topk_sub, indices_topk_sub = torch.topk(
                                        log_probs_sub, k=beam_width_sub, dim=1, largest=True, sorted=True)

                                    for k_sub in six.moves.range(beam_width_sub):
                                        if indices_topk_sub.data[0, k_sub] in [self.eos_1, space_index]:
                                            if t_sub > 0:
                                                # Remove complete hypotheses
                                                complete_sub += [beam_sub[i_beam_sub]]
                                            continue

                                        score_c2w = log_probs_topk_sub.data[0,
                                                                            k_sub] * score_sub_weight

                                        # Add RNNLM score in the sub task
                                        if lm_loss_weight_sub > 0:
                                            if t == 0 and t_sub == 0:
                                                y_rnnlm_sub = Variable(enc_out.data.new(
                                                    1, 1).fill_(self.sos_1).long(), volatile=True)
                                            else:
                                                y_rnnlm_sub = Variable(enc_out.data.new(
                                                    1, 1).fill_(beam_sub[i_beam_sub]['hyp'][-1]).long(), volatile=True)
                                            y_rnnlm_sub = self.rnnlm_1_fwd.embed(
                                                y_rnnlm_sub)
                                            logits_step_rnnlm_sub, lm_out_sub, lm_state_sub = self.rnnlm_1_fwd.predict(
                                                y_rnnlm_sub, h=beam_sub[i_beam_sub]['rnnlm_state'])

                                            lm_log_probs_sub = F.log_softmax(
                                                logits_step_rnnlm_sub.squeeze(1), dim=1)
                                            assert log_probs_sub.size() == lm_log_probs_sub.size()
                                            score_c2w += lm_log_probs_sub.data[0,
                                                                               indices_topk_sub.data[0, k_sub]] * lm_loss_weight_sub
                                        else:
                                            lm_state_sub = None

                                        new_beam_sub.append(
                                            {'hyp': beam_sub[i_beam_sub]['hyp'] + [indices_topk_sub.data[0, k_sub]],
                                             'score': score_c2w,
                                             'dec_out': dec_out_sub,
                                             'hx_list': copy.deepcopy(hx_list_sub),
                                             'cx_list': copy.deepcopy(cx_list_sub),
                                             'cv': cv_sub,
                                             'aw_steps': beam_sub[i_beam_sub]['aw_steps'] + [aw_step_sub],
                                             'rnnlm_state': lm_state_sub})

                                new_beam_sub = sorted(
                                    new_beam_sub, key=lambda x: x['score'], reverse=True)

                                if len(complete_sub) >= beam_width_sub:
                                    complete_sub = complete_sub[: beam_width_sub]
                                    break

                                beam_sub = new_beam_sub[: beam_width_sub]

                            if len(complete_sub) == 0:
                                complete_sub = beam_sub

                            complete_sub = sorted(
                                complete_sub, key=lambda x: x['score'], reverse=True)

                            score_c2w += log_probs.data[0,
                                                        oov_index] * (self.n_classes_sub / self.n_classes) * score_sub_weight
                            # NOTE: approximate OOV prob of A2C by that of A2W
                            charseq = complete_sub[0]['hyp']
                            aw_steps_sub = complete_sub[0]['aw_steps'][1:]
                            # NOTE; remove start aw
                            dec_out_sub = complete_sub[0]['dec_out']
                            hx_list_sub = complete_sub[0]['hx_list']
                            cx_list_sub = complete_sub[0]['cx_list']

                            if debug:
                                print('Step: OOV')
                                print(idx2char(complete_sub[0]['hyp']))

                        elif eos_flag:
                            # Score
                            cv_sub, aw_step_sub = self.attend_1_fwd(
                                enc_out_sub[b:b + 1, :x_lens_sub[b]],
                                x_lens_sub[b:b + 1],
                                beam[i_beam]['dec_out_sub'],
                                beam[i_beam]['aw_steps_sub'][-1])

                            # Generate
                            logits_step_sub = self.fc_1_fwd(F.tanh(
                                self.W_d_1_fwd(dec_out_sub) +
                                self.W_c_1_fwd(cv_sub)))

                            # Path through the log-softmax layer
                            log_probs_sub = F.log_softmax(logits_step_sub.squeeze(1), dim=-1)

                            # Add RNNLM score in the sub task
                            if lm_loss_weight_sub > 0:
                                y_rnnlm_sub = Variable(enc_out.data.new(
                                    1, 1).fill_(beam[i_beam]['hyp_sub'][-1]).long(), volatile=True)
                                y_rnnlm_sub = self.rnnlm_1_fwd.embed(
                                    y_rnnlm_sub)
                                lm_logits_step_sub, lm_out_sub, lm_state_sub = self.rnnlm_1_fwd.predict(
                                    y_rnnlm_sub, h=beam[i_beam]['lm_state_sub'])
                                lm_log_probs_sub = F.log_softmax(
                                    lm_logits_step_sub.squeeze(1), dim=1)
                                assert log_probs_sub.size() == lm_log_probs_sub.size()
                                score_c2w += lm_log_probs_sub.data[0,
                                                                   self.eos_1] * lm_loss_weight_sub
                            else:
                                lm_state_sub = None

                            score_c2w += log_probs_sub.data[0,
                                                            self.eos_1] * score_sub_weight
                            charseq = [self.eos_1]
                            aw_steps_sub = [aw_step_sub]

                            if debug:
                                print('Step: <EOS>')
                                print(idx2word([word_idx]))
                                print(idx2char([self.eos_1]))

                        else:
                            # Decompose a word to characters
                            charseq = word2char(word_idx)
                            # charseq: `[num_chars,]` (list)

                            if t > 0:
                                charseq = [space_index] + charseq

                            if debug:
                                print('Step: decompose')
                                print(idx2word([word_idx]))
                                print(idx2char(charseq))

                            dec_out_sub = beam[i_beam]['dec_out_sub']
                            hx_list_sub = beam[i_beam]['hx_list_sub']
                            cx_list_sub = beam[i_beam]['cx_list_sub']
                            aw_steps_sub = [beam[i_beam]['aw_steps_sub'][-1]]
                            lm_state_sub = beam[i_beam]['lm_state_sub']
                            for t_sub in six.moves.range(len(charseq)):
                                # Score
                                cv_sub, aw_step_sub = self.attend_1_fwd(
                                    enc_out_sub[b:b + 1, :x_lens_sub[b]],
                                    x_lens_sub[b:b + 1],
                                    dec_out_sub, aw_steps_sub[-1])

                                # Generate
                                logits_step_sub = self.fc_1_fwd(F.tanh(
                                    self.W_d_1_fwd(dec_out_sub) +
                                    self.W_c_1_fwd(cv_sub)))

                                # Path through the log-softmax layer
                                log_probs_sub = F.log_softmax(
                                    logits_step_sub.squeeze(1), dim=-1)

                                # Add RNNLM score in the sub task
                                if lm_loss_weight_sub > 0:
                                    if t_sub == 0:
                                        y_rnnlm_sub = Variable(enc_out.data.new(
                                            1, 1).fill_(beam[i_beam]['hyp_sub'][-1]).long(), volatile=True)
                                    else:
                                        y_rnnlm_sub = Variable(enc_out.data.new(
                                            1, 1).fill_(charseq[t_sub - 1]).long(), volatile=True)
                                    y_rnnlm_sub = self.rnnlm_1_fwd.embed(
                                        y_rnnlm_sub)
                                    lm_logits_step_sub, lm_out_sub, lm_state_sub = self.rnnlm_1_fwd.predict(
                                        y_rnnlm_sub, h=lm_state_sub)
                                    lm_log_probs_sub = F.log_softmax(
                                        lm_logits_step_sub.squeeze(1), dim=1)
                                    assert log_probs_sub.size() == lm_log_probs_sub.size()
                                    if charseq[t_sub] == space_index:
                                        score_c2w_until_prev_space += lm_log_probs_sub.data[0,
                                                                                            space_index] * lm_loss_weight_sub
                                    else:
                                        score_c2w += lm_log_probs_sub.data[0,
                                                                           charseq[t_sub]] * lm_loss_weight_sub
                                else:
                                    lm_state_sub = None

                                if charseq[t_sub] == space_index:
                                    score_c2w_until_prev_space += log_probs_sub.data[0,
                                                                                     space_index] * score_sub_weight
                                else:
                                    score_c2w += log_probs_sub.data[0,
                                                                    charseq[t_sub]] * score_sub_weight
                                aw_steps_sub += [aw_step_sub]

                                # teacher-forcing
                                y_sub = Variable(enc_out.data.new(
                                    1, 1).fill_(charseq[t_sub]).long(), volatile=True)
                                y_sub = self.embed_1(y_sub)

                                # Recurrency
                                dec_in_sub = torch.cat([y_sub, cv_sub], dim=-1)
                                dec_out_sub, hx_list_sub, cx_list_sub = self.decoder_1_fwd(
                                    dec_in_sub, hx_list_sub, cx_list_sub)

                            aw_steps_sub = aw_steps_sub[1:]

                        # Substruct character-lelel score up to this step
                        if cancel_prev_score_sub:
                            score -= beam[i_beam]['score_sub']

                        # Rescoreing
                        score_sub_local = score_c2w - score_c2w_until_prev_space
                        # print(score_sub_local)
                        # print((len(charseq) -
                        #        charseq.count(space_index)) ** 0.5)
                        score_sub_local /= (len(charseq) -
                                            charseq.count(space_index)) ** 0.5
                        # print(score_sub_local)
                        # print('=' * 20)

                        score += score_sub_local

                        new_beam.append(
                            {'hyp': beam[i_beam]['hyp'] + [indices_topk.data[0, k]],
                             'hyp_sub': beam[i_beam]['hyp_sub'] + charseq,
                             'score': score,
                             'score_sub': score_sub_local,
                             'dec_out': dec_out,
                             'hx_list': copy.deepcopy(hx_list),
                             'cx_list': copy.deepcopy(cx_list),
                             'dec_out_sub': dec_out_sub,
                             'hx_list_sub': hx_list_sub,
                             'cx_list_sub': cx_list_sub,
                             'cv': cv,
                             'cv_sub': cv_sub,
                             'aw_steps': beam[i_beam]['aw_steps'] + [aw_step],
                             'aw_steps_sub': beam[i_beam]['aw_steps_sub'] + aw_steps_sub,
                             'rnnlm_state': rnnlm_state,
                             'lm_state_sub': lm_state_sub,
                             'pre_cov': cov_sum})

                new_beam = sorted(
                    new_beam, key=lambda x: x['score'], reverse=True)

                # Remove complete hypotheses
                not_complete = []
                for cand in new_beam[:beam_width]:
                    if cand['hyp'][-1] == self.eos_0:
                        complete += [cand]
                    else:
                        not_complete += [cand]
                if len(complete) >= beam_width:
                    complete = complete[:beam_width]
                    break
                beam = not_complete[:beam_width]

            if len(complete) == 0:
                complete = beam

            # NOTE: substruct character-lelel score at the last step
            if cancel_prev_score_sub:
                complete = sorted(
                    complete, key=lambda x: x['score'] - x['score_sub'], reverse=True)
            else:
                complete = sorted(
                    complete, key=lambda x: x['score'], reverse=True)
            best_hyps += [np.array(complete[0]['hyp'][1:])]
            aw += [complete[0]['aw_steps'][1:]]
            best_hyps_sub += [np.array(complete[0]['hyp_sub'][1:])]
            aw_sub += [complete[0]['aw_steps_sub'][1:]]
            if complete[0]['hyp'][-1] == self.eos_0:
                eos_flags[b] = True
            if complete[0]['hyp_sub'][-1] == self.eos_1:
                eos_flags_sub[b] = True

            if debug:
                print(complete[0]['score'])
                print(complete[0]['score_sub'])

        # Concatenate in L dimension
        for b in six.moves.range(len(aw)):
            aw_sub[b] = var2np(torch.stack(aw_sub[b], dim=1).squeeze(0))
            if self.att_n_heads_1 > 1:
                aw_sub[b] = aw_sub[b][:, :, 0]
                # TODO(hirofumi): fix for MHA

            aw[b] = var2np(torch.stack(aw[b], dim=1).squeeze(0))
            if self.num_heads_0 > 1:
                aw[b] = aw[b][:, :, 0]
                # TODO(hirofumi): fix for MHA

        # Exclude <EOS>
        if exclude_eos:
            best_hyps = [best_hyps[b][:-1] if eos_flags[b]
                         else best_hyps[b] for b in six.moves.range(batch)]
            best_hyps_sub = [best_hyps_sub[b][:-1] if eos_flags_sub[b]
                             else best_hyps_sub[b] for b in six.moves.range(batch)]

        return best_hyps, aw, best_hyps_sub, aw_sub
