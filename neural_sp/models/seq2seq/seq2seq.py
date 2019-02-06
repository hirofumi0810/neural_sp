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
import math
import numpy as np
import torch

from neural_sp.models.base import ModelBase
from neural_sp.models.linear import Embedding
from neural_sp.models.linear import LinearND
from neural_sp.models.rnnlm.rnnlm import RNNLM
from neural_sp.models.seq2seq.decoders.decoder import Decoder
from neural_sp.models.seq2seq.encoders.frame_stacking import stack_frame
from neural_sp.models.seq2seq.encoders.rnn import RNNEncoder
from neural_sp.models.seq2seq.encoders.splicing import splice
from neural_sp.models.seq2seq.encoders.transformer import TransformerEncoder
from neural_sp.models.torch_utils import np2tensor
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
        self.vocab_sub1 = args.vocab_sub1
        self.vocab_sub2 = args.vocab_sub2
        self.vocab_sub3 = args.vocab_sub3
        self.blank = 0
        self.unk = 1
        self.sos = 2  # NOTE: the same index as <eos>
        self.eos = 2
        self.pad = 3
        # NOTE: reserved in advance

        # for the sub tasks
        self.main_weight = 1 - args.sub1_weight - args.sub2_weight - args.sub3_weight
        self.sub1_weight = args.sub1_weight
        self.sub2_weight = args.sub2_weight
        self.sub3_weight = args.sub3_weight
        self.mtl_per_batch = args.mtl_per_batch
        self.task_specific_layer = args.task_specific_layer

        # for CTC
        self.ctc_weight = min(args.ctc_weight, self.main_weight)
        self.ctc_weight_sub1 = min(args.ctc_weight_sub1, self.sub1_weight)
        self.ctc_weight_sub2 = min(args.ctc_weight_sub2, self.sub2_weight)
        self.ctc_weight_sub3 = min(args.ctc_weight_sub3, self.sub3_weight)

        # for backward decoder
        self.bwd_weight = min(args.bwd_weight, self.main_weight)
        self.bwd_weight_sub1 = min(args.bwd_weight_sub1, self.sub1_weight)
        self.bwd_weight_sub2 = min(args.bwd_weight_sub2, self.sub2_weight)
        self.bwd_weight_sub3 = min(args.bwd_weight_sub3, self.sub3_weight)
        self.fwd_weight = self.main_weight - self.bwd_weight - self.ctc_weight
        self.fwd_weight_sub1 = self.sub1_weight - self.bwd_weight_sub1 - self.ctc_weight_sub1
        self.fwd_weight_sub2 = self.sub2_weight - self.bwd_weight_sub2 - self.ctc_weight_sub2
        self.fwd_weight_sub3 = self.sub3_weight - self.bwd_weight_sub3 - self.ctc_weight_sub3
        self.twin_net_weight = args.twin_net_weight
        self.agreement_weight = args.agreement_weight
        if args.twin_net_weight > 0:
            assert 0 < self.fwd_weight < 1
            assert 0 < self.bwd_weight < 1
            assert args.mtl_per_batch
        if args.agreement_weight > 0:
            assert 0 < self.fwd_weight < 1
            assert 0 < self.bwd_weight < 1
            # assert args.mtl_per_batch

        # regularization
        self.gaussian_noise_std = args.gaussian_noise_std
        self._gaussian_noise = True if args.gaussian_noise_std > 0 and args.gaussian_noise_timing == 'constant' else False

        # Encoder
        if args.enc_type == 'transformer':
            self.enc = TransformerEncoder(
                input_dim=args.input_dim if args.input_type == 'speech' else args.emb_dim,
                nlayers=args.transformer_enc_nlayers,
                d_model=args.d_model,
                d_ff=args.d_ff,
                attn_type=args.self_attn_type,
                nheads=args.self_attn_nheads,
                dropout_in=args.dropout_in,
                dropout=args.dropout_enc,
                dropout_att=args.dropout_att,
                nstacks=args.nstacks,
                nsplices=args.nsplices,
                conv_in_channel=args.conv_in_channel,
                conv_channels=args.conv_channels,
                conv_kernel_sizes=args.conv_kernel_sizes,
                conv_strides=args.conv_strides,
                conv_poolings=args.conv_poolings,
                conv_batch_norm=args.conv_batch_norm,
                conv_bottleneck_dim=args.conv_bottleneck_dim)
        else:
            self.enc = RNNEncoder(
                input_dim=args.input_dim if args.input_type == 'speech' else args.emb_dim,
                rnn_type=args.enc_type,
                nunits=args.enc_nunits,
                nprojs=args.enc_nprojs,
                nlayers=args.enc_nlayers,
                nlayers_sub1=args.enc_nlayers_sub1,
                nlayers_sub2=args.enc_nlayers_sub2,
                nlayers_sub3=args.enc_nlayers_sub3,
                dropout_in=args.dropout_in,
                dropout=args.dropout_enc,
                subsample=[int(s) for s in args.subsample.split('_')],
                subsample_type=args.subsample_type,
                nstacks=args.nstacks,
                nsplices=args.nsplices,
                conv_in_channel=args.conv_in_channel,
                conv_channels=args.conv_channels,
                conv_kernel_sizes=args.conv_kernel_sizes,
                conv_strides=args.conv_strides,
                conv_poolings=args.conv_poolings,
                conv_batch_norm=args.conv_batch_norm,
                conv_bottleneck_dim=args.conv_bottleneck_dim,
                residual=args.enc_residual,
                add_ffl=args.enc_add_ffl,
                nin=0,
                # layer_norm=args.layer_norm,
                task_specific_layer=args.task_specific_layer)
            # NOTE: pure CNN encoder is also included

        # Bridge layer between the encoder and decoder
        self.is_bridge = False
        if args.enc_type in ['cnn', 'transformer'] or args.bridge_layer:
            self.bridge = LinearND(self.enc.output_dim, args.dec_nunits,
                                   dropout=args.dropout_enc)
            self.is_bridge = True
            if self.sub1_weight > 0:
                self.bridge_sub1 = LinearND(self.enc.output_dim, args.dec_nunits,
                                            dropout=args.dropout_enc)
            if self.sub2_weight > 0:
                self.bridge_sub2 = LinearND(self.enc.output_dim, args.dec_nunits,
                                            dropout=args.dropout_enc)
            if self.sub3_weight > 0:
                self.bridge_sub3 = LinearND(self.enc.output_dim, args.dec_nunits,
                                            dropout=args.dropout_enc)
            self.enc_nunits = args.dec_nunits

        # main task
        directions = []
        if self.fwd_weight > 0 or self.ctc_weight > 0:
            directions.append('fwd')
        if self.bwd_weight > 0:
            directions.append('bwd')
        for dir in directions:
            # Cold fusion
            if args.rnnlm_cold_fusion and dir == 'fwd':
                logger.inof('cold fusion')
                raise NotImplementedError()
                # TODO(hirofumi): cold fusion for backward RNNLM
            else:
                args.rnnlm_cold_fusion = False

            # Decoder
            dec = Decoder(
                sos=self.sos,
                eos=self.eos,
                pad=self.pad,
                blank=self.blank,
                enc_nunits=self.enc_nunits,
                attn_type=args.attn_type,
                attn_dim=args.attn_dim,
                attn_sharpening_factor=args.attn_sharpening,
                attn_sigmoid_smoothing=args.attn_sigmoid,
                attn_conv_out_channels=args.attn_conv_nchannels,
                attn_conv_kernel_size=args.attn_conv_width,
                attn_nheads=args.attn_nheads,
                rnn_type=args.dec_type,
                nunits=args.dec_nunits,
                nprojs=args.dec_nprojs,
                nlayers=args.dec_nlayers,
                loop_type=args.dec_loop_type,
                residual=args.dec_residual,
                add_ffl=args.dec_add_ffl,
                layerwise_attention=args.dec_layerwise_attention,
                emb_dim=args.emb_dim,
                tie_embedding=args.tie_embedding,
                vocab=self.vocab,
                logits_temp=args.logits_temp,
                dropout=args.dropout_dec,
                dropout_emb=args.dropout_emb,
                dropout_att=args.dropout_att,
                ss_prob=args.ss_prob,
                ss_type=args.ss_type,
                lsm_prob=args.lsm_prob,
                layer_norm=args.layer_norm,
                fl_weight=args.focal_loss_weight,
                fl_gamma=args.focal_loss_gamma,
                ctc_weight=self.ctc_weight if dir == 'fwd' else 0,
                ctc_fc_list=[int(fc) for fc in args.ctc_fc_list.split(
                    '_')] if args.ctc_fc_list is not None and len(args.ctc_fc_list) > 0 else [],
                input_feeding=args.input_feeding,
                backward=(dir == 'bwd'),
                agreement_weight=args.agreement_weight if dir == 'bwd' else 0,
                twin_net_weight=args.twin_net_weight,
                rnnlm_cold_fusion=args.rnnlm_cold_fusion,
                cold_fusion=args.cold_fusion,
                rnnlm_init=args.rnnlm_init,
                lmobj_weight=args.lmobj_weight,
                share_lm_softmax=args.share_lm_softmax,
                global_weight=self.main_weight - self.bwd_weight if dir == 'fwd' else self.bwd_weight,
                mtl_per_batch=args.mtl_per_batch)
            setattr(self, 'dec_' + dir, dec)
            # TODO(hirofumi): transformer decoder

        # sub task
        for sub in ['sub1', 'sub2', 'sub3']:
            if getattr(self, sub + '_weight') > 0:
                directions = []
                if getattr(self, 'fwd_weight_' + sub) > 0 or getattr(self, 'ctc_weight_' + sub) > 0:
                    directions.append('fwd')
                if getattr(self, 'bwd_weight_' + sub) > 0:
                    directions.append('bwd')
                for dir_sub in directions:
                    dec_sub = Decoder(
                        sos=self.sos,
                        eos=self.eos,
                        pad=self.pad,
                        blank=self.blank,
                        enc_nunits=self.enc_nunits,
                        attn_type=args.attn_type,
                        attn_dim=args.attn_dim,
                        attn_sharpening_factor=args.attn_sharpening,
                        attn_sigmoid_smoothing=args.attn_sigmoid,
                        attn_conv_out_channels=args.attn_conv_nchannels,
                        attn_conv_kernel_size=args.attn_conv_width,
                        attn_nheads=1,
                        rnn_type=args.dec_type,
                        nunits=args.dec_nunits,
                        nprojs=args.dec_nprojs,
                        nlayers=args.dec_nlayers,
                        loop_type=args.dec_loop_type,
                        residual=args.dec_residual,
                        add_ffl=args.dec_add_ffl,
                        layerwise_attention=args.dec_layerwise_attention,
                        emb_dim=args.emb_dim,
                        tie_embedding=args.tie_embedding,
                        vocab=getattr(self, 'vocab_' + sub),
                        logits_temp=args.logits_temp,
                        dropout=args.dropout_dec,
                        dropout_emb=args.dropout_emb,
                        dropout_att=args.dropout_att,
                        ss_prob=args.ss_prob,
                        ss_type=args.ss_type,
                        lsm_prob=args.lsm_prob,
                        layer_norm=args.layer_norm,
                        fl_weight=args.focal_loss_weight,
                        fl_gamma=args.focal_loss_gamma,
                        ctc_weight=getattr(self, 'ctc_weight_' + sub) if dir_sub == 'fwd' else 0,
                        ctc_fc_list=[int(fc) for fc in getattr(args, 'ctc_fc_list_' + sub).split('_')
                                     ] if getattr(args, 'ctc_fc_list_' + sub) is not None and len(getattr(args, 'ctc_fc_list_' + sub)) > 0 else [],
                        input_feeding=args.input_feeding,
                        backward=(dir_sub == 'bwd'),
                        twin_net_weight=args.twin_net_weight,
                        # rnnlm_cold_fusion=args.rnnlm_cold_fusion,
                        # cold_fusion=args.cold_fusion,
                        lmobj_weight=getattr(args, 'lmobj_weight_' + sub),
                        share_lm_softmax=args.share_lm_softmax,
                        global_weight=getattr(self, sub + '_weight'),
                        mtl_per_batch=args.mtl_per_batch)
                    setattr(self, 'dec_' + dir_sub + '_' + sub, dec_sub)

        if args.input_type == 'text':
            if args.vocab == args.vocab_sub1:
                # Share the embedding layer between input and output
                self.embed_in = dec.embed
            else:
                self.embed_in = Embedding(vocab=args.vocab_sub1,
                                          emb_dim=args.emb_dim,
                                          dropout=args.dropout_emb,
                                          ignore_index=self.pad)

        # Initialize weight matrices
        self.init_weights(args.param_init, dist=args.param_init_dist)

        # Initialize bias vectors with zero
        self.init_weights(0, dist='constant', keys=['bias'])

        # Initialize CNN layers
        self.init_weights(args.param_init, dist='xavier_uniform',
                          keys=['conv'], ignore_keys=['score'])

        # Recurrent weights are orthogonalized
        if args.rec_weight_orthogonal:
            # encoder
            if args.enc_type != 'cnn':
                self.init_weights(args.param_init, dist='orthogonal',
                                  keys=[args.enc_type, 'weight'])
            # TODO(hirofumi): in case of CNN + LSTM
            # decoder
            self.init_weights(args.param_init, dist='orthogonal',
                              keys=[args.dec_type, 'weight'])

        # Initialize bias in forget gate with 1
        self.init_forget_gate_bias_with_one()

        # Initialize bias in gating with -1 for cold fusion
        if args.rnnlm_cold_fusion:
            self.init_weights(-1, dist='constant', keys=['cf_linear_lm_gate.fc.bias'])

        # Initialize for transformer
        if args.enc_type == 'transformer':
            self.init_weights(args.param_init, dist='xavier_uniform',
                              keys=['transformer'], ignore_keys=['conv'])

    def scheduled_sampling_trigger(self):
        # main task
        directions = []
        if self.fwd_weight > 0:
            directions.append('fwd')
        if self.bwd_weight > 0:
            directions.append('bwd')
        for dir in directions:
            getattr(self, 'dec_' + dir).start_scheduled_sampling()

        # sub task
        for sub in ['sub1', 'sub2', 'sub3']:
            if getattr(self, sub + '_weight') > 0:
                directions = []
                if getattr(self, 'fwd_weight_' + sub) > 0:
                    directions.append('fwd')
                if getattr(self, 'bwd_weight_' + sub) > 0:
                    directions.append('bwd')
                for dir_sub in directions:
                    getattr(self, 'dec_' + dir_sub + '_' + sub).start_scheduled_sampling()

    def forward(self, batch, reporter=None, task='all', is_eval=False):
        """Forward computation.

        Args:
            batch (dict):
                xs (list): A list of length `[B]`, which contains arrays of size `[T, input_dim]`
                ys (list): A list of length `[B]`, which contains arrays of size `[L]`
                ys_sub1 (list): A list of lenght `[B]`, which contains arrays of size `[L_sub1]`
                ys_sub2 (list): A list of lenght `[B]`, which contains arrays of size `[L_sub2]`
                ys_sub3 (list): A list of lenght `[B]`, which contains arrays of size `[L_sub3]`
            reporter ():
            task (str): all or ys* or ys_sub1* or ys_sub2* or ys_sub3*
            is_eval (bool): the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (FloatTensor): `[1]`
            reporter ():

        """
        if is_eval:
            self.eval()
            with torch.no_grad():
                loss, observation = self._forward(batch, task)
        else:
            self.train()
            loss, observation = self._forward(batch, task)

        # Report here
        if reporter is not None:
            reporter.add(observation, is_eval)

        return loss, reporter

    def _forward(self, batch, task):
        # Encode input features
        if self.input_type == 'speech':
            if self.mtl_per_batch:
                flip = True if 'bwd' in task else False
                enc_outs, perm_ids = self.encode(batch['xs'], task, flip=flip)
            else:
                flip = True if self.bwd_weight == 1 else False
                enc_outs, perm_ids = self.encode(batch['xs'], 'all', flip=flip)
        else:
            enc_outs, perm_ids = self.encode(batch['ys_sub1'])

        observation = {}
        loss = torch.zeros((1,), dtype=torch.float32).cuda(self.device_id)

        # for the forward decoder in the main task
        if (self.fwd_weight > 0 or self.ctc_weight > 0) and task in ['all', 'ys', 'ys.ctc', 'ys.lmobj']:
            if perm_ids is None:
                ys = batch['ys'][:]  # for lmobj
            else:
                ys = [batch['ys'][:][i] for i in perm_ids]
            if task == 'ys' and self.twin_net_weight > 0:
                reverse_dec = self.dec_bwd
            else:
                reverse_dec = None
            loss_fwd, obs_fwd = self.dec_fwd(
                enc_outs['ys']['xs'], enc_outs['ys']['xlens'], ys, task, reverse_dec)
            loss += loss_fwd
            observation['loss.att'] = obs_fwd['loss_att']
            observation['loss.ctc'] = obs_fwd['loss_ctc']
            observation['loss.lmobj'] = obs_fwd['loss_lmobj']
            observation['loss.twinnet'] = obs_fwd['loss_twin']
            observation['acc.att'] = obs_fwd['acc_att']
            observation['acc.lmobj'] = obs_fwd['acc_lmobj']
            observation['ppl.att'] = obs_fwd['ppl_att']
            observation['ppl.lmobj'] = obs_fwd['ppl_lmobj']

        # for the backward decoder in the main task
        if self.bwd_weight > 0 and task in ['all', 'ys.bwd']:
            ys = [batch['ys'][:][i] for i in perm_ids]
            if self.agreement_weight > 0:
                reverse_dec = self.dec_fwd
            else:
                reverse_dec = None
            loss_bwd, obs_bwd = self.dec_bwd(
                enc_outs['ys']['xs'], enc_outs['ys']['xlens'], ys, task, reverse_dec)
            loss += loss_bwd
            observation['loss.att-bwd'] = obs_bwd['loss_att']
            observation['loss.ctc-bwd'] = obs_bwd['loss_ctc']
            observation['loss.lmobj-bwd'] = obs_bwd['loss_lmobj']
            observation['loss.agreement-bwd'] = obs_bwd['loss_agreement']
            observation['acc.att-bwd'] = obs_bwd['acc_att']
            observation['acc.lmobj-bwd'] = obs_bwd['acc_lmobj']
            observation['ppl.att-bwd'] = obs_bwd['ppl_att']
            observation['ppl.lmobj-bwd'] = obs_bwd['ppl_lmobj']

        # only fwd for sub tasks
        for sub in ['sub1', 'sub2', 'sub3']:
            # for the forward decoder in the sub tasks
            if (getattr(self, 'fwd_weight_' + sub) > 0 or getattr(self, 'ctc_weight_' + sub) > 0) and task in ['all', 'ys_' + sub, 'ys_' + sub + '.ctc', 'ys_' + sub + '.lmobj']:
                if perm_ids is None:
                    ys_sub = [batch['ys_' + sub][:][i] for i in perm_ids]  # for lmobj
                else:
                    ys_sub = batch['ys_' + sub][:]
                loss_sub, obs_fwd_sub = getattr(self, 'dec_fwd_' + sub)(
                    enc_outs['ys_' + sub]['xs'], enc_outs['ys_' + sub]['xlens'], ys_sub, task)
                loss += loss_sub
                observation['loss.att-' + sub] = obs_fwd_sub['loss_att']
                observation['loss.ctc-' + sub] = obs_fwd_sub['loss_ctc']
                observation['loss.lmobj-' + sub] = obs_fwd_sub['loss_lmobj']
                observation['acc.att-' + sub] = obs_fwd_sub['acc_att']
                observation['acc.lmobj-' + sub] = obs_fwd_sub['acc_lmobj']
                observation['ppl.att-' + sub] = obs_fwd_sub['ppl_att']
                observation['ppl.lmobj-' + sub] = obs_fwd_sub['ppl_lmobj']

                # for the backward decoder in the sub tasks
            if getattr(self, 'bwd_weight_' + sub) > 0 and task in ['all', 'ys_' + sub + '.bwd']:
                if perm_ids is None:
                    ys_sub = [batch['ys_' + sub][:][i] for i in perm_ids]  # for lmobj
                else:
                    ys_sub = batch['ys_' + sub][:]
                loss_sub, obs_bwd_sub = getattr(self, 'dec_fwd_' + sub)(
                    enc_outs['ys_' + sub]['xs'], enc_outs['ys_' + sub]['xlens'], ys_sub, task)
                loss += loss_sub
                observation['loss.att-bwd-' + sub] = obs_bwd_sub['loss_att']
                observation['loss.ctc-bwd-' + sub] = obs_bwd_sub['loss_ctc']
                observation['loss.lmobj-bwd-' + sub] = obs_bwd_sub['loss_lmobj']
                observation['acc.att-bwd-' + sub] = obs_bwd_sub['acc_att']
                observation['acc.lmobj-bwd-' + sub] = obs_bwd_sub['acc_lmobj']
                observation['ppl.att-bwd-' + sub] = obs_bwd_sub['ppl_att']
                observation['ppl.lmobj-bwd-' + sub] = obs_bwd_sub['ppl_lmobj']

        return loss, observation

    def encode(self, xs, task='all', flip=False):
        """Encode acoustic or text features.

        Args:
            xs (list): A list of length `[B]`, which contains Tensor of size `[T, input_dim]`
            task (str): all or ys* or ys_sub1* or ys_sub2* or ys_sub3*
            flip (bool): if True, flip acoustic features in the time-dimension
        Returns:
            enc_outs (dict):
            perm_ids ():

        """
        if 'lmobj' in task:
            eouts = {'ys': {'xs': None, 'xlens': None},
                     'ys_sub1': {'xs': None, 'xlens': None},
                     'ys_sub2': {'xs': None, 'xlens': None},
                     'ys_sub3': {'xs': None, 'xlens': None}}
            return eouts, None
        else:
            # Sort by lenghts in the descending order
            perm_ids = sorted(list(range(0, len(xs), 1)),
                              key=lambda i: len(xs[i]), reverse=True)
            xs = [xs[i] for i in perm_ids]
            # NOTE: must be descending order for pack_padded_sequence

            if self.input_type == 'speech':
                # Frame stacking
                if self.nstacks > 1:
                    xs = [stack_frame(x, self.nstacks, self.nskips)for x in xs]

                # Splicing
                if self.nsplices > 1:
                    xs = [splice(x, self.nsplices, self.nstacks) for x in xs]

                xlens = [len(x) for x in xs]
                # Flip acoustic features in the reverse order
                if flip:
                    xs = [torch.from_numpy(np.flip(x, axis=0).copy()).float().cuda(self.device_id) for x in xs]
                else:
                    xs = [np2tensor(x, self.device_id).float() for x in xs]
                xs = pad_list(xs, 0.0)

            elif self.input_type == 'text':
                xlens = [len(x) for x in xs]
                xs = [np2tensor(np.fromiter(x, dtype=np.int64), self.device_id).long() for x in xs]
                xs = pad_list(xs, self.pad)
                xs = self.embed_in(xs)

            enc_outs = self.enc(xs, xlens, task.split('.')[0])

            if self.main_weight < 1 and self.enc_type == 'cnn':
                for sub in ['sub1', 'sub2', 'sub3']:
                    enc_outs['ys_' + sub]['xs'] = enc_outs['ys']['xs'].clone()
                    enc_outs['ys_' + sub]['xlens'] = copy.deepcopy(enc_outs['ys']['xlens'])

            # Bridge between the encoder and decoder
            if self.main_weight > 0 and self.is_bridge and (task in ['all', 'ys']):
                enc_outs['ys']['xs'] = self.bridge(enc_outs['ys']['xs'])
            if self.sub1_weight > 0 and self.is_bridge and (task in ['all', 'ys_sub1']):
                enc_outs['ys_sub1']['xs'] = self.bridge_sub1(enc_outs['ys_sub1']['xs'])
            if self.sub2_weight > 0 and self.is_bridge and (task in ['all', 'ys_sub2']):
                enc_outs['ys_sub2']['xs'] = self.bridge_sub2(enc_outs['ys_sub2']['xs'])
            if self.sub3_weight > 0 and self.is_bridge and (task in ['all', 'ys_sub3']):
                enc_outs['ys_sub3']['xs'] = self.bridge_sub3(enc_outs['ys_sub3']['xs'])

            return enc_outs, perm_ids

    def get_ctc_posteriors(self, xs, task='ys', temperature=1, topk=None):
        self.eval()
        with torch.no_grad():
            enc_outs, perm_ids = self.encode(xs, task)
            dir = 'fwd' if self.fwd_weight >= self.bwd_weight else 'bwd'
            if task == 'ys_sub1':
                dir += '_sub1'
            elif task == 'ys_sub2':
                dir += '_sub2'

            if task == 'ys':
                assert self.ctc_weight > 0
            elif task == 'ys_sub1':
                assert self.ctc_weight_sub1 > 0
            elif task == 'ys_sub2':
                assert self.ctc_weight_sub2 > 0
            elif task == 'ys_sub3':
                assert self.ctc_weight_sub3 > 0
            ctc_probs, indices_topk = getattr(self, 'dec_' + dir).ctc_posteriors(
                enc_outs[task]['xs'], enc_outs[task]['xlens'], temperature, topk)
            return ctc_probs, indices_topk, enc_outs[task]['xlens']

    def decode(self, xs, params, nbest=1, exclude_eos=False,
               id2token=None, refs=None, task='ys', ensemble_models=[]):
        """Decoding in the inference stage.

        Args:
            xs (list): A list of length `[B]`, which contains arrays of size `[T, input_dim]`
            params (dict): hyper-parameters for decoding
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
            id2token (): converter from index to token
            refs (list): gold transcriptions to compute log likelihood
            task (str): ys* or ys_sub1* or ys_sub2* or ys_sub3*
            ensemble_models (list): list of Seq2seq classes
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aws (list): A list of length `[B]`, which contains arrays of size `[L, T]`
            perm_ids (list): A list of length `[B]`

        """
        self.eval()
        with torch.no_grad():
            if task.split('.')[0] == 'ys':
                dir = 'bwd' if self.bwd_weight > 0 and params['recog_bwd_attention'] else 'fwd'
            elif task.split('.')[0] == 'ys_sub1':
                dir = 'bwd' if self.bwd_weight_sub1 > 0 and params['recog_bwd_attention'] else 'fwd'
                dir += '_sub1'
            elif task.split('.')[0] == 'ys_sub2':
                dir = 'bwd' if self.bwd_weight_sub2 > 0 and params['recog_bwd_attention'] else 'fwd'
                dir += '_sub2'
            elif task.split('.')[0] == 'ys_sub3':
                dir = 'bwd' if self.bwd_weight_sub3 > 0 and params['recog_bwd_attention'] else 'fwd'
                dir += '_sub3'
            else:
                raise ValueError(task)

            # encode
            if self.input_type == 'speech' and self.mtl_per_batch and 'bwd' in dir:
                enc_outs, perm_ids = self.encode(xs, task, flip=True)
            else:
                enc_outs, perm_ids = self.encode(xs, task, flip=False)

            #########################
            # CTC
            #########################
            if (self.fwd_weight == 0 and self.bwd_weight == 0) or (self.ctc_weight > 0 and params['recog_ctc_weight'] == 1):
                rnnlm = None
                if params['recog_rnnlm_weight'] > 0:
                    rnnlm = getattr(self, 'rnnlm_' + dir)

                best_hyps = getattr(self, 'dec_' + dir).decode_ctc(
                    enc_outs[task]['xs'], enc_outs[task]['xlens'],
                    params['recog_beam_width'], rnnlm)
                return best_hyps, None, perm_ids

            #########################
            # Attention
            #########################
            else:
                if params['recog_beam_width'] == 1 and not params['recog_fwd_bwd_attention']:
                    best_hyps, aws = getattr(self, 'dec_' + dir).greedy(
                        enc_outs[task]['xs'], enc_outs[task]['xlens'],
                        params['recog_max_len_ratio'], exclude_eos)
                else:
                    if params['recog_ctc_weight'] > 0:
                        ctc_log_probs = self.dec_fwd.ctc_log_probs(enc_outs[task]['xs'])
                    else:
                        ctc_log_probs = None

                    # forward-backward decoding
                    if params['recog_fwd_bwd_attention']:
                        # forward decoder
                        rnnlm_fwd, rnnlm_bwd = None, None
                        if params['recog_rnnlm_weight'] > 0:
                            rnnlm_fwd = self.rnnlm_fwd
                            if params['recog_reverse_lm_rescoring']:
                                rnnlm_bwd = self.rnnlm_bwd

                        # ensemble (forward)
                        ensemble_eouts_fwd = []
                        ensemble_elens_fwd = []
                        ensemble_decoders_fwd = []
                        if len(ensemble_models) > 0:
                            for i_e, model in enumerate(ensemble_models):
                                enc_outs_e_fwd, _ = model.encode(xs, task, flip=False)
                                ensemble_eouts_fwd += [enc_outs_e_fwd[task]['xs']]
                                ensemble_elens_fwd += [enc_outs_e_fwd[task]['xlens']]
                                ensemble_decoders_fwd += [model.dec_fwd]
                                # NOTE: only support for the main task now

                        nbest_hyps_fwd, aws_fwd, scores_fwd, scores_cp_fwd = self.dec_fwd.beam_search(
                            enc_outs[task]['xs'], enc_outs[task]['xlens'],
                            params, rnnlm_fwd, rnnlm_bwd, ctc_log_probs,
                            params['recog_beam_width'], False, id2token, refs,
                            ensemble_eouts_fwd, ensemble_elens_fwd, ensemble_decoders_fwd)

                        # backward decoder
                        rnnlm_bwd, rnnlm_fwd = None, None
                        if params['recog_rnnlm_weight'] > 0:
                            rnnlm_bwd = self.rnnlm_bwd
                            if params['recog_reverse_lm_rescoring']:
                                rnnlm_fwd = self.rnnlm_fwd

                        # ensemble (backward)
                        ensemble_eouts_bwd = []
                        ensemble_elens_bwd = []
                        ensemble_decoders_bwd = []
                        if len(ensemble_models) > 0:
                            for i_e, model in enumerate(ensemble_models):
                                if self.input_type == 'speech' and self.mtl_per_batch:
                                    enc_outs_e_bwd, _ = model.encode(xs, task, flip=True)
                                else:
                                    enc_outs_e_bwd, _ = model.encode(xs, task, flip=False)
                                ensemble_eouts_bwd += [enc_outs_e_bwd[task]['xs']]
                                ensemble_elens_bwd += [enc_outs_e_bwd[task]['xlens']]
                                ensemble_decoders_bwd += [model.dec_bwd]
                                # NOTE: only support for the main task now
                                # TODO(hirofumi): merge with the forward for the efficiency

                        flip = False
                        if self.input_type == 'speech' and self.mtl_per_batch:
                            flip = True
                            enc_outs_bwd, _ = self.encode(xs, task, flip=True)
                        else:
                            enc_outs_bwd = enc_outs
                        nbest_hyps_bwd, aws_bwd, scores_bwd, scores_cp_bwd = self.dec_bwd.beam_search(
                            enc_outs_bwd[task]['xs'], enc_outs[task]['xlens'],
                            params, rnnlm_bwd, rnnlm_fwd, ctc_log_probs,
                            params['recog_beam_width'], False, id2token, refs,
                            ensemble_eouts_bwd, ensemble_elens_bwd, ensemble_decoders_bwd)

                        # forward-backward attention
                        best_hyps = fwd_bwd_attention(
                            nbest_hyps_fwd, aws_fwd, scores_fwd, scores_cp_fwd,
                            nbest_hyps_bwd, aws_bwd, scores_bwd, scores_cp_bwd,
                            flip, self.eos, params['recog_gnmt_decoding'], params['recog_length_penalty'],
                            id2token, refs)
                        aws = None
                    else:
                        # ensemble
                        ensemble_eouts = []
                        ensemble_elens = []
                        ensemble_decoders = []
                        if len(ensemble_models) > 0:
                            for i_e, model in enumerate(ensemble_models):
                                if model.input_type == 'speech' and model.mtl_per_batch and 'bwd' in dir:
                                    enc_outs_e, _ = model.encode(xs, task, flip=True)
                                else:
                                    enc_outs_e, _ = model.encode(xs, task, flip=False)
                                ensemble_eouts += [enc_outs_e[task]['xs']]
                                ensemble_elens += [enc_outs_e[task]['xlens']]
                                ensemble_decoders += [getattr(model, 'dec_' + dir)]
                                # NOTE: only support for the main task now

                        rnnlm, rnnlm_rev = None, None
                        if params['recog_rnnlm_weight'] > 0:
                            rnnlm = getattr(self, 'rnnlm_' + dir)
                            if params['recog_reverse_lm_rescoring']:
                                if dir == 'fwd':
                                    rnnlm_rev = self.rnnlm_bwd
                                else:
                                    raise NotImplementedError()

                        nbest_hyps, aws, scores, _ = getattr(self, 'dec_' + dir).beam_search(
                            enc_outs[task]['xs'], enc_outs[task]['xlens'],
                            params, rnnlm, rnnlm_rev, ctc_log_probs,
                            nbest, exclude_eos, id2token, refs,
                            ensemble_eouts, ensemble_elens, ensemble_decoders)

                        if nbest == 1:
                            best_hyps = [hyp[0] for hyp in nbest_hyps]
                            aws = [aw[0] for aw in aws]
                        else:
                            return nbest_hyps, aws, scores, perm_ids
                        # NOTE: nbest >= 2 is used for MWER training only

                return best_hyps, aws, perm_ids


def fwd_bwd_attention(nbest_hyps_fwd, aws_fwd, scores_fwd, scores_cp_fwd,
                      nbest_hyps_bwd, aws_bwd, scores_bwd, scores_cp_bwd,
                      flip, eos, gnmt_decoding, lp_weight, id2token=None, refs=None):
    """Forward-backward joint decoding.

    Args:
        nbest_hyps_fwd (list): A list of length `[B]`, which contains list of n hypotheses
        aws_fwd (list): A list of length `[B]`, which contains arrays of size `[L, T]`
        scores_fwd (list):
        scores_cp_fwd (list):
        nbest_hyps_bwd (list):
        aws_bwd (list):
        scores_bwd (list):
        scores_cp_bwd (list):
        flip (bool):
        eos (int):
        gnmt_decoding ():
        lp_weight ():
        id2token (): converter from index to token
        refs ():
    Returns:

    """
    logger = logging.getLogger("decoding")
    bs = len(nbest_hyps_fwd)
    nbest = len(nbest_hyps_fwd[0])

    best_hyps = []
    for b in range(bs):
        max_time = len(aws_fwd[b][0])

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
                        if flip:
                            t_prev = max_time - aws_bwd[b][n_b][i_b + 1].argmax(-1).item()
                            t_curr = aws_fwd[b][n_f][i_f].argmax(-1).item()
                            t_next = max_time - aws_bwd[b][n_b][i_b - 1].argmax(-1).item()
                        else:
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
                            # if gnmt_decoding:
                            #     lp = (math.pow(5 + len(nbest_hyps_fwd[b][n_f][:i_f + 1]) + len(nbest_hyps_bwd[b][n_b][i_b + 1:]),
                            #                    lp_weight)) / math.pow(6, lp_weight)
                            #     new_score /= lp

                            # coverage
                            # score_cp_curr_fwd = scores_cp_fwd[b][n_f][i_f] - scores_cp_fwd[b][n_f][i_f - 1]
                            # score_cp_curr_bwd = scores_cp_bwd[b][n_b][i_b] - scores_cp_bwd[b][n_b][i_b + 1]
                            # score_cp_curr = max(score_cp_curr_fwd, score_cp_curr_bwd)
                            # new_score += scores_cp_fwd[b][n_f][i_f - 1] + scores_cp_bwd[b][n_b][i_b + 1] + score_cp_curr
                            merged.append({'hyp': new_hyp, 'score': new_score})

                            logger.info('time matching')
                            if id2token is not None:
                                if refs is not None:
                                    logger.info('Ref: %s' % refs[b].lower())
                                logger.info('hyp (fwd): %s' % id2token(nbest_hyps_fwd[b][n_f]))
                                logger.info('hyp (bwd): %s' % id2token(nbest_hyps_bwd[b][n_b]))
                                logger.info('hyp (fwd-bwd): %s' % id2token(new_hyp))
                            logger.info('log prob (fwd): %.3f' % scores_fwd[b][n_f][-1])
                            logger.info('log prob (bwd): %.3f' % scores_bwd[b][n_b][0])
                            logger.info('log prob (fwd-bwd): %.3f' % new_score)

        merged = sorted(merged, key=lambda x: x['score'], reverse=True)
        best_hyps.append(merged[0]['hyp'])

    return best_hyps
