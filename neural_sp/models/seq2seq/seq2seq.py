#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Attention-based RNN sequence-to-sequence model (including CTC)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch

from neural_sp.bin.train_utils import load_checkpoint
from neural_sp.models.base import ModelBase
from neural_sp.models.model_utils import Embedding
from neural_sp.models.model_utils import LinearND
from neural_sp.models.lm.brnnlm import BRNNLM
from neural_sp.models.lm.rnnlm import RNNLM
from neural_sp.models.seq2seq.decoders.fwd_bwd_attention import fwd_bwd_attention
from neural_sp.models.seq2seq.decoders.rnn import RNNDecoder
from neural_sp.models.seq2seq.decoders.transformer import TransformerDecoder
from neural_sp.models.seq2seq.encoders.frame_stacking import stack_frame
from neural_sp.models.seq2seq.encoders.rnn import RNNEncoder
from neural_sp.models.seq2seq.encoders.sequence_summary import SequenceSummaryNetwork
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
        self.n_stacks = args.n_stacks
        self.n_skips = args.n_skips
        self.n_splices = args.n_splices
        self.enc_type = args.enc_type
        self.enc_n_units = args.enc_n_units
        if args.enc_type in ['blstm', 'bgru']:
            self.enc_n_units *= 2
        self.bridge_layer = args.bridge_layer

        # for OOV resolution
        self.enc_n_layers = args.enc_n_layers
        self.enc_n_layers_sub1 = args.enc_n_layers_sub1
        self.subsample = [int(s) for s in args.subsample.split('_')]

        # for attention layer
        self.attn_n_heads = args.attn_n_heads

        # for decoder
        self.vocab = args.vocab
        self.vocab_sub1 = args.vocab_sub1
        self.vocab_sub2 = args.vocab_sub2
        self.blank = 0
        self.unk = 1
        self.eos = 2
        self.pad = 3
        # NOTE: reserved in advance

        # for the sub tasks
        self.main_weight = 1 - args.sub1_weight - args.sub2_weight
        self.sub1_weight = args.sub1_weight
        self.sub2_weight = args.sub2_weight
        self.mtl_per_batch = args.mtl_per_batch
        self.task_specific_layer = args.task_specific_layer

        # for CTC
        self.ctc_weight = min(args.ctc_weight, self.main_weight)
        self.ctc_weight_sub1 = min(args.ctc_weight_sub1, self.sub1_weight)
        self.ctc_weight_sub2 = min(args.ctc_weight_sub2, self.sub2_weight)

        # for backward decoder
        self.bwd_weight = min(args.bwd_weight, self.main_weight)
        self.fwd_weight = self.main_weight - self.bwd_weight - self.ctc_weight
        self.fwd_weight_sub1 = self.sub1_weight - self.ctc_weight_sub1
        self.fwd_weight_sub2 = self.sub2_weight - self.ctc_weight_sub2

        # Feature extraction
        self.ssn = None
        if args.sequence_summary_network:
            assert args.input_type == 'speech'
            self.ssn = SequenceSummaryNetwork(args.input_dim,
                                              n_units=512,
                                              n_layers=3,
                                              bottleneck_dim=100,
                                              dropout=0)

        # Encoder
        if args.enc_type == 'transformer':
            self.enc = TransformerEncoder(
                input_dim=args.input_dim if args.input_type == 'speech' else args.emb_dim,
                attn_type=args.transformer_attn_type,
                attn_n_heads=args.transformer_attn_n_heads,
                n_layers=args.transformer_enc_n_layers,
                d_model=args.d_model,
                d_ff=args.d_ff,
                # pe_type=args.pe_type,
                pe_type=False,
                dropout_in=args.dropout_in,
                dropout=args.dropout_enc,
                dropout_att=args.dropout_att,
                layer_norm_eps=args.layer_norm_eps,
                n_stacks=args.n_stacks,
                n_splices=args.n_splices,
                conv_in_channel=args.conv_in_channel,
                conv_channels=args.conv_channels,
                conv_kernel_sizes=args.conv_kernel_sizes,
                conv_strides=args.conv_strides,
                conv_poolings=args.conv_poolings,
                conv_batch_norm=args.conv_batch_norm,
                conv_residual=args.conv_residual,
                conv_bottleneck_dim=args.conv_bottleneck_dim)
        else:
            self.enc = RNNEncoder(
                input_dim=args.input_dim if args.input_type == 'speech' else args.emb_dim,
                rnn_type=args.enc_type,
                n_units=args.enc_n_units,
                n_projs=args.enc_n_projs,
                n_layers=args.enc_n_layers,
                n_layers_sub1=args.enc_n_layers_sub1,
                n_layers_sub2=args.enc_n_layers_sub2,
                dropout_in=args.dropout_in,
                dropout=args.dropout_enc,
                subsample=list(map(int, args.subsample.split('_'))) +
                [1] * (args.enc_n_layers - len(args.subsample.split('_'))),
                subsample_type=args.subsample_type,
                n_stacks=args.n_stacks,
                n_splices=args.n_splices,
                conv_in_channel=args.conv_in_channel,
                conv_channels=args.conv_channels,
                conv_kernel_sizes=args.conv_kernel_sizes,
                conv_strides=args.conv_strides,
                conv_poolings=args.conv_poolings,
                conv_batch_norm=args.conv_batch_norm,
                conv_residual=args.conv_residual,
                conv_bottleneck_dim=args.conv_bottleneck_dim,
                residual=args.enc_residual,
                nin=0,
                # layer_norm=args.layer_norm,
                task_specific_layer=args.task_specific_layer)
            # NOTE: pure CNN/TDS encoders are also included

        if args.freeze_encoder:
            for p in self.enc.parameters():
                p.requires_grad = False

        # Bridge layer between the encoder and decoder
        self.is_bridge = False
        if args.enc_type in ['cnn', 'tds', 'transformer'] or args.dec_type == 'transformer' or args.bridge_layer:
            self.bridge = LinearND(self.enc.output_dim,
                                   args.d_model if args.dec_type == 'transformer' else args.dec_n_units,
                                   dropout=args.dropout_enc)
            self.is_bridge = True
            if self.sub1_weight > 0:
                self.bridge_sub1 = LinearND(self.enc.output_dim, args.dec_n_units,
                                            dropout=args.dropout_enc)
            if self.sub2_weight > 0:
                self.bridge_sub2 = LinearND(self.enc.output_dim, args.dec_n_units,
                                            dropout=args.dropout_enc)
            self.enc_n_units = args.dec_n_units

        # main task
        directions = []
        if self.fwd_weight > 0 or self.ctc_weight > 0:
            directions.append('fwd')
        if self.bwd_weight > 0:
            directions.append('bwd')
        for dir in directions:
            # Cold fusion
            if args.lm_fusion and dir == 'fwd':
                if 'bi' in args.lm_fusion_type:
                    lm = BRNNLM(args.lm_conf)
                else:
                    lm = RNNLM(args.lm_conf)
                    lm, _ = load_checkpoint(lm, args.lm_fusion)
            else:
                args.lm_conf = False
                lm = None
            # TODO(hirofumi): cold fusion for backward RNNLM

            # Decoder
            if args.dec_type == 'transformer':
                dec = TransformerDecoder(
                    eos=self.eos,
                    unk=self.unk,
                    pad=self.pad,
                    blank=self.blank,
                    enc_n_units=args.d_model,
                    attn_type=args.transformer_attn_type,
                    attn_n_heads=args.transformer_attn_n_heads,
                    n_layers=args.transformer_dec_n_layers,
                    d_model=args.d_model,
                    d_ff=args.d_ff,
                    pe_type=args.pe_type,
                    tie_embedding=args.tie_embedding,
                    vocab=self.vocab,
                    dropout=args.dropout_dec,
                    dropout_emb=args.dropout_emb,
                    dropout_att=args.dropout_att,
                    lsm_prob=args.lsm_prob,
                    layer_norm_eps=args.layer_norm_eps,
                    ctc_weight=self.ctc_weight if dir == 'fwd' else 0,
                    ctc_fc_list=[int(fc) for fc in args.ctc_fc_list.split(
                        '_')] if args.ctc_fc_list is not None and len(args.ctc_fc_list) > 0 else [],
                    backward=(dir == 'bwd'),
                    global_weight=self.main_weight - self.bwd_weight if dir == 'fwd' else self.bwd_weight,
                    mtl_per_batch=args.mtl_per_batch)
            else:
                dec = RNNDecoder(
                    eos=self.eos,
                    unk=self.unk,
                    pad=self.pad,
                    blank=self.blank,
                    enc_n_units=self.enc_n_units,
                    attn_type=args.attn_type,
                    attn_dim=args.attn_dim,
                    attn_sharpening_factor=args.attn_sharpening,
                    attn_sigmoid_smoothing=args.attn_sigmoid,
                    attn_conv_out_channels=args.attn_conv_n_channels,
                    attn_conv_kernel_size=args.attn_conv_width,
                    attn_n_heads=args.attn_n_heads,
                    rnn_type=args.dec_type,
                    n_units=args.dec_n_units,
                    n_projs=args.dec_n_projs,
                    n_layers=args.dec_n_layers,
                    loop_type=args.dec_loop_type,
                    residual=args.dec_residual,
                    bottleneck_dim=args.dec_bottleneck_dim,
                    emb_dim=args.emb_dim,
                    tie_embedding=args.tie_embedding,
                    vocab=self.vocab,
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
                    # lm=args.lm_conf,
                    lm=lm,  # TODO(hirofumi): load RNNLM in the model init.
                    lm_fusion_type=args.lm_fusion_type,
                    contextualize=args.contextualize,
                    lm_init=args.lm_init,
                    lmobj_weight=args.lmobj_weight,
                    share_lm_softmax=args.share_lm_softmax,
                    global_weight=self.main_weight - self.bwd_weight if dir == 'fwd' else self.bwd_weight,
                    mtl_per_batch=args.mtl_per_batch,
                    adaptive_softmax=args.adaptive_softmax)
            setattr(self, 'dec_' + dir, dec)

        # sub task
        for sub in ['sub1', 'sub2']:
            if getattr(self, sub + '_weight') > 0:
                if args.dec_type == 'transformer':
                    raise NotImplementedError
                else:
                    dec_sub = RNNDecoder(
                        eos=self.eos,
                        unk=self.unk,
                        pad=self.pad,
                        blank=self.blank,
                        enc_n_units=self.enc_n_units,
                        attn_type=args.attn_type,
                        attn_dim=args.attn_dim,
                        attn_sharpening_factor=args.attn_sharpening,
                        attn_sigmoid_smoothing=args.attn_sigmoid,
                        attn_conv_out_channels=args.attn_conv_n_channels,
                        attn_conv_kernel_size=args.attn_conv_width,
                        attn_n_heads=1,
                        rnn_type=args.dec_type,
                        n_units=args.dec_n_units,
                        n_projs=args.dec_n_projs,
                        n_layers=args.dec_n_layers,
                        loop_type=args.dec_loop_type,
                        residual=args.dec_residual,
                        bottleneck_dim=args.dec_bottleneck_dim,
                        emb_dim=args.emb_dim,
                        tie_embedding=args.tie_embedding,
                        vocab=getattr(self, 'vocab_' + sub),
                        dropout=args.dropout_dec,
                        dropout_emb=args.dropout_emb,
                        dropout_att=args.dropout_att,
                        ss_prob=args.ss_prob,
                        ss_type=args.ss_type,
                        lsm_prob=args.lsm_prob,
                        layer_norm=args.layer_norm,
                        fl_weight=args.focal_loss_weight,
                        fl_gamma=args.focal_loss_gamma,
                        ctc_weight=getattr(self, 'ctc_weight_' + sub),
                        ctc_fc_list=[int(fc) for fc in getattr(args, 'ctc_fc_list_' + sub).split('_')
                                     ] if getattr(args, 'ctc_fc_list_' + sub) is not None and len(getattr(args, 'ctc_fc_list_' + sub)) > 0 else [],
                        input_feeding=args.input_feeding,
                        backward=False,
                        lm=None,
                        lm_fusion_type='',
                        contextualize='',
                        lm_init=None,
                        lmobj_weight=getattr(args, 'lmobj_weight_' + sub),
                        share_lm_softmax=args.share_lm_softmax,
                        global_weight=getattr(self, sub + '_weight'),
                        mtl_per_batch=args.mtl_per_batch)
                setattr(self, 'dec_fwd_' + sub, dec_sub)

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
        if args.enc_type == 'transformer':
            self.reset_parameters(args.param_init, dist='xavier_uniform',
                                  keys=['enc'], ignore_keys=['score', 'embed_in'])
            self.reset_parameters(args.d_model**-0.5, dist='normal',
                                  keys=['embed_in'])
        else:
            self.reset_parameters(args.param_init, dist=args.param_init_dist,
                                  keys=['enc'], ignore_keys=['conv'])

        if args.dec_type == 'transformer':
            self.reset_parameters(args.param_init, dist='xavier_uniform',
                                  keys=['dec'], ignore_keys=['score', 'embed'])
            self.reset_parameters(args.d_model**-0.5, dist='normal',
                                  keys=['embed'])
        else:
            self.reset_parameters(args.param_init, dist=args.param_init_dist,
                                  keys=['dec'])

        # Initialize bias vectors with zero
        self.reset_parameters(0, dist='constant', keys=['bias'])

        # Initialize CNN layers
        self.reset_parameters(args.param_init,
                              dist='xavier_uniform',
                              #   dist='kaiming_uniform',
                              keys=['conv'], ignore_keys=['score'])

        # Recurrent weights are orthogonalized
        if args.rec_weight_orthogonal:
            if args.enc_type not in ['cnn', 'tds']:
                self.reset_parameters(args.param_init, dist='orthogonal',
                                      keys=[args.enc_type, 'weight'])
            # TODO(hirofumi): in case of CNN + LSTM
            self.reset_parameters(args.param_init, dist='orthogonal',
                                  keys=[args.dec_type, 'weight'])

        # Initialize bias in forget gate with 1
        self.init_forget_gate_bias_with_one()

        # Initialize bias in gating with -1 for cold fusion
        if args.lm_fusion:
            self.reset_parameters(-1, dist='constant', keys=['linear_lm_gate.fc.bias'])

        if args.lm_fusion_type == 'deep' and args.lm_fusion:
            for n, p in self.named_parameters():
                if 'output' in n or 'output_bn' in n or 'linear' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

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
        for sub in ['sub1', 'sub2']:
            if getattr(self, sub + '_weight') > 0:
                directions = []
                if getattr(self, 'fwd_weight_' + sub) > 0:
                    directions.append('fwd')
                for dir_sub in directions:
                    getattr(self, 'dec_' + dir_sub + '_' + sub).start_scheduled_sampling()

    def forward(self, batch, reporter=None, task='all', is_eval=False):
        """Forward computation.

        Args:
            batch (dict):
                xs (list): input data of size `[T, input_dim]`
                xlens (list): lengths of each element in xs
                ys (list): reference labels in the main task of size `[L]`
                ys_sub1 (list): reference labels in the 1st auxiliary task of size `[L_sub1]`
                ys_sub2 (list): reference labels in the 2nd auxiliary task of size `[L_sub2]`
                utt_ids (list): name of utterances
                speakers (list): name of speakers
            reporter ():
            task (str): all or ys* or ys_sub*
            is_eval (bool): the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (FloatTensor): `[1]`
            reporter ():

        """
        if is_eval:
            self.eval()
            with torch.no_grad():
                loss, reporter = self._forward(batch, task, reporter)
        else:
            self.train()
            loss, reporter = self._forward(batch, task, reporter)

        return loss, reporter

    def _forward(self, batch, task, reporter):
        # Encode input features
        if self.input_type == 'speech':
            if self.mtl_per_batch:
                flip = True if 'bwd' in task else False
                enc_outs = self.encode(batch['xs'], task, flip=flip)
            else:
                flip = True if self.bwd_weight == 1 else False
                enc_outs = self.encode(batch['xs'], 'all', flip=flip)
        else:
            enc_outs = self.encode(batch['ys_sub1'])

        observation = {}
        loss = torch.zeros((1,), dtype=torch.float32).cuda(self.device_id)

        # for the forward decoder in the main task
        if (self.fwd_weight > 0 or self.ctc_weight > 0) and task in ['all', 'ys', 'ys.ctc', 'ys.lmobj', 'ys.lm']:
            loss_fwd, obs_fwd = self.dec_fwd(enc_outs['ys']['xs'], enc_outs['ys']
                                             ['xlens'], batch['ys'], task, batch['ys_hist'])
            loss += loss_fwd
            observation['loss.att'] = obs_fwd['loss_att']
            observation['loss.ctc'] = obs_fwd['loss_ctc']
            observation['loss.lmobj'] = obs_fwd['loss_lmobj']
            observation['acc.att'] = obs_fwd['acc_att']
            observation['acc.lmobj'] = obs_fwd['acc_lmobj']
            observation['ppl.att'] = obs_fwd['ppl_att']
            observation['ppl.lmobj'] = obs_fwd['ppl_lmobj']

        # for the backward decoder in the main task
        if self.bwd_weight > 0 and task in ['all', 'ys.bwd']:
            loss_bwd, obs_bwd = self.dec_bwd(enc_outs['ys']['xs'], enc_outs['ys']['xlens'], batch['ys'], task)
            loss += loss_bwd
            observation['loss.att-bwd'] = obs_bwd['loss_att']
            observation['loss.ctc-bwd'] = obs_bwd['loss_ctc']
            observation['loss.lmobj-bwd'] = obs_bwd['loss_lmobj']
            observation['acc.att-bwd'] = obs_bwd['acc_att']
            observation['acc.lmobj-bwd'] = obs_bwd['acc_lmobj']
            observation['ppl.att-bwd'] = obs_bwd['ppl_att']
            observation['ppl.lmobj-bwd'] = obs_bwd['ppl_lmobj']

        # only fwd for sub tasks
        for sub in ['sub1', 'sub2']:
            # for the forward decoder in the sub tasks
            if (getattr(self, 'fwd_weight_' + sub) > 0 or getattr(self, 'ctc_weight_' + sub) > 0) and task in ['all', 'ys_' + sub, 'ys_' + sub + '.ctc', 'ys_' + sub + '.lmobj']:
                loss_sub, obs_fwd_sub = getattr(self, 'dec_fwd_' + sub)(
                    enc_outs['ys_' + sub]['xs'], enc_outs['ys_' + sub]['xlens'], batch['ys_' + sub], task)
                loss += loss_sub
                observation['loss.att-' + sub] = obs_fwd_sub['loss_att']
                observation['loss.ctc-' + sub] = obs_fwd_sub['loss_ctc']
                observation['loss.lmobj-' + sub] = obs_fwd_sub['loss_lmobj']
                observation['acc.att-' + sub] = obs_fwd_sub['acc_att']
                observation['acc.lmobj-' + sub] = obs_fwd_sub['acc_lmobj']
                observation['ppl.att-' + sub] = obs_fwd_sub['ppl_att']
                observation['ppl.lmobj-' + sub] = obs_fwd_sub['ppl_lmobj']

        if reporter is not None:
            is_eval = not self.training
            reporter.add(observation, is_eval)

        return loss, reporter

    def encode(self, xs, task='all', flip=False):
        """Encode acoustic or text features.

        Args:
            xs (list): A list of length `[B]`, which contains Tensor of size `[T, input_dim]`
            task (str): all or ys* or ys_sub1* or ys_sub2*
            flip (bool): if True, flip acoustic features in the time-dimension
        Returns:
            enc_outs (dict):

        """
        if 'lmobj' in task:
            eouts = {'ys': {'xs': None, 'xlens': None},
                     'ys_sub1': {'xs': None, 'xlens': None},
                     'ys_sub2': {'xs': None, 'xlens': None}}
            return eouts
        else:
            if self.input_type == 'speech':
                # Frame stacking
                if self.n_stacks > 1:
                    xs = [stack_frame(x, self.n_stacks, self.n_skips)for x in xs]

                # Splicing
                if self.n_splices > 1:
                    xs = [splice(x, self.n_splices, self.n_stacks) for x in xs]

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

            # sequence summary network
            if self.ssn is not None:
                xs += self.ssn(xs, xlens)

            # encoder
            enc_outs = self.enc(xs, xlens, task.split('.')[0])

            if self.main_weight < 1 and self.enc_type in ['cnn', 'tds']:
                for sub in ['sub1', 'sub2']:
                    enc_outs['ys_' + sub]['xs'] = enc_outs['ys']['xs'].clone()
                    enc_outs['ys_' + sub]['xlens'] = enc_outs['ys']['xlens'][:]

            # Bridge between the encoder and decoder
            if self.main_weight > 0 and self.is_bridge and (task in ['all', 'ys']):
                enc_outs['ys']['xs'] = self.bridge(enc_outs['ys']['xs'])
            if self.sub1_weight > 0 and self.is_bridge and (task in ['all', 'ys_sub1']):
                enc_outs['ys_sub1']['xs'] = self.bridge_sub1(enc_outs['ys_sub1']['xs'])
            if self.sub2_weight > 0 and self.is_bridge and (task in ['all', 'ys_sub2']):
                enc_outs['ys_sub2']['xs'] = self.bridge_sub2(enc_outs['ys_sub2']['xs'])

            return enc_outs

    def get_ctc_posteriors(self, xs, task='ys', temperature=1, topk=None):
        self.eval()
        with torch.no_grad():
            enc_outs = self.encode(xs, task)
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
            ctc_probs, indices_topk = getattr(self, 'dec_' + dir).ctc_posteriors(
                enc_outs[task]['xs'], enc_outs[task]['xlens'], temperature, topk)
            return ctc_probs, indices_topk, enc_outs[task]['xlens']

    def decode(self, xs, params, idx2token, nbest=1, exclude_eos=False,
               refs_id=None, refs_text=None, utt_ids=None, speakers=None,
               task='ys', ensemble_models=[]):
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
                lm_weight (float): the weight of RNNLM score
                resolving_unk (bool): not used (to make compatible)
                fwd_bwd_attention (bool):
            idx2token (): converter from index to token
            nbest (int):
            exclude_eos (bool): exclude <eos> from best_hyps_id
            refs_id (list): gold token IDs to compute log likelihood
            refs_text (list): gold transcriptions
            utt_ids (list):
            speakers (list):
            task (str): ys* or ys_sub1* or ys_sub2*
            ensemble_models (list): list of Seq2seq classes
        Returns:
            best_hyps_id (list): A list of length `[B]`, which contains arrays of size `[L]`
            aws (list): A list of length `[B]`, which contains arrays of size `[L, T, n_heads]`

        """
        self.eval()
        with torch.no_grad():
            if task.split('.')[0] == 'ys':
                dir = 'bwd' if self.bwd_weight > 0 and params['recog_bwd_attention'] else 'fwd'
            elif task.split('.')[0] == 'ys_sub1':
                dir = 'fwd_sub1'
            elif task.split('.')[0] == 'ys_sub2':
                dir = 'fwd_sub2'
            else:
                raise ValueError(task)

            # encode
            if self.input_type == 'speech' and self.mtl_per_batch and 'bwd' in dir:
                enc_outs = self.encode(xs, task, flip=True)
            else:
                enc_outs = self.encode(xs, task, flip=False)

            #########################
            # CTC
            #########################
            if (self.fwd_weight == 0 and self.bwd_weight == 0) or (self.ctc_weight > 0 and params['recog_ctc_weight'] == 1):
                lm = None
                if params['recog_lm_weight'] > 0 and hasattr(self, 'lm_fwd') and self.lm_fwd is not None:
                    lm = getattr(self, 'lm_' + dir)

                best_hyps_id = getattr(self, 'dec_' + dir).decode_ctc(
                    enc_outs[task]['xs'], enc_outs[task]['xlens'],
                    params['recog_beam_width'], lm, params['recog_lm_weight'])
                return best_hyps_id, None, (None, None)

            #########################
            # Attention
            #########################
            else:
                cache_info = (None, None)

                if params['recog_beam_width'] == 1 and not params['recog_fwd_bwd_attention']:
                    best_hyps_id, aws = getattr(self, 'dec_' + dir).greedy(
                        enc_outs[task]['xs'], enc_outs[task]['xlens'],
                        params['recog_max_len_ratio'], exclude_eos, idx2token, refs_id,
                        speakers, params['recog_oracle'])
                else:
                    assert params['recog_batch_size'] == 1

                    if params['recog_ctc_weight'] > 0:
                        ctc_log_probs = self.dec_fwd.ctc_log_probs(enc_outs[task]['xs'])
                    else:
                        ctc_log_probs = None

                    # forward-backward decoding
                    if params['recog_fwd_bwd_attention']:
                        # forward decoder
                        lm_fwd, lm_bwd = None, None
                        if params['recog_lm_weight'] > 0 and hasattr(self, 'lm_fwd') and self.lm_fwd is not None:
                            lm_fwd = self.lm_fwd
                            if params['recog_reverse_lm_rescoring'] and hasattr(self, 'lm_bwd') and self.lm_bwd is not None:
                                lm_bwd = self.lm_bwd

                        # ensemble (forward)
                        ensemble_eouts_fwd = []
                        ensemble_elens_fwd = []
                        ensemble_decs_fwd = []
                        if len(ensemble_models) > 0:
                            for i_e, model in enumerate(ensemble_models):
                                enc_outs_e_fwd = model.encode(xs, task, flip=False)
                                ensemble_eouts_fwd += [enc_outs_e_fwd[task]['xs']]
                                ensemble_elens_fwd += [enc_outs_e_fwd[task]['xlens']]
                                ensemble_decs_fwd += [model.dec_fwd]
                                # NOTE: only support for the main task now

                        nbest_hyps_id_fwd, aws_fwd, scores_fwd, cache_info = self.dec_fwd.beam_search(
                            enc_outs[task]['xs'], enc_outs[task]['xlens'],
                            params, idx2token, lm_fwd, lm_bwd, ctc_log_probs,
                            params['recog_beam_width'], False, refs_id, utt_ids, speakers,
                            ensemble_eouts_fwd, ensemble_elens_fwd, ensemble_decs_fwd)

                        # backward decoder
                        lm_bwd, lm_fwd = None, None
                        if params['recog_lm_weight'] > 0 and hasattr(self, 'lm_bwd') and self.lm_bwd is not None:
                            lm_bwd = self.lm_bwd
                            if params['recog_reverse_lm_rescoring'] and hasattr(self, 'lm_fwd') and self.lm_fwd is not None:
                                lm_fwd = self.lm_fwd

                        # ensemble (backward)
                        ensemble_eouts_bwd = []
                        ensemble_elens_bwd = []
                        ensemble_decs_bwd = []
                        if len(ensemble_models) > 0:
                            for i_e, model in enumerate(ensemble_models):
                                if self.input_type == 'speech' and self.mtl_per_batch:
                                    enc_outs_e_bwd = model.encode(xs, task, flip=True)
                                else:
                                    enc_outs_e_bwd = model.encode(xs, task, flip=False)
                                ensemble_eouts_bwd += [enc_outs_e_bwd[task]['xs']]
                                ensemble_elens_bwd += [enc_outs_e_bwd[task]['xlens']]
                                ensemble_decs_bwd += [model.dec_bwd]
                                # NOTE: only support for the main task now
                                # TODO(hirofumi): merge with the forward for the efficiency

                        flip = False
                        if self.input_type == 'speech' and self.mtl_per_batch:
                            flip = True
                            enc_outs_bwd = self.encode(xs, task, flip=True)
                        else:
                            enc_outs_bwd = enc_outs
                        nbest_hyps_id_bwd, aws_bwd, scores_bwd, _ = self.dec_bwd.beam_search(
                            enc_outs_bwd[task]['xs'], enc_outs[task]['xlens'],
                            params, idx2token, lm_bwd, lm_fwd, ctc_log_probs,
                            params['recog_beam_width'], False, refs_id, utt_ids, speakers,
                            ensemble_eouts_bwd, ensemble_elens_bwd, ensemble_decs_bwd)

                        # forward-backward attention
                        best_hyps_id = fwd_bwd_attention(
                            nbest_hyps_id_fwd, aws_fwd, scores_fwd,
                            nbest_hyps_id_bwd, aws_bwd, scores_bwd,
                            flip, self.eos, params['recog_gnmt_decoding'], params['recog_length_penalty'],
                            idx2token, refs_id)
                        aws = None
                    else:
                        # ensemble
                        ensemble_eouts = []
                        ensemble_elens = []
                        ensemble_decs = []
                        if len(ensemble_models) > 0:
                            for i_e, model in enumerate(ensemble_models):
                                if model.input_type == 'speech' and model.mtl_per_batch and 'bwd' in dir:
                                    enc_outs_e = model.encode(xs, task, flip=True)
                                else:
                                    enc_outs_e = model.encode(xs, task, flip=False)
                                ensemble_eouts += [enc_outs_e[task]['xs']]
                                ensemble_elens += [enc_outs_e[task]['xlens']]
                                ensemble_decs += [getattr(model, 'dec_' + dir)]
                                # NOTE: only support for the main task now

                        lm, lm_rev = None, None
                        if params['recog_lm_weight'] > 0 and hasattr(self, 'lm_' + dir) and getattr(self, 'lm_' + dir) is not None:
                            lm = getattr(self, 'lm_' + dir)
                            if params['recog_reverse_lm_rescoring']:
                                if dir == 'fwd':
                                    lm_rev = self.lm_bwd
                                else:
                                    raise NotImplementedError

                        nbest_hyps_id, aws, scores, cache_info = getattr(self, 'dec_' + dir).beam_search(
                            enc_outs[task]['xs'], enc_outs[task]['xlens'],
                            params, idx2token, lm, lm_rev, ctc_log_probs,
                            nbest, exclude_eos, refs_id, utt_ids, speakers,
                            ensemble_eouts, ensemble_elens, ensemble_decs)

                        if nbest == 1:
                            best_hyps_id = [hyp[0] for hyp in nbest_hyps_id]
                            aws = [aw[0] for aw in aws]
                        else:
                            return nbest_hyps_id, aws, scores, cache_info
                        # NOTE: nbest >= 2 is used for MWER training only

                return best_hyps_id, aws, cache_info
