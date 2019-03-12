#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN decoder (including CTC loss calculation)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import warpctc_pytorch
except:
    raise ImportError('Install warpctc_pytorch.')

from neural_sp.models.criterion import cross_entropy_lsm
from neural_sp.models.criterion import focal_loss
from neural_sp.models.criterion import kldiv_lsm_ctc
from neural_sp.models.model_utils import Embedding
from neural_sp.models.model_utils import LinearND
from neural_sp.models.model_utils import ResidualFeedForward
from neural_sp.models.rnnlm.rnnlm_seq import SeqRNNLM
from neural_sp.models.seq2seq.decoders.attention import AttentionMechanism
from neural_sp.models.seq2seq.decoders.ctc_beam_search_decoder import BeamSearchDecoder
from neural_sp.models.seq2seq.decoders.ctc_beam_search_decoder import CTCPrefixScore
from neural_sp.models.seq2seq.decoders.ctc_greedy_decoder import GreedyDecoder
from neural_sp.models.seq2seq.decoders.multihead_attention import MultiheadAttentionMechanism
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list
from neural_sp.models.torch_utils import tensor2np


random.seed(1)

logger = logging.getLogger("decoding")


class RNNDecoder(nn.Module):
    """RNN decoder.

    Args:
        sos (int): index for <sos>
        eos (int): index for <eos>
        pad (int): index for <pad>
        blank (int): index for <blank>
        enc_n_units (int):
        attn_type (str):
        attn_dim (int):
        attn_sharpening_factor (float):
        attn_sigmoid_smoothing (bool):
        attn_conv_out_channels (int):
        attn_conv_kernel_size (int):
        attn_n_heads (int): number of attention heads
        rnn_type (str): lstm or gru
        n_units (int): number of units in each RNN layer
        n_projs (int): number of units in each projection layer
        n_layers (int): number of RNN layers
        loop_type (str): normal or lmdecoder or conditional or rnmt
        residual (bool):
        add_ffl (bool):
        layerwise_attention (bool):
        emb_dim (int): dimension of the embedding in target spaces.
        tie_embedding (bool):
        vocab (int): number of nodes in softmax layer
        dropout (float): probability to drop nodes in the RNN layer
        dropout_emb (float): probability to drop nodes of the embedding layer
        dropout_att (float): dropout probabilities for attention distributions
        ss_prob (float): scheduled sampling probability
        ss_type (str): constant or saturation
        lsm_prob (float): label smoothing probability
        layer_norm (bool): layer normalization
        ctc_weight (float):
        ctc_fc_list (list):
        input_feeding (bool):
        backward (bool): decode in the backward order
        rnnlm_cold_fusion (RNNLM):
        cold_fusion_type (str): the type of cold fusion
            prob: probability from RNNLM
            hidden: hidden states of RNNLM
        rnnlm_init (RNNLM):
        lmobj_weight (float):
        share_lm_softmax (bool):
        global_weight (float):
        mtl_per_batch (bool):

    """

    def __init__(self,
                 sos,
                 eos,
                 pad,
                 blank,
                 enc_n_units,
                 attn_type,
                 attn_dim,
                 attn_sharpening_factor,
                 attn_sigmoid_smoothing,
                 attn_conv_out_channels,
                 attn_conv_kernel_size,
                 attn_n_heads,
                 rnn_type,
                 n_units,
                 n_projs,
                 n_layers,
                 residual,
                 add_ffl,
                 layerwise_attention,
                 loop_type,
                 emb_dim,
                 tie_embedding,
                 vocab,
                 dropout,
                 dropout_emb,
                 dropout_att,
                 ss_prob,
                 ss_type,
                 lsm_prob,
                 layer_norm,
                 fl_weight,
                 fl_gamma,
                 ctc_weight,
                 ctc_fc_list,
                 input_feeding,
                 backward,
                 rnnlm_cold_fusion,
                 cold_fusion_type,
                 rnnlm_init,
                 lmobj_weight,
                 share_lm_softmax,
                 global_weight,
                 mtl_per_batch):

        super(RNNDecoder, self).__init__()

        self.sos = sos
        self.eos = eos
        self.pad = pad
        self.blank = blank
        self.rnn_type = rnn_type
        assert rnn_type in ['lstm', 'gru']
        self.enc_n_units = enc_n_units
        self.dec_n_units = n_units
        self.n_projs = n_projs
        self.n_layers = n_layers
        self.loop_type = loop_type
        if loop_type in ['conditional', 'lmdecoder', 'rnmt']:
            assert n_layers >= 2
        self.residual = residual
        self.add_ffl = add_ffl
        self.layerwise_attention = layerwise_attention
        self.ss_prob = ss_prob
        self.ss_type = ss_type
        if ss_type == 'constant':
            self._ss_prob = ss_prob
        elif ss_type == 'saturation':
            self._ss_prob = 0  # start from 0
        self.lsm_prob = lsm_prob
        self.layer_norm = layer_norm
        self.fl_weight = fl_weight
        self.fl_gamma = fl_gamma
        self.ctc_weight = ctc_weight
        self.ctc_fc_list = ctc_fc_list
        self.input_feeding = input_feeding
        if input_feeding:
            assert loop_type == 'normal'
        self.backward = backward
        self.rnnlm_cf = rnnlm_cold_fusion
        self.cold_fusion_type = cold_fusion_type
        self.rnnlm_init = rnnlm_init
        if rnnlm_init:
            assert loop_type == 'lmdecoder'
        self.lmobj_weight = lmobj_weight
        if lmobj_weight > 0:
            assert loop_type in ['normal', 'lmdecoder']
            assert not input_feeding
        self.share_lm_softmax = share_lm_softmax
        self.global_weight = global_weight
        self.mtl_per_batch = mtl_per_batch

        # for cache
        self.global_cache_keys = []
        self.global_cache_values = []
        self.global_cache_values_lm = []
        self.prev_speaker = ''

        if ctc_weight > 0:
            # Fully-connected layers for CTC
            if len(ctc_fc_list) > 0:
                fc_layers = OrderedDict()
                for i in range(len(ctc_fc_list)):
                    input_dim = enc_n_units if i == 0 else ctc_fc_list[i - 1]
                    fc_layers['fc' + str(i)] = LinearND(input_dim, ctc_fc_list[i], dropout=dropout)
                fc_layers['fc' + str(len(ctc_fc_list))] = LinearND(ctc_fc_list[-1], vocab, dropout=0)
                self.output_ctc = nn.Sequential(fc_layers)
            else:
                self.output_ctc = LinearND(enc_n_units, vocab)
            self.decode_ctc_greedy = GreedyDecoder(blank=blank)
            self.decode_ctc_beam = BeamSearchDecoder(blank=blank)
            self.warpctc_loss = warpctc_pytorch.CTCLoss(size_average=True)

        if ctc_weight < global_weight:
            # Attention layer
            if attn_n_heads > 1:
                self.score = MultiheadAttentionMechanism(
                    enc_n_units=self.enc_n_units,
                    dec_n_units=n_units if n_projs == 0 else n_projs,
                    attn_type=attn_type,
                    attn_dim=attn_dim,
                    sharpening_factor=attn_sharpening_factor,
                    sigmoid_smoothing=attn_sigmoid_smoothing,
                    conv_out_channels=attn_conv_out_channels,
                    conv_kernel_size=attn_conv_kernel_size,
                    nheads=attn_n_heads,
                    dropout=dropout_att)
            else:
                self.score = AttentionMechanism(
                    enc_n_units=self.enc_n_units,
                    dec_n_units=n_units if n_projs == 0 else n_projs,
                    attn_type=attn_type,
                    attn_dim=attn_dim,
                    sharpening_factor=attn_sharpening_factor,
                    sigmoid_smoothing=attn_sigmoid_smoothing,
                    conv_out_channels=attn_conv_out_channels,
                    conv_kernel_size=attn_conv_kernel_size,
                    dropout=dropout_att)

            # for decoder initialization with pre-trained RNNLM
            if rnnlm_init:
                assert rnnlm_init.predictor.vocab == vocab
                assert rnnlm_init.predictor.n_units == n_units
                assert rnnlm_init.predictor.n_layers == 1  # TODO(hirofumi): on-the-fly

            # for MTL with RNNLM objective
            if lmobj_weight > 0 and loop_type == 'lmdecoder':
                if share_lm_softmax:
                    self.output_lmobj = self.output  # share paramters
                else:
                    self.output_lmobj = LinearND(n_units, vocab)

            # Decoder
            self.rnn = nn.ModuleList()
            self.dropout = nn.ModuleList()
            if self.n_projs > 0:
                self.proj = nn.ModuleList()
            if add_ffl:
                self.ffl = nn.ModuleList()
            if rnn_type == 'lstm':
                rnn_cell = nn.LSTMCell
            elif rnn_type == 'gru':
                rnn_cell = nn.GRUCell

            if loop_type == 'normal':
                dec_idim = n_units if input_feeding else enc_n_units
                self.rnn += [rnn_cell(emb_dim + dec_idim, n_units)]
                dec_idim = n_units
                if self.n_projs > 0:
                    self.proj += [LinearND(n_units, n_projs)]
                    dec_idim = n_projs
                self.dropout += [nn.Dropout(p=dropout)]
                if add_ffl:
                    self.ffl += [ResidualFeedForward(dec_idim, dec_idim * 4, dropout, layer_norm)]
                for l in range(n_layers - 1):
                    self.rnn += [rnn_cell(dec_idim, n_units)]
                    if self.n_projs > 0:
                        self.proj += [LinearND(n_units, n_projs)]
                    self.dropout += [nn.Dropout(p=dropout)]
                    if add_ffl:
                        self.ffl += [ResidualFeedForward(dec_idim, dec_idim * 4, dropout, layer_norm)]
            elif loop_type == 'lmdecoder':
                if add_ffl:
                    raise ValueError()
                # 1st layer
                self.rnn += [rnn_cell(emb_dim, n_units)]
                if self.n_projs > 0:
                    self.proj += [LinearND(n_units, n_projs)]
                self.dropout += [nn.Dropout(p=dropout)]
                # 2nd layer
                if self.n_projs > 0:
                    self.rnn += [rnn_cell(n_projs + enc_n_units, n_units)]
                    self.proj += [LinearND(n_units, n_projs)]
                else:
                    self.rnn += [rnn_cell(n_units + enc_n_units, n_units)]
                self.dropout += [nn.Dropout(p=dropout)]
                for l in range(n_layers - 2):
                    if self.n_projs > 0:
                        self.rnn += [rnn_cell(n_projs, n_units)]
                        self.proj += [LinearND(n_units, n_projs)]
                    else:
                        self.rnn += [rnn_cell(n_units, n_units)]
                    self.dropout += [nn.Dropout(p=dropout)]
            elif loop_type == 'conditional':
                if add_ffl:
                    raise ValueError()
                # 1st layer
                self.rnn += [rnn_cell(emb_dim, n_units)]
                if self.n_projs > 0:
                    self.proj += [LinearND(n_units, n_projs)]
                self.dropout += [nn.Dropout(p=dropout)]
                # 2nd layer
                self.rnn += [rnn_cell(enc_n_units, n_units)]
                if self.n_projs > 0:
                    self.proj += [LinearND(n_units, n_projs)]
                self.dropout += [nn.Dropout(p=dropout)]
                for l in range(n_layers - 2):
                    if self.n_projs > 0:
                        self.rnn += [rnn_cell(n_projs, n_units)]
                        self.proj += [LinearND(n_units, n_projs)]
                    else:
                        self.rnn += [rnn_cell(n_units, n_units)]
                    self.dropout += [nn.Dropout(p=dropout)]
            elif loop_type == 'rnmt':
                if add_ffl:
                    raise ValueError()
                assert residual
                self.rnn += [rnn_cell(emb_dim, n_units)]
                if self.n_projs > 0:
                    self.proj += [LinearND(n_units, n_projs)]
                self.dropout += [nn.Dropout(p=dropout)]
                for l in range(n_layers - 1):
                    if self.n_projs > 0:
                        self.rnn += [rnn_cell(n_projs + enc_n_units, n_units)]
                        self.proj += [LinearND(n_units, n_projs)]
                    else:
                        self.rnn += [rnn_cell(n_units + enc_n_units, n_units)]
                    self.dropout += [nn.Dropout(p=dropout)]
            else:
                raise NotImplementedError(loop_type)

            # cold fusion
            if rnnlm_cold_fusion is not None:
                if self.n_projs > 0:
                    self.cf_linear_dec_feat = LinearND(n_projs + enc_n_units, n_units)
                else:
                    self.cf_linear_dec_feat = LinearND(n_units + enc_n_units, n_units)
                if cold_fusion_type == 'hidden':
                    self.cf_linear_lm_feat = LinearND(rnnlm_cold_fusion.n_units, n_units)
                elif cold_fusion_type == 'prob':
                    self.cf_linear_lm_feat = LinearND(rnnlm_cold_fusion.vocab, n_units)
                else:
                    raise ValueError(cold_fusion_type)
                self.cf_linear_lm_gate = LinearND(n_units * 2, n_units)
                self.output_bn = LinearND(n_units * 2, n_units)

                # fix RNNLM parameters
                for p in self.rnnlm_cf.parameters():
                    p.requires_grad = False
            else:
                if self.n_projs > 0:
                    self.output_bn = LinearND(n_projs + enc_n_units, n_units)
                else:
                    self.output_bn = LinearND(n_units + enc_n_units, n_units)

            self.embed = Embedding(vocab, emb_dim,
                                   dropout=dropout_emb,
                                   ignore_index=pad)
            self.output = LinearND(n_units, vocab)
            # NOTE: include bias even when tying weights

            # Optionally tie weights as in:
            # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
            # https://arxiv.org/abs/1608.05859
            # and
            # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
            # https://arxiv.org/abs/1611.01462
            if tie_embedding:
                if n_units != emb_dim:
                    raise ValueError('When using the tied flag, n_units must be equal to emb_dim.')
                self.output.fc.weight = self.embed.embed.weight

    @property
    def device_id(self):
        return torch.cuda.device_of(next(self.parameters()).data).idx

    def start_scheduled_sampling(self):
        self._ss_prob = self.ss_prob

    def forward(self, eouts, elens, ys, task='all'):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, dec_units]`
            elens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            task (str): all or ys or ys_sub*
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None,
                       'loss_att': None, 'loss_ctc': None, 'loss_lmobj': None,
                       'acc_att': None, 'acc_lmobj': None,
                       'ppl_att': None, 'ppl_lmobj': None}
        loss = eouts.new_zeros((1,))

        # CTC loss
        if self.ctc_weight > 0 and (not self.mtl_per_batch or (self.mtl_per_batch and 'ctc' in task)):
            loss_ctc = self.forward_ctc(eouts, elens, ys)
            observation['loss_ctc'] = loss_ctc.item()
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight

        # LM objective
        if self.lmobj_weight > 0 and 'lmobj' in task:
            loss_lmobj, acc_lmobj, ppl_lmobj = self.forward_lmobj(ys)
            observation['loss_lmobj'] = loss_lmobj.item()
            observation['acc_lmobj'] = acc_lmobj
            observation['ppl_lmobj'] = ppl_lmobj
            if self.mtl_per_batch:
                loss += loss_lmobj
            else:
                loss += loss_lmobj * self.lmobj_weight

        # XE loss
        if self.global_weight - self.ctc_weight > 0 and 'ctc' not in task and 'lmobj' not in task:
            loss_att, acc_att, ppl_att = self.forward_att(eouts, elens, ys)
            observation['loss_att'] = loss_att.item()
            observation['acc_att'] = acc_att
            observation['ppl_att'] = ppl_att
            if self.mtl_per_batch:
                loss += loss_att
            else:
                loss += loss_att * (self.global_weight - self.ctc_weight)

        observation['loss'] = loss.item()
        return loss, observation

    def forward_ctc(self, eouts, elens, ys):
        """Compute CTC loss.

        Args:
            eouts (FloatTensor): `[B, T, dec_units]`
            elens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[B, L, vocab]`

        """
        logits = self.output_ctc(eouts)

        # Compute the auxiliary CTC loss
        elens_ctc = np2tensor(np.fromiter(elens, dtype=np.int32), -1).int()
        ys_ctc = [np2tensor(np.fromiter(y, dtype=np.int64)).long() for y in ys]  # always fwd
        ylens = np2tensor(np.fromiter([y.size(0) for y in ys_ctc], dtype=np.int32), -1).int()
        ys_ctc = torch.cat(ys_ctc, dim=0).int()
        # NOTE: Concatenate all elements in ys for warpctc_pytorch
        # NOTE: do not copy to GPUs here

        # Compute CTC loss
        loss = self.warpctc_loss(logits.transpose(1, 0).cpu(),  # time-major
                                 ys_ctc, elens_ctc, ylens)
        # NOTE: ctc loss has already been normalized by bs
        # NOTE: index 0 is reserved for blank in warpctc_pytorch

        if self.device_id >= 0:
            loss = loss.cuda(self.device_id)

        # Label smoothing for CTC
        if self.lsm_prob > 0 and self.ctc_weight == 1:
            loss = loss * (1 - self.lsm_prob) + kldiv_lsm_ctc(
                logits, ylens=elens,
                lsm_prob=self.lsm_prob, size_average=True) * self.lsm_prob

        return loss

    def forward_lmobj(self, ys):
        """Compute XE loss for LM objective.

        Args:
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[1]`
            acc (float):
            ppl (float):

        """
        bs = len(ys)
        w = next(self.parameters())

        # Append <sos> and <eos>
        sos = w.new_zeros((1,)).fill_(self.sos).long()
        eos = w.new_zeros((1,)).fill_(self.eos).long()
        if self.backward:
            ys = [np2tensor(np.fromiter(y[::-1], dtype=np.int64), self.device_id).long() for y in ys]
            ys_in = [torch.cat([eos, y], dim=0) for y in ys]
            ys_out = [torch.cat([y, sos], dim=0) for y in ys]
        else:
            ys = [np2tensor(np.fromiter(y, dtype=np.int64), self.device_id).long() for y in ys]
            ys_in = [torch.cat([sos, y], dim=0) for y in ys]
            ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        ys_in_pad = pad_list(ys_in, self.pad)
        ys_out_pad = pad_list(ys_out, -1)

        # Initialization
        dstates = self.init_dec_state(bs, self.n_layers)
        cv = w.new_zeros((bs, 1, self.enc_n_units))
        attn_v = w.new_zeros((bs, 1, self.dec_n_units))

        # Pre-computation of embedding
        ys_emb = self.embed(ys_in_pad)

        logits = []
        for t in range(ys_in_pad.size(1)):
            y_emb = ys_emb[:, t:t + 1]

            # Recurrency
            dstates = self.recurrency(y_emb, cv, dstates['dstate'])

            # Generate
            if self.loop_type == 'lmdecoder':
                logits_t = self.output_lmobj(dstates['dout_lmdec'])
            elif self.loop_type == 'normal':
                attn_v = self.generate(cv, dstates['dout_gen'])
                logits_t = self.output(attn_v)
            logits.append(logits_t)

        # Compute XE loss for RNNLM objective
        logits = torch.cat(logits, dim=1)
        loss = F.cross_entropy(logits.view((-1, logits.size(2))),
                               ys_out_pad.view(-1),
                               ignore_index=-1, size_average=False) / bs
        ppl = math.exp(loss.item())

        # Compute token-level accuracy in teacher-forcing
        pad_pred = logits.view(ys_out_pad.size(0), ys_out_pad.size(1), logits.size(-1)).argmax(2)
        mask = ys_out_pad != -1
        numerator = torch.sum(pad_pred.masked_select(mask) == ys_out_pad.masked_select(mask))
        denominator = torch.sum(mask)
        acc = float(numerator) * 100 / float(denominator)

        return loss, acc, ppl

    def forward_att(self, eouts, elens, ys):
        """Compute XE loss for the sequence-to-sequence model.

        Args:
            eouts (FloatTensor): `[B, T, dec_units]`
            elens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[B, L, vocab]`
            acc (float):
            ppl (float):

        """
        bs = eouts.size(0)

        # Append <sos> and <eos>
        sos = eouts.new_zeros(1).fill_(self.sos).long()
        eos = eouts.new_zeros(1).fill_(self.eos).long()
        if self.backward:
            _ys = [np2tensor(np.fromiter(y[::-1], dtype=np.int64), self.device_id).long() for y in ys]
            ys_in = [torch.cat([eos, y], dim=0) for y in _ys]
            ys_out = [torch.cat([y, sos], dim=0) for y in _ys]
        else:
            _ys = [np2tensor(np.fromiter(y, dtype=np.int64), self.device_id).long() for y in ys]
            ys_in = [torch.cat([sos, y], dim=0) for y in _ys]
            ys_out = [torch.cat([y, eos], dim=0) for y in _ys]
        ys_in_pad = pad_list(ys_in, self.pad)
        ys_out_pad = pad_list(ys_out, -1)

        # Initialization
        dstates = self.init_dec_state(bs, self.n_layers, eouts, elens)
        cv = eouts.new_zeros(bs, 1, self.enc_n_units)
        attn_v = eouts.new_zeros(bs, 1, self.dec_n_units)
        self.score.reset()
        aw = None
        rnnlm_state = (None, None)

        # Pre-computation of embedding
        ys_emb = self.embed(ys_in_pad)
        if self.rnnlm_cf:
            ys_lm_emb = self.rnnlm_cf.embed(ys_in_pad)

        logits = []
        for t in range(ys_in_pad.size(1)):
            # Sample for scheduled sampling
            is_sample = t > 0 and self._ss_prob > 0 and random.random() < self._ss_prob
            if is_sample:
                y_emb = self.embed(logits[-1].detach().argmax(-1))
            else:
                y_emb = ys_emb[:, t:t + 1]

            # Recurrency (1st)
            dec_in = attn_v if self.input_feeding else cv
            if self.loop_type in ['conditional', 'rnmt']:
                dstates = self.recurrency_step1(y_emb, dec_in, dstates)
            elif self.loop_type in ['normal', 'lmdecoder']:
                dstates = self.recurrency(y_emb, dec_in, dstates['dstate'])

            # Update RNNLM states for cold fusion
            if self.rnnlm_cf:
                if is_sample:
                    y_lm_emb = self.rnnlm_cf.embed(logits[-1].detach().argmax(-1)).squeeze(1)
                else:
                    y_lm_emb = ys_lm_emb[:, t]
                logits_lm_t, lm_out, rnnlm_state = self.rnnlm_cf.predict(
                    y_lm_emb, rnnlm_state)
            else:
                logits_lm_t, lm_out = None, None

            # Score
            cv, aw = self.score(eouts, elens, dstates['dout_score'], aw)

            # Recurrency (2nd, only for the internal decoder)
            if self.loop_type in ['conditional', 'rnmt']:
                dstates = self.recurrency_step2(cv, dstates)

            # Generate
            attn_v = self.generate(cv, dstates['dout_gen'], logits_lm_t, lm_out)
            logits.append(self.output(attn_v))

        # Compute XE sequence loss
        logits = torch.cat(logits, dim=1)
        if self.lsm_prob > 0:
            # Label smoothing
            ylens = [y.size(0) for y in ys_out]
            loss = cross_entropy_lsm(
                logits, ys=ys_out_pad, ylens=ylens,
                lsm_prob=self.lsm_prob, size_average=True)
        else:
            loss = F.cross_entropy(
                logits.view((-1, logits.size(2))),
                ys_out_pad.view(-1),  # long
                ignore_index=-1, size_average=False) / bs
        ppl = np.exp(loss.item())

        # Focal loss
        if self.fl_weight > 0:
            ylens = [y.size(0) for y in ys_out]
            fl = focal_loss(logits, ys=ys_out_pad, ylens=ylens,
                            gamma=self.fl_gamma, size_average=True)
            loss = loss * (1 - self.fl_weight) + fl * self.fl_weight

        # Compute token-level accuracy in teacher-forcing
        pad_pred = logits.view(ys_out_pad.size(0), ys_out_pad.size(1), logits.size(-1)).argmax(2)
        mask = ys_out_pad != -1
        numerator = (pad_pred.masked_select(mask) == ys_out_pad.masked_select(mask)).sum()
        denominator = mask.sum()
        acc = float(numerator) * 100 / float(denominator)

        return loss, acc, ppl

    def init_dec_state(self, bs, n_layers, eouts=None, elens=None):
        """Initialize decoder state.

        Args:
            eouts (FloatTensor): `[B, T, dec_units]`
            elens (list): A list of length `[B]`
            n_layers (int):
        Returns:
            dstates (dict):
                dout (FloatTensor): `[B, 1, dec_units]`
                dstate (tuple): A tuple of (hxs, cxs)
                    hxs (list of FloatTensor):
                    cxs (list of FloatTensor):

        """
        dstates = {'dout_score': None,  # for attention score
                   'dout_gen': None,  # for token generation
                   'dstate': None,
                   'dstate1': None,
                   'dstate2': None}
        w = next(self.parameters())

        dstates['dout_score'] = w.new_zeros((bs, 1, self.dec_n_units))
        dstates['dout_gen'] = w.new_zeros((bs, 1, self.dec_n_units))
        if self.loop_type in ['conditional', 'rnmt']:
            hxs1 = [w.new_zeros((bs, self.dec_n_units))
                    for l in range(1)]
            cxs1 = [w.new_zeros((bs, self.dec_n_units))
                    for l in range(1)] if self.rnn_type == 'lstm' else []
            dstates['dstate1'] = (hxs1, cxs1)
            hxs2 = [w.new_zeros((bs, self.dec_n_units))
                    for l in range(self.n_layers - 1)]
            cxs2 = [w.new_zeros((bs, self.dec_n_units))
                    for l in range(self.n_layers - 1)] if self.rnn_type == 'lstm' else []
            dstates['dstate2'] = (hxs2, cxs2)
        else:
            hxs = [w.new_zeros((bs, self.dec_n_units))
                   for l in range(self.n_layers)]
            cxs = [w.new_zeros((bs, self.dec_n_units))
                   for l in range(self.n_layers)] if self.rnn_type == 'lstm' else []
            dstates['dstate'] = (hxs, cxs)
        return dstates

    def recurrency(self, y_emb, cv, dstate):
        """Recurrency function.

        Args:
            y_emb (FloatTensor): `[B, 1, emb_dim]`
            cv (FloatTensor): `[B, 1, enc_n_units]`
            dstate (tuple): A tuple of (hxs, cxs)
        Returns:
            dstates_new (dict):
                dout_score (FloatTensor): `[B, 1, n_units]`
                dout_gen (FloatTensor): `[B, 1, n_units]`
                dstate (tuple): A tuple of (hxs, cxs)
                    hxs (list of FloatTensor):
                    cxs (list of FloatTensor):

        """
        hxs, cxs = dstate
        y_emb = y_emb.squeeze(1)
        cv = cv.squeeze(1)

        dstates_new = {'dout_score': None,  # for attention score
                       'dout_gen': None,  # for token generation
                       'dout_lmdec': None,
                       'dstate': None}
        if self.loop_type == 'lmdecoder':
            if self.rnn_type == 'lstm':
                hxs[0], cxs[0] = self.rnn[0](y_emb, (hxs[0], cxs[0]))
            elif self.rnn_type == 'gru':
                hxs[0] = self.rnn[0](y_emb, hxs[0])
        elif self.loop_type == 'normal':
            if self.rnn_type == 'lstm':
                hxs[0], cxs[0] = self.rnn[0](torch.cat([y_emb, cv], dim=-1), (hxs[0], cxs[0]))
            elif self.rnn_type == 'gru':
                hxs[0] = self.rnn[0](torch.cat([y_emb, cv], dim=-1), hxs[0])
        dout = hxs[0]
        if self.n_projs > 0:
            dout = torch.tanh(self.proj[0](dout))
        dout = self.dropout[0](dout)
        if self.add_ffl:
            dout = self.ffl[0](dout)

        if self.loop_type == 'lmdecoder' and self.lmobj_weight > 0:
            dstates_new['dout_lmdec'] = dout.unsqueeze(1)

        if self.loop_type == 'normal':
            # the bottom layer
            dstates_new['dout_score'] = dout.unsqueeze(1)

        for l in range(1, self.n_layers):
            if self.loop_type == 'lmdecoder' and l == 1:
                if self.rnn_type == 'lstm':
                    hxs[l], cxs[l] = self.rnn[l](torch.cat([dout, cv], dim=-1), (hxs[l], cxs[l]))
                elif self.rnn_type == 'gru':
                    hxs[l] = self.rnn[l](torch.cat([dout, cv], dim=-1), hxs[l])
            else:
                if self.rnn_type == 'lstm':
                    hxs[l], cxs[l] = self.rnn[l](dout, (hxs[l], cxs[l]))
                elif self.rnn_type == 'gru':
                    hxs[l] = self.rnn[l](dout, hxs[l])
            dout_tmp = hxs[l]
            if self.n_projs > 0:
                dout_tmp = torch.tanh(self.proj[l](dout_tmp))
            dout_tmp = self.dropout[l](dout_tmp)
            if self.add_ffl:
                dout_tmp = self.ffl[l](dout_tmp)

            if self.loop_type == 'lmdecoder' and l == 1:
                # the bottom layer
                dstates_new['dout_score'] = dout_tmp.unsqueeze(1)

            if self.residual:
                dout = dout_tmp + dout
            else:
                dout = dout_tmp

        # the top layer
        dstates_new['dout_gen'] = dout.unsqueeze(1)
        dstates_new['dstate'] = (hxs[:], cxs[:])
        return dstates_new

    def recurrency_step1(self, y_emb, cv, dstates):
        """Recurrency function for the internal deocder (before attention scoring).

        Args:
            y_emb (FloatTensor): `[B, 1, emb_dim]`
            cv (FloatTensor): `[B, 1, enc_n_units]`
            dstates (dict):
                dstates1 (tuple): A tuple of (hxs, cxs)
                    hxs (list of FloatTensor):
                    cxs (list of FloatTensor):
                dstates2 (tuple): A tuple of (hxs, cxs)
                    hxs (list of FloatTensor):
                    cxs (list of FloatTensor):
        Returns:
            dstates_new (dict):
                dout_score (FloatTensor): `[B, 1, n_units]`
                dstate1 (tuple): A tuple of (hxs, cxs)
                    hxs (list of FloatTensor):
                    cxs (list of FloatTensor):
                dstate2 (tuple): A tuple of (hxs, cxs)
                    hxs (list of FloatTensor):
                    cxs (list of FloatTensor):

        """
        hxs, cxs = dstates['dstate1']
        y_emb = y_emb.squeeze(1)
        cv = cv.squeeze(1)

        dstates_new = {'dout_score': None,  # for attention score
                       'dstate1': None,
                       'dstate2': dstates['dstate2']}
        if self.loop_type == 'conditional':
            if self.rnn_type == 'lstm':
                hxs[0], cxs[0] = self.rnn[0](y_emb, (hxs[0], cxs[0]))
            elif self.rnn_type == 'gru':
                hxs[0] = self.rnn[0](y_emb, hxs[0])
        elif self.loop_type == 'rnmt':
            if self.rnn_type == 'lstm':
                hxs[0], cxs[0] = self.rnn[0](y_emb, (hxs[0], cxs[0]))
            elif self.rnn_type == 'gru':
                hxs[0] = self.rnn[0](y_emb, hxs[0])
        dout = hxs[0]
        if self.n_projs > 0:
            dout = torch.tanh(self.proj[0](dout))
        dout = self.dropout[0](dout)

        # the bottom layer
        dstates_new['dout_score'] = dout.unsqueeze(1)
        dstates_new['dstate1'] = (hxs[:], cxs[:])
        return dstates_new

    def recurrency_step2(self, cv, dstates):
        """Recurrency function for the internal deocder (after attention scoring).

        Args:
            cv (FloatTensor): `[B, 1, enc_n_units]`
            dstates (dict):
                dout_score (FloatTensor): `[B, 1, n_units]`
                dstate1 (tuple): A tuple of (hxs, cxs),
                    hxs (list of FloatTensor):
                    cxs (list of FloatTensor):
                dstate2 (tuple): A tuple of (hxs, cxs),
                    hxs (list of FloatTensor):
                    cxs (list of FloatTensor):
        Returns:
            dstates_new (dict):
                dout_gen (FloatTensor): `[B, 1, n_units]`
                dstate1 (tuple): A tuple of (hxs, cxs),
                    hxs (list of FloatTensor):
                    cxs (list of FloatTensor):
                dstate2 (tuple): A tuple of (hxs, cxs),
                    hxs (list of FloatTensor):
                    cxs (list of FloatTensor):

        """
        hxs, cxs = dstates['dstate2']
        cv = cv.squeeze(1)

        dstates_new = {'dout_gen': None,  # for token generation
                       'dstate1': None,
                       'dstate2': None}

        dout = dstates['dout_score'].squeeze(1)
        for l in range(1, self.n_layers):
            if self.loop_type == 'conditional':
                if l == 1:
                    if self.rnn_type == 'lstm':
                        dstates['dstate1'][0][0], dstates['dstate1'][1][0] = self.rnn[l](
                            cv, (dstates['dstate1'][0][0], dstates['dstate1'][1][0]))
                    elif self.rnn_type == 'gru':
                        dstates['dstate1'][0][0] = self.rnn[l](cv, dstates['dstate1'][0][0])
                else:
                    if self.rnn_type == 'lstm':
                        hxs[l - 1], cxs[l - 1] = self.rnn[l](dout, (hxs[l - 1], cxs[l - 1]))
                    elif self.rnn_type == 'gru':
                        hxs[l - 1] = self.rnn[l](dout, hxs[l - 1])
            elif self.loop_type == 'rnmt':
                if self.rnn_type == 'lstm':
                    hxs[l - 1], cxs[l - 1] = self.rnn[l](torch.cat([dout, cv], dim=-1), (hxs[l - 1], cxs[l - 1]))
                elif self.rnn_type == 'gru':
                    hxs[l - 1] = self.rnn[l](torch.cat([dout, cv], dim=-1), hxs[l - 1])

            if self.loop_type == 'conditional' and l == 1:
                dout_tmp = dstates['dstate1'][0][0]
                if self.n_projs > 0:
                    dout_tmp = torch.tanh(self.proj[l](dout_tmp))
                dout_tmp = self.dropout[l](dout_tmp)
                if self.residual:
                    dout = dout_tmp + dout
                else:
                    dout = dout_tmp
            else:
                dout_tmp = hxs[l - 1]
                if self.n_projs > 0:
                    dout_tmp = torch.tanh(self.proj[l](dout_tmp))
                dout_tmp = self.dropout[l](dout_tmp)
                if self.residual:
                    dout = dout_tmp + dout
                else:
                    dout = dout_tmp

        # the top layer
        dstates_new['dout_gen'] = dout.unsqueeze(1)
        dstates_new['dstate1'] = (dstates['dstate1'][0][:], dstates['dstate1'][1][:])
        dstates_new['dstate2'] = (hxs[:], cxs[:])
        return dstates_new

    def generate(self, cv, dout, logits_lm_t, lm_out):
        """Generate function.

        Args:
            cv (FloatTensor): `[B, 1, enc_n_units]`
            dout (FloatTensor): `[B, 1, dec_units]`
            logits_lm_t (FloatTensor): `[B, vocab]`
            lm_out (FloatTensor): `[B, lm_nunits]`
        Returns:
            logits_t (FloatTensor): `[B, 1, vocab]`

        """
        if self.rnnlm_cf:
            # cold fusion
            if self.cold_fusion_type == 'hidden':
                lm_feat = self.cf_linear_lm_feat(lm_out.unsqueeze(1))
            elif self.cold_fusion_type == 'prob':
                lm_feat = self.cf_linear_lm_feat(logits_lm_t.unsqueeze(1))
            dec_feat = self.cf_linear_dec_feat(torch.cat([dout, cv], dim=-1))
            gate = torch.sigmoid(self.cf_linear_lm_gate(torch.cat([dec_feat, lm_feat], dim=-1)))
            gated_lm_feat = gate * lm_feat
            logits_t = self.output_bn(torch.cat([dec_feat, gated_lm_feat], dim=-1))
        else:
            logits_t = self.output_bn(torch.cat([dout, cv], dim=-1))
        return torch.tanh(logits_t)

    def greedy(self, eouts, elens, max_len_ratio, exclude_eos=False):
        """Greedy decoding in the inference stage.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (list): A list of length `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            exclude_eos (bool):
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`

        """
        bs, max_xlen, enc_n_units = eouts.size()

        # Initialization
        dstates = self.init_dec_state(bs, self.n_layers, eouts, elens)
        cv = eouts.new_zeros(bs, 1, self.enc_n_units)
        attn_v = eouts.new_zeros(bs, 1, self.dec_n_units)
        self.score.reset()
        aw = None
        rnnlm_state = None

        if self.backward:
            sos, eos = self.eos, self.sos
        else:
            sos, eos = self.sos, self.eos

        # Start from <sos> (<eos> in case of the backward decoder)
        y = eouts.new_zeros(bs, 1).fill_(sos).long()

        best_hyps_tmp, aws_tmp = [], []
        ylens = np.zeros((bs,), dtype=np.int32)
        eos_flags = [False] * bs
        for t in range(int(math.floor(max_xlen * max_len_ratio)) + 1):
            # Recurrency (1st)
            y_emb = self.embed(y)
            dec_in = attn_v if self.input_feeding else cv
            if self.loop_type in ['conditional', 'rnmt']:
                dstates = self.recurrency_step1(y_emb, dec_in, dstates)
            elif self.loop_type in ['normal', 'lmdecoder']:
                dstates = self.recurrency(y_emb, dec_in, dstates['dstate'])

            # Update RNNLM states for cold fusion
            if self.rnnlm_cf:
                y_lm = self.rnnlm_cf.embed(y)
                logits_lm_t, lm_out, rnnlm_state = self.rnnlm_cf.predict(y_lm, rnnlm_state)
            else:
                logits_lm_t, lm_out = None, None

            # Score
            cv, aw = self.score(eouts, elens, dstates['dout_score'], aw)

            # Recurrency (2nd, only for the internal decoder)
            if self.loop_type in ['conditional', 'rnmt']:
                dstates = self.recurrency_step2(cv, dstates)

            # Generate
            attn_v = self.generate(cv, dstates['dout_gen'], logits_lm_t, lm_out)
            logits_t = self.output(attn_v)

            # Pick up 1-best
            y = logits_t.detach().argmax(-1)
            best_hyps_tmp += [y]
            if self.score.nheads > 1:
                aws_tmp += [aw[0]]
            else:
                aws_tmp += [aw]

            # Count lengths of hypotheses
            for b in range(bs):
                if not eos_flags[b]:
                    if y[b].item() == eos:
                        eos_flags[b] = True
                    ylens[b] += 1
                    # NOTE: include <eos>

            # Break if <eos> is outputed in all mini-bs
            if sum(eos_flags) == bs:
                break

        # Concatenate in L dimension
        best_hyps_tmp = torch.cat(best_hyps_tmp, dim=1)
        aws_tmp = torch.stack(aws_tmp, dim=1)

        # Convert to numpy
        best_hyps_tmp = tensor2np(best_hyps_tmp)
        aws_tmp = tensor2np(aws_tmp)

        # Truncate by the first <eos> (<sos> in case of the backward decoder)
        if self.backward:
            # Reverse the order
            best_hyps = [best_hyps_tmp[b, :ylens[b]][::-1] for b in range(bs)]
            aws = [aws_tmp[b, :ylens[b]][::-1] for b in range(bs)]
        else:
            best_hyps = [best_hyps_tmp[b, :ylens[b]] for b in range(bs)]
            aws = [aws_tmp[b, :ylens[b]] for b in range(bs)]

        # Exclude <eos> (<sos> in case of the backward decoder)
        if exclude_eos:
            if self.backward:
                best_hyps = [best_hyps[b][1:] if eos_flags[b]
                             else best_hyps[b] for b in range(bs)]
            else:
                best_hyps = [best_hyps[b][:-1] if eos_flags[b]
                             else best_hyps[b] for b in range(bs)]

        return best_hyps, aws

    def beam_search(self, eouts, elens, params, rnnlm, rnnlm_rev=None, ctc_log_probs=None,
                    nbest=1, exclude_eos=False, id2token=None, refs=None,
                    ensemble_eouts=None, ensemble_elens=None, ensemble_decoders=[],
                    speakers=None):
        """Beam search decoding in the inference stage.

        Args:
            eouts (FloatTensor): `[B, T, dec_units]`
            elens (list): A list of length `[B]`
            params (dict):
                beam_width (int): size of beam
                max_len_ratio (int): maximum sequence length of tokens
                min_len_ratio (float): minimum sequence length of tokens
                length_penalty (float): length penalty
                coverage_penalty (float): coverage penalty
                coverage_threshold (float): threshold for coverage penalty
                rnnlm_weight (float): weight of RNNLM score
                n_caches (int):
            rnnlm (torch.nn.Module):
            rnnlm_rev (torch.nn.Module):
            ctc_log_probs (torch.FloatTensor):
            nbest (int):
            exclude_eos (bool):
            id2token (): converter from index to token
            refs ():
            ensemble_eouts (list): list of FloatTensor
            ensemble_elens (list) list of list
            ensemble_decoders (list): list of torch.nn.Module
            speakers (list):
        Returns:
            nbest_hyps (list): A list of length `[B]`, which contains list of n hypotheses
            aws (list): A list of length `[B]`, which contains arrays of size `[L, T]`
            scores (list):

        """
        bs, _, enc_n_units = eouts.size()
        n_models = len(ensemble_decoders) + 1

        # TODO(hirofumi): fix later
        beam_width = params['recog_beam_width']
        ctc_weight = params['recog_ctc_weight']
        lp_weight = params['recog_length_penalty']
        cp_weight = params['recog_coverage_penalty']
        cp_threshold = params['recog_coverage_threshold']
        rnnlm_weight = params['recog_rnnlm_weight']
        gnmt_decoding = params['recog_gnmt_decoding']
        n_caches = params['recog_ncaches']

        # For cold fusion
        if rnnlm_weight > 0 and self.cold_fusion_type:
            assert self.rnnlm_cf
            self.rnnlm_cf.eval()

        # For shallow fusion
        if rnnlm is not None:
            rnnlm.eval()
        if rnnlm_rev is not None:
            rnnlm_rev.eval()

        # For joint CTC-Attention decoding
        if ctc_weight > 0 and ctc_log_probs is not None:
            if self.backward:
                ctc_prefix_score = CTCPrefixScore(tensor2np(ctc_log_probs)[0][::-1], self.blank, self.eos)
            else:
                ctc_prefix_score = CTCPrefixScore(tensor2np(ctc_log_probs)[0], self.blank, self.eos)

        if self.backward:
            sos, eos = self.eos, self.sos
        else:
            sos, eos = self.sos, self.eos

        nbest_hyps, aws, scores, scores_cp = [], [], [], []
        eos_flags = []
        for b in range(bs):
            # Initialization per utterance
            dstates = self.init_dec_state(1, self.n_layers, eouts[b:b + 1], elens[b:b + 1])
            if self.input_feeding:
                cv = eouts.new_zeros(1, 1, self.dec_n_units)
                # NOTE: this is equivalent to attn_v
            else:
                cv = eouts.new_zeros(1, 1, self.enc_n_units)
            self.score.reset()

            # Ensemble initialization
            ensemble_dstates = []
            ensemble_cv = []
            if n_models > 0:
                for dec in ensemble_decoders:
                    ensemble_dstates += [dec.init_dec_state(1, dec.n_layers, eouts[b:b + 1], elens[b:b + 1])]
                    if dec.input_feeding:
                        ensemble_cv += [eouts.new_zeros(1, 1, dec.dec_n_units)]
                        # NOTE: this is equivalent to attn_v
                    else:
                        ensemble_cv += [eouts.new_zeros(1, 1, dec.enc_n_units)]
                    dec.score.reset()

            if speakers[b] != self.prev_speaker:
                self.reset_global_cache()
            self.prev_speaker = speakers[b]

            complete = []
            beam = [{'hyp': [sos],
                     'score': 0,
                     'scores': [0],
                     'scores_cp': [0],
                     'score_att': 0,
                     'score_ctc': 0,
                     'score_lm': 0,
                     'dstates': dstates,
                     'cv': cv,
                     'aws': [None],
                     'rnnlm_hxs': None,
                     'rnnlm_cxs': None,
                     'cp_prev': 0,
                     'ensemble_dstates': ensemble_dstates,
                     'ensemble_cv': ensemble_cv,
                     'ensemble_aws':[[None] for _ in range(n_models)],
                     'ctc_state':  ctc_prefix_score.initial_state() if ctc_weight > 0 and ctc_log_probs is not None else None,
                     'ctc_score': 0,
                     'local_cache_keys': [],
                     'local_cache_values': [],
                     'local_cache_values_lm': [],
                     'cache_keys_history': [None],
                     'cache_probs_history': [torch.zeros((1, 1, 1), dtype=torch.float32)] if len(self.global_cache_keys) == 0 else [],
                     'cache_probs_history_lm': [torch.zeros((1, 1, 1), dtype=torch.float32)] if len(self.global_cache_keys) == 0 else [],
                     }]
            ylen_max = int(math.floor(elens[b] * params['recog_max_len_ratio'])) + 1
            for t in range(ylen_max):
                new_beam = []
                for i_beam in range(len(beam)):
                    # Recurrency (1st) for the main model
                    y = eouts.new_zeros(1, 1).fill_(beam[i_beam]['hyp'][-1]).long()
                    y_emb = self.embed(y)
                    if self.loop_type in ['conditional', 'rnmt']:
                        dstates = self.recurrency_step1(y_emb,
                                                        beam[i_beam]['cv'],
                                                        beam[i_beam]['dstates'])
                    elif self.loop_type in ['normal', 'lmdecoder']:
                        dstates = self.recurrency(y_emb,
                                                  beam[i_beam]['cv'],
                                                  beam[i_beam]['dstates']['dstate'])
                    # Recurrency (1st) for the ensemble
                    ensemble_dstates = []
                    if n_models > 0:
                        for i_e, dec in enumerate(ensemble_decoders):
                            y_emb = dec.embed(y)
                            if dec.loop_type in ['conditional', 'rnmt']:
                                ensemble_dstates += [dec.recurrency_step1(
                                    y_emb,
                                    beam[i_beam]['ensemble_cv'][i_e],
                                    beam[i_beam]['ensemble_dstates'][i_e])]
                            elif dec.loop_type in ['normal', 'lmdecoder']:
                                ensemble_dstates += [dec.recurrency(
                                    y_emb,
                                    beam[i_beam]['ensemble_cv'][i_e],
                                    beam[i_beam]['ensemble_dstates'][i_e]['dstate'])]

                    # Score for the main model
                    cv, aw = self.score(eouts[b:b + 1, :elens[b]],
                                        elens[b:b + 1],
                                        dstates['dout_score'],
                                        beam[i_beam]['aws'][-1])
                    # Score for the ensemble
                    ensemble_cv = []
                    ensemble_aws = []
                    if n_models > 0:
                        for i_e, dec in enumerate(ensemble_decoders):
                            con_vec_e, aw_e = dec.score(
                                ensemble_eouts[i_e][b:b + 1, :ensemble_elens[i_e][b]],
                                ensemble_elens[i_e][b:b + 1],
                                ensemble_dstates[i_e]['dout_score'],
                                beam[i_beam]['ensemble_aws'][i_e][-1])
                            ensemble_cv += [con_vec_e]
                            ensemble_aws += [aw_e]

                    if self.rnnlm_cf:
                        # Update RNNLM states for cold fusion
                        y_lm = eouts.new_zeros(1, 1).fill_(beam[i_beam]['hyp'][-1]).long()
                        y_lm_emb = self.rnnlm_cf.embed(y_lm).squeeze(1)
                        logits_lm_t, lm_out, rnnlm_state = self.rnnlm_cf.predict(
                            y_lm_emb, (beam[i_beam]['rnnlm_hxs'], beam[i_beam]['rnnlm_cxs']))
                    elif rnnlm_weight > 0:
                        # Update RNNLM states for shallow fusion
                        y_lm = eouts.new_zeros(1, 1).fill_(beam[i_beam]['hyp'][-1]).long()
                        y_lm_emb = rnnlm.embed(y_lm).squeeze(1)
                        logits_lm_t, lm_out, rnnlm_state = rnnlm.predict(
                            y_lm_emb, (beam[i_beam]['rnnlm_hxs'], beam[i_beam]['rnnlm_cxs']))
                        lm_out = lm_out.unsqueeze(1)
                    else:
                        logits_lm_t, lm_out, rnnlm_state = None, None, None

                    # Recurrency (2nd, only for the internal decoder) for the main model
                    if self.loop_type in ['conditional', 'rnmt']:
                        dstates = self.recurrency_step2(cv, dstates)
                    # Recurrency (2nd, only for the internal decoder) for the ensemble
                    ensemble_dstates_tmp = []
                    if n_models > 0:
                        for i_e, dec in enumerate(ensemble_decoders):
                            if dec.loop_type in ['conditional', 'rnmt']:
                                ensemble_dstates_tmp += [dec.recurrency_step2(
                                    ensemble_cv[i_e],
                                    ensemble_dstates[i_e])]
                            else:
                                ensemble_dstates_tmp += [ensemble_dstates[i_e]]
                    ensemble_dstates = ensemble_dstates_tmp[:]

                    # Generate for the main model
                    attn_v = self.generate(cv, dstates['dout_gen'], logits_lm_t, lm_out)
                    probs = F.softmax(self.output(attn_v).squeeze(1), dim=1)
                    # NOTE: `[1 (B), 1, vocab]` -> `[1 (B), vocab]`

                    # Generate for RNNLM
                    if rnnlm_weight > 0:
                        lm_probs = F.softmax(logits_lm_t.squeeze(1), dim=1)

                    # Cache decoding
                    exist_cache = len(self.global_cache_keys + beam[i_beam]['local_cache_keys']) > 0
                    cache_probs_sum = torch.zeros_like(probs)
                    cache_probs_lm_sum = torch.zeros_like(probs)
                    cache_theta = 0.2  # smoothing parameter
                    cache_lambda = 0.2  # cache weight
                    if n_caches > 0 and exist_cache:
                        # Compute inner-product over caches
                        cache_keys_all = self.global_cache_keys + beam[i_beam]['local_cache_keys']
                        cache_values_all = self.global_cache_values + beam[i_beam]['local_cache_values']
                        if rnnlm_weight > 0:
                            cache_values_all_lm = self.global_cache_values_lm + beam[i_beam]['local_cache_values_lm']
                        cache_keys = cache_keys_all[-n_caches:]
                        cache_values = torch.cat(cache_values_all[-n_caches:], dim=1)  # `[1, L, dec_n_units]`
                        cache_probs = F.softmax(cache_theta * torch.matmul(
                            cache_values, dstates['dout_gen'].transpose(2, 1)), dim=1)  # `[1, L, 1]`
                        if rnnlm_weight > 0:
                            cache_values_lm = torch.cat(cache_values_all_lm[-n_caches:], dim=1)  # `[1, L, lm_nunits]`
                            cache_probs_lm = F.softmax(cache_theta * torch.matmul(
                                cache_values_lm, lm_out.transpose(2, 1)), dim=1)  # `[1, L, 1]`
                        else:
                            cache_probs_lm = None

                        # Sum all probabilities
                        for c in set(beam[i_beam]['local_cache_keys']):
                            for offset in [i for i, key in enumerate(cache_keys) if key == c]:
                                cache_probs_sum[0, c] += cache_probs[0, offset, 0]
                        probs = (1 - cache_lambda) * probs + cache_lambda * cache_probs_sum
                        if rnnlm_weight > 0:
                            for c in set(beam[i_beam]['local_cache_keys']):
                                for offset in [i for i, key in enumerate(cache_keys) if key == c]:
                                    cache_probs_lm_sum[0, c] += cache_probs_lm[0, offset, 0]
                            lm_probs = (1 - cache_lambda) * lm_probs + cache_lambda * cache_probs_lm_sum
                    else:
                        cache_keys = None
                        cache_probs = None
                        cache_probs_lm = None

                    log_probs = torch.log(probs)
                    # Generate for the ensemble
                    if n_models > 0:
                        for i_e, dec in enumerate(ensemble_decoders):
                            attn_v = dec.generate(ensemble_cv[i_e],
                                                  ensemble_dstates[i_e]['dout_gen'],
                                                  logits_lm_t, lm_out)
                            log_probs += F.log_softmax(dec.output(attn_v).squeeze(1), dim=1)
                        # re-normalize
                        log_probs /= (n_models + 1)
                        # TODO(hirofumi): cache

                    # Pick up the top-k scores
                    log_probs_topk, ids_topk = torch.topk(
                        log_probs, k=beam_width, dim=1, largest=True, sorted=True)

                    scores_att = beam[i_beam]['score_att'] + log_probs_topk
                    local_scores = scores_att.clone()

                    # Add length penalty
                    lp = 1
                    if lp_weight > 0:
                        if gnmt_decoding:
                            lp = (math.pow(5 + (len(beam[i_beam]['hyp']) - 1 + 1), lp_weight)) / math.pow(6, lp_weight)
                            local_scores /= lp
                        else:
                            local_scores += (len(beam[i_beam]['hyp']) - 1 + 1) * lp_weight

                    # Add coverage penalty
                    if cp_weight > 0:
                        aw_mat = torch.stack(beam[i_beam]['aws'][1:] + [aw], dim=-1)  # `[B, T, len(hyp)]`
                        if gnmt_decoding:
                            aw_mat = torch.log(aw_mat.sum(-1))
                            cp = torch.where(aw_mat < 0, aw_mat, torch.zeros_like(aw_mat)).sum()
                            # TODO (hirofumi): mask by elens[b]
                            local_scores += cp * cp_weight
                        else:
                            # Recompute converage penalty in each step
                            if cp_threshold == 0:
                                cp = aw_mat.sum() / self.score.nheads
                            else:
                                cp = torch.where(aw_mat > cp_threshold, aw_mat,
                                                 torch.zeros_like(aw_mat)).sum() / self.score.nheads
                            local_scores += cp * cp_weight
                            # local_scores += (cp - beam[i_beam]['cp_prev']) * cp_weight  # old
                    else:
                        cp = torch.zeros((), dtype=torch.float32)
                        aw_mat = None

                    local_scores *= (1 - ctc_weight)

                    # Add RNNLM score
                    if rnnlm_weight > 0:
                        lm_log_probs = torch.log(lm_probs)
                        scores_lm = beam[i_beam]['score_lm'] + lm_log_probs[0, ids_topk[0]]
                        score_lm_norm = scores_lm / lp  # normalize by length
                        local_scores += score_lm_norm * rnnlm_weight
                    else:
                        scores_lm = torch.zeros((beam_width,), dtype=torch.float32)

                    # CTC score
                    if ctc_weight > 0 and ctc_log_probs is not None:
                        ctc_scores, ctc_states = ctc_prefix_score(
                            beam[i_beam]['hyp'], ids_topk[0], beam[i_beam]['ctc_state'])
                        scores_ctc = beam[i_beam]['score_ctc'] + torch.from_numpy(
                            ctc_scores - beam[i_beam]['ctc_score']).cuda(self.device_id)
                        scores_ctc_norm = scores_ctc / lp  # normalize
                        local_scores += scores_ctc_norm * ctc_weight
                        local_scores, joint_ids_topk = torch.topk(local_scores, beam_width, dim=1)
                        ids_topk = ids_topk[:, joint_ids_topk[0]]
                    else:
                        scores_ctc = torch.zeros((beam_width,), dtype=torch.float32)

                    for k in range(beam_width):
                        top_idx = ids_topk[0, k].item()

                        # Exclude short hypotheses
                        if top_idx == eos and len(beam[i_beam]['hyp']) - 1 < elens[b] * params['recog_min_len_ratio']:
                            continue

                        score_att = scores_att[0, k].item()
                        score_ctc = scores_ctc[k].item()
                        score_lm = scores_lm[k].item()
                        score_t = score_att * (1 - ctc_weight) + score_ctc * ctc_weight + score_lm * rnnlm_weight

                        new_beam.append(
                            {'hyp': beam[i_beam]['hyp'] + [top_idx],
                             'score': local_scores[0, k].item(),
                             #  'scores': beam[i_beam]['scores'] + [score_t],
                             'scores': beam[i_beam]['scores'] + [local_scores[0, k].item()],
                             'scores_cp': beam[i_beam]['scores_cp'] + [cp * cp_weight],
                             'score_att': score_att,  # NOTE: total score
                             'score_cp': cp,
                             'score_ctc': score_ctc,  # NOTE: total score
                             'score_lm': score_lm,  # NOTE: total score
                             'dstates': dstates,
                             'cv': attn_v if self.input_feeding else cv,
                             'aws': beam[i_beam]['aws'] + [aw],
                             'rnnlm_hxs': rnnlm_state[0][:] if rnnlm_state is not None else None,
                             'rnnlm_cxs': rnnlm_state[1][:] if rnnlm_state is not None else None,
                             'cp_prev': cp,
                             'ensemble_dstates': ensemble_dstates,
                             'ensemble_cv': ensemble_cv,
                             'ensemble_aws': ensemble_aws,
                             'ctc_state': ctc_states[joint_ids_topk[0, k]] if ctc_log_probs is not None else None,
                             'ctc_score': ctc_scores[joint_ids_topk[0, k]] if ctc_log_probs is not None else None,
                             'local_cache_keys': beam[i_beam]['local_cache_keys'] + [top_idx],
                             'local_cache_values': beam[i_beam]['local_cache_values'] + [dstates['dout_gen']],
                             'local_cache_values_lm': beam[i_beam]['local_cache_values_lm'] + [lm_out],
                             'cache_keys_history': beam[i_beam]['cache_keys_history'] + [cache_keys] if exist_cache else beam[i_beam]['cache_keys_history'],
                             'cache_probs_history': beam[i_beam]['cache_probs_history'] + [cache_probs] if exist_cache else beam[i_beam]['cache_probs_history'],
                             'cache_probs_history_lm': beam[i_beam]['cache_probs_history_lm'] + [cache_probs_lm] if exist_cache else beam[i_beam]['cache_probs_history_lm'],
                             })

                new_beam = sorted(new_beam, key=lambda x: x['score'], reverse=True)

                # Remove complete hypotheses
                not_complete = []
                for cand in new_beam[:beam_width]:
                    if cand['hyp'][-1] == eos:
                        complete += [cand]
                    else:
                        not_complete += [cand]

                # end detection
                if end_detect(complete, t):
                    logger.info('end detected at %d', t)
                    break

                # Pruning
                if len(complete) >= beam_width:
                    complete = complete[:beam_width]
                    break
                beam = not_complete[:beam_width]

            # Pruning
            if len(complete) == 0:
                complete = beam
            elif len(complete) < nbest and nbest > 1:
                complete.extend(beam[:nbest - len(complete)])

            # backward LM rescoring
            if rnnlm_rev is not None and rnnlm_weight > 0:
                for cand in complete:
                    # Initialize
                    rnnlm_hxs, rnnlm_cxs = None, None
                    score_lm_rev = 0
                    lp = 1

                    # Append <eos>
                    if cand['hyp'][-1] != eos:
                        cand['hyp'].append(eos)

                    if lp_weight > 0 and gnmt_decoding:
                        lp = (math.pow(5 + (len(cand['hyp']) - 1 + 1), lp_weight)) / math.pow(6, lp_weight)
                    for t in range(len(cand['hyp'][1:])):
                        y_lm = eouts.new_zeros(1, 1).fill_(cand['hyp'][-1 - t]).long()
                        y_lm_emb = rnnlm_rev.embed(y_lm).squeeze(1)
                        logits_lm_t, _, (rnnlm_hxs, rnnlm_cxs) = rnnlm_rev.predict(
                            y_lm_emb, (rnnlm_hxs, rnnlm_cxs))
                        lm_log_probs = F.log_softmax(logits_lm_t.squeeze(1), dim=1)
                        score_lm_rev += lm_log_probs[0, cand['hyp'][-2 - t]]
                    score_lm_rev /= lp  # normalize
                    cand['score'] += score_lm_rev * rnnlm_weight
                    cand['score_lm_rev'] = score_lm_rev * rnnlm_weight

            # Sort by score
            complete = sorted(complete, key=lambda x: x['score'], reverse=True)

            # N-best list
            if self.backward:
                # Reverse the order
                nbest_hyps += [[np.array(complete[n]['hyp'][1:][::-1]) for n in range(nbest)]]
                if self.score.nheads > 1:
                    aws += [[complete[n]['aws'][0, 1:][::-1] for n in range(nbest)]]
                else:
                    aws += [[complete[n]['aws'][1:][::-1] for n in range(nbest)]]
                scores += [[complete[n]['scores'][1:][::-1] for n in range(nbest)]]
                scores_cp += [[complete[n]['scores_cp'][1:][::-1] for n in range(nbest)]]
            else:
                nbest_hyps += [[np.array(complete[n]['hyp'][1:]) for n in range(nbest)]]
                if self.score.nheads > 1:
                    aws += [[complete[n]['aws'][0, 1:] for n in range(nbest)]]
                else:
                    aws += [[complete[n]['aws'][1:] for n in range(nbest)]]
                scores += [[complete[n]['scores'][1:] for n in range(nbest)]]
                scores_cp += [[complete[n]['scores_cp'][1:] for n in range(nbest)]]

            # Check <eos>
            eos_flag = [True if complete[n]['hyp'][-1] == eos else False for n in range(nbest)]
            eos_flags.append(eos_flag)

            if id2token is not None:
                if refs is not None:
                    logger.info('Ref: %s' % refs[b])
                for n in range(nbest):
                    logger.info('Hyp: %s' % id2token(nbest_hyps[0][n]))
            if refs is not None:
                logger.info('log prob (ref): ')
            for n in range(nbest):
                logger.info('log prob (hyp): %.7f' % complete[n]['score'])
                logger.info('log prob (hyp, att): %.7f' % (complete[n]['score_att'] * (1 - ctc_weight)))
                logger.info('log prob (hyp, cp): %.7f' % (complete[n]['score_cp'] * cp_weight))
                if ctc_weight > 0 and ctc_log_probs is not None:
                    logger.info('log prob (hyp, ctc): %.7f' % (complete[n]['score_ctc'] * ctc_weight))
                if rnnlm_weight > 0:
                    logger.info('log prob (hyp, lm): %.7f' % (complete[n]['score_lm'] * rnnlm_weight))
                    if rnnlm_rev is not None:
                        logger.info('log prob (hyp, lm rev): %.7f' % (complete[n]['score_lm_rev']))

        # Concatenate in L dimension
        for b in range(len(aws)):
            for n in range(nbest):
                aws[b][n] = tensor2np(torch.stack(aws[b][n], dim=1).squeeze(0))

        # Exclude <eos> (<sos> in case of the backward decoder)
        if exclude_eos:
            if self.backward:
                nbest_hyps = [[nbest_hyps[b][n][1:] if eos_flags[b][n]
                               else nbest_hyps[b][n] for n in range(nbest)] for b in range(bs)]
            else:
                nbest_hyps = [[nbest_hyps[b][n][:-1] if eos_flags[b][n]
                               else nbest_hyps[b][n] for n in range(nbest)] for b in range(bs)]

        # Store in cache
        if n_caches > 0:
            hyp_len = len(complete[0]['hyp'][1:])
            self.store_global_cache(complete[0]['local_cache_keys'],
                                    complete[0]['local_cache_values'],
                                    complete[0]['local_cache_values_lm'],
                                    n_caches)
            cache_keys_history = complete[0]['cache_keys_history']
            cache_probs_history = torch.zeros(
                (1, complete[0]['cache_probs_history'][-1].size(1), hyp_len), dtype=torch.float32)
            for i, p in enumerate(complete[0]['cache_probs_history']):
                if p.size(1) < n_caches:
                    cache_probs_history[0, :p.size(1), i] = p[0, :, 0].cpu()
                else:
                    cache_probs_history[0, :n_caches - (hyp_len - 1 - i), i] = p[0, (hyp_len - 1 - i):, 0].cpu()
            if rnnlm_weight > 0:
                cache_probs_history_lm = torch.zeros(
                    (1, complete[0]['cache_probs_history_lm'][-1].size(1), hyp_len), dtype=torch.float32)
                for i, p in enumerate(complete[0]['cache_probs_history_lm']):
                    if p.size(1) < n_caches:
                        cache_probs_history_lm[0, :p.size(1), i] = p[0, :, 0].cpu()
                    else:
                        cache_probs_history_lm[0, :n_caches - (hyp_len - 1 - i), i] = p[0, (hyp_len - 1 - i):, 0].cpu()
            else:
                cache_probs_history_lm = None
        else:
            cache_keys_history = None
            cache_probs_history = None
            cache_probs_history_lm = None

        # return nbest_hyps, aws, scores, scores_cp, (cache_probs_history, cache_keys_history)
        return nbest_hyps, aws, scores, scores_cp, (cache_probs_history_lm, cache_keys_history)

    def store_global_cache(self, keys, values, values_lm, n_caches):
        self.global_cache_keys += keys
        self.global_cache_values += values
        self.global_cache_values_lm += values_lm

        # Truncate cache
        if len(self.global_cache_keys) > n_caches:
            self.global_cache_keys = self.global_cache_keys[:n_caches]
            self.global_cache_values = self.global_cache_values[:n_caches]
            self.global_cache_values_lm = self.global_cache_values_lm[:n_caches]

    def reset_global_cache(self):
        """Reset global cache when the speaker/session is changed
        """
        self.global_cache_keys = []
        self.global_cache_values = []
        self.global_cache_values_lm = []

    def decode_ctc(self, eouts, xlens, beam_width=1, rnnlm=None):
        """Decoding by the CTC layer in the inference stage.

            This is only used for Joint CTC-Attention model.
        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            beam_width (int): size of beam
            rnnlm ():
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`

        """
        log_probs = F.log_softmax(self.output_ctc(eouts), dim=-1)
        if beam_width == 1:
            best_hyps = self.decode_ctc_greedy(log_probs, xlens)
        else:
            best_hyps = self.decode_ctc_beam(log_probs, xlens, beam_width, rnnlm)
            # TODO(hirofumi): add decoding paramters
        return best_hyps

    def ctc_posteriors(self, eouts, xlens, temperature, topk):
        # Path through the softmax layer
        logits_ctc = self.output_ctc(eouts)
        ctc_probs = F.softmax(logits_ctc / temperature, dim=-1)
        if topk is None:
            topk = ctc_probs.size(-1)
        _, ids_topk = torch.topk(ctc_probs.sum(1), k=topk, dim=-1, largest=True, sorted=True)
        return tensor2np(ctc_probs), tensor2np(ids_topk)


def end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
    """End detection for joint CTC-attention

    desribed in Eq. (50) of S. Watanabe et al
    "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"

    Args:
        ended_hyps (list):
        i (int):
        M (int):
        D_end (float):
    Returns:
        bool

    [Reference]:
        https://github.com/espnet/espnet

    """
    if len(ended_hyps) == 0:
        return False
    count = 0
    best_hyp = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[0]
    for m in range(M):
        # get ended_hyps with their length is i - m
        hyp_len = i - m
        hyps_same_length = [x for x in ended_hyps if len(x['hyp']) - 1 == hyp_len]
        # NOTE: key:hyp includes <sos>
        if len(hyps_same_length) > 0:
            best_hyp_same_length = sorted(hyps_same_length, key=lambda x: x['score'], reverse=True)[0]
            if best_hyp_same_length['score'] - best_hyp['score'] < D_end:
                count += 1

    if count == M:
        return True
    else:
        return False
