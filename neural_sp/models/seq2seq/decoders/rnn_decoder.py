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
        rnnlm (RNNLM):
        lm_fusion_type (str): the type of RNNLM fusion
        n_caches (int):
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
                 rnnlm,
                 lm_fusion_type,
                 n_caches,
                 rnnlm_init,
                 lmobj_weight,
                 share_lm_softmax,
                 global_weight,
                 mtl_per_batch):

        super(RNNDecoder, self).__init__()

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
        self.layerwise_attn = layerwise_attention
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
        self.rnnlm = rnnlm
        self.lm_fusion_type = lm_fusion_type
        self.n_caches = n_caches
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
        self.fifo_cache_ids = []
        self.fifo_cache_sp_key = None
        self.fifo_cache_lm_key = None
        self.prev_spk = ''
        self.dstates_final = None
        self.lmstate_final = (None, None)
        self.cache_fusion = (self.rnnlm is not None and 'cache' in self.lm_fusion_type)

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
                    key_dim=self.enc_n_units,
                    query_dim=n_units if n_projs == 0 else n_projs,
                    attn_type=attn_type,
                    attn_dim=attn_dim,
                    sharpening_factor=attn_sharpening_factor,
                    sigmoid_smoothing=attn_sigmoid_smoothing,
                    conv_out_channels=attn_conv_out_channels,
                    conv_kernel_size=attn_conv_kernel_size,
                    n_heads=attn_n_heads,
                    dropout=dropout_att)
            else:
                self.score = AttentionMechanism(
                    key_dim=self.enc_n_units,
                    query_dim=n_units if n_projs == 0 else n_projs,
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
                if lm_fusion_type in ['cold_hidden_recurrency', 'cache_dot_recurrency', 'cache_add_recurrency']:
                    dec_idim += n_units
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
            dout_dim = n_projs if self.n_projs > 0 else n_units
            if self.rnnlm is not None:
                self.cf_linear_dec_feat = LinearND(dout_dim + enc_n_units, n_units)

                if 'cache' in self.lm_fusion_type:
                    assert n_caches > 0

                if lm_fusion_type == 'cold_hidden_generate':
                    self.cf_linear_lm_feat = LinearND(self.rnnlm.n_units, n_units)
                    self.cf_linear_lm_gate = LinearND(n_units * 2, n_units)
                elif lm_fusion_type == 'cold_prob_generate':
                    self.cf_linear_lm_feat = LinearND(self.rnnlm.vocab, n_units)
                    self.cf_linear_lm_gate = LinearND(n_units * 2, n_units)
                elif lm_fusion_type == 'cold_hidden_recurrency':
                    self.cf_linear_lm_feat = LinearND(self.rnnlm.n_units, n_units)
                    self.cf_linear_lm_gate = LinearND(n_units * 2, n_units)
                elif lm_fusion_type in ['cache_dot_generate', 'cache_dot_recurrency',
                                        'cache_add_generate', 'cache_add_recurrency',
                                        'cache_dot_generate_unfreeze', 'cache_add_generate_unfreeze',
                                        'cache_cold_dot_generate']:
                    self.cf_linear_lm_feat = LinearND(self.rnnlm.n_units, n_units)
                    self.score_cf = AttentionMechanism(
                        key_dim=self.rnnlm.n_units,
                        query_dim=n_units + self.rnnlm.n_units,
                        attn_type='dot' if 'dot' in lm_fusion_type else 'add',
                        attn_dim=n_units)
                    self.sentinel = LinearND(n_units, 1, bias=False)
                    if lm_fusion_type == 'cache_cold_dot_generate':
                        self.cf_linear_lm_gate = LinearND(n_units * 2, n_units)
                else:
                    raise ValueError(lm_fusion_type)
                if lm_fusion_type in ['cold_hidden_recurrency', 'cache_dot_recurrency', 'cache_add_recurrency']:
                    self.output_bn = LinearND(n_units + enc_n_units, n_units)
                elif lm_fusion_type == 'cache_cold_dot_generate':
                    self.output_bn = LinearND(n_units * 3, n_units)
                else:
                    self.output_bn = LinearND(n_units * 2, n_units)

                # fix RNNLM parameters
                if 'unfreeze' in lm_fusion_type:
                    for p in self.rnnlm.parameters():
                        p.requires_grad = True
                else:
                    for p in self.rnnlm.parameters():
                        p.requires_grad = False
            else:
                self.output_bn = LinearND(dout_dim + enc_n_units, n_units)

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

    def forward(self, eouts, elens, ys, task='all', ys_cache=[]):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            task (str): all or ys or ys_sub*
            ys_cache (list):
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None,
                       'loss_att': None, 'loss_ctc': None,
                       'loss_lmobj': None, 'loss_rnnlm': None,
                       'acc_att': None, 'acc_lmobj': None,
                       'ppl_att': None, 'ppl_lmobj': None, 'ppl_rnnlm': None}
        loss = eouts.new_zeros((1,))

        # if self.rnnlm is not None:
        #     self.rnnlm.eval()

        # CTC loss
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task):
            loss_ctc = self.forward_ctc(eouts, elens, ys)
            observation['loss_ctc'] = loss_ctc.item()
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight

        # LM objective for the decoder
        if self.lmobj_weight > 0 and (task == 'all' or 'lmobj' in task):
            loss_lmobj, acc_lmobj, ppl_lmobj = self.forward_lmobj(ys)
            observation['loss_lmobj'] = loss_lmobj.item()
            observation['acc_lmobj'] = acc_lmobj
            observation['ppl_lmobj'] = ppl_lmobj
            if self.mtl_per_batch:
                loss += loss_lmobj
            else:
                loss += loss_lmobj * self.lmobj_weight

        # RNNLM joint training
        if self.rnnlm is not None and 'mtl' in self.lm_fusion_type and (task == 'all' or 'rnnlm' in task):
            loss_rnnlm, ppl_rnnlm = self.forward_rnnlm(ys, ys_cache)
            observation['loss_rnnlm'] = loss_rnnlm.item()
            observation['ppl_rnnlm'] = ppl_rnnlm
            loss += loss_rnnlm

        # XE loss
        if self.global_weight - self.ctc_weight > 0 and (task == 'all' or ('ctc' not in task and 'lmobj' not in task and 'rnnlm' not in task)):
            loss_att, acc_att, ppl_att = self.forward_att(eouts, elens, ys, ys_cache)
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
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[B, L, vocab]`

        """
        logits = self.output_ctc(eouts)

        # Compute the auxiliary CTC loss
        elens_ctc = np2tensor(np.fromiter(elens, dtype=np.int64), -1).int()
        ys_ctc = [np2tensor(np.fromiter(y, dtype=np.int64)).long() for y in ys]  # always fwd
        ylens = np2tensor(np.fromiter([y.size(0) for y in ys_ctc], dtype=np.int64), -1).int()
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
        eos = w.new_zeros((1,)).fill_(self.eos).long()
        ys = [np2tensor(np.fromiter(y[::-1] if self.backward else y, dtype=np.int64),
                        self.device_id).long() for y in ys]
        ys_in = [torch.cat([eos, y], dim=0) for y in ys]
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
            # Recurrency
            dstates = self.recurrency(ys_emb[:, t:t + 1], cv, dstates['dstate'])

            # Generate
            if self.loop_type == 'lmdecoder':
                logits_t = self.output_lmobj(dstates['dout_lmdec'])
            elif self.loop_type == 'normal':
                attn_v, _, _ = self.generate(cv, dstates['dout_gen'])
                logits_t = self.output(attn_v)
            logits.append(logits_t)

        # Compute XE loss for RNNLM objective
        logits = torch.cat(logits, dim=1)
        loss = F.cross_entropy(logits.view((-1, logits.size(2))),
                               ys_out_pad.view(-1),
                               ignore_index=-1, size_average=False) / bs

        # Compute token-level accuracy in teacher-forcing
        pad_pred = logits.view(ys_out_pad.size(0), ys_out_pad.size(1), logits.size(-1)).argmax(2)
        mask = ys_out_pad != -1
        numerator = torch.sum(pad_pred.masked_select(mask) == ys_out_pad.masked_select(mask))
        denominator = torch.sum(mask)
        acc = float(numerator) * 100 / float(denominator)
        ppl = np.exp(loss.item())

        return loss, acc, ppl

    def forward_rnnlm(self, ys, ys_cache):
        """Compute XE loss for RNNLM during LM fusion.

        Args:
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            ys_cache (list):
        Returns:
            loss (FloatTensor): `[1]`
            ppl (float):

        """
        if len(ys_cache) > 0:
            ys = [ys_cache[b] + y for b, y in enumerate(ys)]
        loss, _, _ = self.rnnlm(ys)
        ppl = np.exp(loss.item())
        return loss, ppl

    def forward_att(self, eouts, elens, ys, ys_cache):
        """Compute XE loss for the attention-based sequence-to-sequence model.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            ys_cache (list):
        Returns:
            loss (FloatTensor): `[B, L, vocab]`
            acc (float):
            ppl (float):

        """
        bs = eouts.size(0)

        # Append <sos> and <eos>
        eos = eouts.new_zeros(1).fill_(self.eos).long()
        _ys = [np2tensor(np.fromiter(y[::-1] if self.backward else y, dtype=np.int64),
                         self.device_id).long() for y in ys]
        ys_in = [torch.cat([eos, y], dim=0) for y in _ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in _ys]
        ys_in_pad = pad_list(ys_in, self.pad)
        ys_out_pad = pad_list(ys_out, -1)

        # Initialization
        dstates = self.init_dec_state(bs, self.n_layers, eouts, elens)
        cv = eouts.new_zeros(bs, 1, self.enc_n_units)
        attn_v = eouts.new_zeros(bs, 1, self.dec_n_units)
        self.score.reset()
        aw = None
        lmstate = (None, None)
        lm_feat = eouts.new_zeros(bs, 1, self.dec_n_units)

        # Pre-computation of embedding
        ys_emb = self.embed(ys_in_pad)
        if self.rnnlm is not None:
            ys_lm_emb = self.rnnlm.encode(ys_in_pad)

        # Pre-computation of RNNLM states for history
        if self.cache_fusion or self.n_caches > 0:
            # NOTE: ys_cache already includes <sos> and <eos>
            ys_cache_pad = pad_list([np2tensor(np.fromiter(y, dtype=np.int64), self.device_id).long()
                                     for y in ys_cache], self.pad)
            ys_lm_emb_cache = self.rnnlm.encode(ys_cache_pad)
            ylens_cache = [len(y) for y in ys_cache]

            lmouts = []
            hxs = [None] * bs
            cxs = [None] * bs
            for t_c in range(max(ylens_cache)):
                lmout, lmstate = self.rnnlm.decode(ys_lm_emb_cache[:, t_c:t_c + 1], lmstate)
                for b in [i for i, ylen in enumerate(ylens_cache) if ylen == t_c + 1]:
                    hxs[b] = lmstate[0][:, b:b + 1]
                    if self.rnnlm.rnn_type == 'lstm':
                        cxs[b] = lmstate[1][:, b:b + 1]
                lmouts.append(lmout)

            # ylens_cache[b] == 0 case
            for b in [i for i, ylen in enumerate(ylens_cache) if ylen == 0]:
                hxs_b, cxs_b = self.rnnlm.initialize_hidden(batch_size=1)
                hxs[b] = hxs_b
                if self.rnnlm.rnn_type == 'lstm':
                    cxs[b] = cxs_b

            # Concatenate lmstate, lmouts
            hxs = torch.cat(hxs, dim=1)  # `[lm_n_layers, B, lm_n_units]`
            if self.rnnlm.rnn_type == 'lstm':
                cxs = torch.cat(cxs, dim=1)
            lmstate = (hxs, cxs)

            # Pad the leftmost in lmouts
            if self.cache_fusion:
                lmouts = torch.cat(lmouts, dim=1)  # `[B, cache_size, lm_n_units]`
                lmouts_list = []
                for b in range(bs):
                    pad_len = self.n_caches - ylens_cache[b]
                    if pad_len > 0:
                        lmouts_list.append(lmouts[b, :ylens_cache[b]])
                    else:
                        lmouts_list.append(ys_lm_emb_cache.new_zeros(pad_len, self.rnnlm.n_units))
                self.fifo_cache_lm_key = pad_list(lmouts_list, pad_left=True)[:, -self.n_caches:]
                # NOTE: RNNLM are not trained so as to decode <pad> tokens

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
            if self.lm_fusion_type in ['cold_hidden_recurrency', 'cache_dot_recurrency', 'cache_add_recurrency']:
                dec_in = torch.cat([dec_in, lm_feat], dim=-1)
            if self.loop_type in ['conditional', 'rnmt']:
                dstates = self.recurrency_step1(y_emb, dec_in, dstates)
            elif self.loop_type in ['normal', 'lmdecoder']:
                dstates = self.recurrency(y_emb, dec_in, dstates['dstate'])

            # Update RNNLM states for cold fusion
            lmout = None
            if self.rnnlm is not None:
                if is_sample:
                    y_lm_emb = self.rnnlm.encode(logits[-1].detach().argmax(-1))
                else:
                    y_lm_emb = ys_lm_emb[:, t:t + 1]
                lmout, lmstate = self.rnnlm.decode(y_lm_emb, lmstate)

            # Score
            cv, aw = self.score(eouts, elens, eouts, dstates['dout_score'], aw)

            # Recurrency (2nd, only for the internal decoder)
            if self.loop_type in ['conditional', 'rnmt']:
                dstates = self.recurrency_step2(cv, dstates)

            # Generate
            attn_v, _, lm_feat = self.generate(cv, dstates['dout_gen'], lmout,
                                               cache_keys=self.fifo_cache_lm_key)
            logits.append(self.output(attn_v))
        logits = torch.cat(logits, dim=1)

        # Reset cache
        self.fifo_cache_lm_key = None

        # Compute XE sequence loss
        if self.lsm_prob > 0:
            # Label smoothing
            loss = cross_entropy_lsm(
                logits,
                ys=ys_out_pad,
                ylens=[y.size(0) for y in ys_out],
                lsm_prob=self.lsm_prob, size_average=True)
        else:
            loss = F.cross_entropy(
                logits.view((-1, logits.size(2))),
                ys_out_pad.view(-1),  # long
                ignore_index=-1, size_average=False) / bs

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
        ppl = np.exp(loss.item())

        return loss, acc, ppl

    def init_dec_state(self, bs, n_layers, eouts=None, elens=None):
        """Initialize decoder state.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (list): A list of length `[B]`
            n_layers (int):
        Returns:
            dstates (dict):
                dout (FloatTensor): `[B, 1, dec_n_units]`
                dstate (tuple): A tuple of (hxs, cxs)
                    hxs (list of FloatTensor):
                    cxs (list of FloatTensor):

        """
        dstates = {'dout_score': None,  # for attention scoring
                   'dout_gen': None,  # for token generation
                   'dstate': None,
                   'dstate1': None, 'dstate2': None}
        w = next(self.parameters())
        zero_state = w.new_zeros((bs, self.dec_n_units))
        dstates['dout_score'] = w.new_zeros((bs, 1, self.dec_n_units))
        dstates['dout_gen'] = w.new_zeros((bs, 1, self.dec_n_units))
        if self.loop_type in ['conditional', 'rnmt']:
            hxs1 = [zero_state for l in range(1)]
            cxs1 = [zero_state for l in range(1)] if self.rnn_type == 'lstm' else []
            dstates['dstate1'] = (hxs1, cxs1)
            hxs2 = [zero_state for l in range(self.n_layers - 1)]
            cxs2 = [zero_state for l in range(self.n_layers - 1)] if self.rnn_type == 'lstm' else []
            dstates['dstate2'] = (hxs2, cxs2)
        else:
            hxs = [zero_state for l in range(self.n_layers)]
            cxs = [zero_state for l in range(self.n_layers)] if self.rnn_type == 'lstm' else []
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

        dstates_new = {'dout_score': None,  # for attention scoring
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
        """Recurrency function for the internal decoder (before attention scoring).

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
        """Recurrency function for the internal decoder (after attention scoring).

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
                       'dstate1': None, 'dstate2': None}

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

    def generate(self, cv, dout, lmout, cache_keys):
        """Generate function.

        Args:
            cv (FloatTensor): `[B, 1, enc_n_units]`
            dout (FloatTensor): `[B, 1, dec_n_units]`
            lmout (FloatTensor): `[B, 1, lm_n_units]`
            cache_keys (FloatTensor): `[B, n_caches, cache_dim]`
        Returns:
            attn_v (FloatTensor): `[B, 1, vocab]`
            aw_cache (FloatTensor): `[B, n_caches, 1]`
            gated_lm_feat (FloatTensor): `[B, 1 , dec_n_units]`

        """
        aw_cache = None
        gated_lm_feat = None
        if self.rnnlm is not None:
            # cold fusion
            dec_feat = self.cf_linear_dec_feat(torch.cat([dout, cv], dim=-1))
            if self.lm_fusion_type in ['cold_hidden_recurrency']:
                lm_feat = self.cf_linear_lm_feat(lmout)
                gate = torch.sigmoid(self.cf_linear_lm_gate(torch.cat([dec_feat, lm_feat], dim=-1)))
                gated_lm_feat = gate * lm_feat  # element-wise
                out = self.output_bn(torch.cat([dout, cv], dim=-1))
            elif self.lm_fusion_type in ['cache_dot_recurrency', 'cache_add_recurrency']:
                self.score_cf.reset()
                cache_lens = [cache_keys.size(1)] * dout.size(0)
                e_cache = self.score_cf(cache_keys, cache_lens,
                                        value=cache_keys,
                                        query=torch.cat([dec_feat, lmout], dim=-1),
                                        return_logits=True)  # `[B, n_caches]`
                sentinel_vec = self.sentinel(dec_feat)  # `[B, 1, 1]`
                e_cache = torch.cat([e_cache, sentinel_vec.squeeze(2)], dim=1)
                aw_cache = F.softmax(e_cache, dim=-1)  # `[B, n_caches + 1]`
                gate = aw_cache[:, -1].unsqueeze(1).unsqueeze(2)
                aw_cache = aw_cache[:, :-1]
                cv_cache = torch.matmul(aw_cache.unsqueeze(1), cache_keys)
                aw_cache = aw_cache.unsqueeze(2)
                lm_feat = self.cf_linear_lm_feat(cv_cache)
                gated_lm_feat = gate * lm_feat
                out = self.output_bn(torch.cat([dout, cv], dim=-1))
            else:
                if self.lm_fusion_type == 'cold_hidden_generate':
                    lm_feat = self.cf_linear_lm_feat(lmout)
                    gate = torch.sigmoid(self.cf_linear_lm_gate(torch.cat([dec_feat, lm_feat], dim=-1)))
                    gated_lm_feat = gate * lm_feat
                elif self.lm_fusion_type == 'cold_prob_generate':
                    lm_feat = self.cf_linear_lm_feat(self.rnnlm.generate(lmout))
                    gate = torch.sigmoid(self.cf_linear_lm_gate(torch.cat([dec_feat, lm_feat], dim=-1)))
                    gated_lm_feat = gate * lm_feat
                elif self.lm_fusion_type in ['cache_dot_generate', 'cache_add_generate',
                                             'cache_dot_generate_unfreeze', 'cache_add_generate_unfreeze',
                                             'cache_cold_dot_generate']:
                    self.score_cf.reset()
                    cache_lens = [cache_keys.size(1)] * dout.size(0)
                    e_cache = self.score_cf(cache_keys, cache_lens,
                                            value=cache_keys,
                                            query=torch.cat([dec_feat, lmout], dim=-1),
                                            return_logits=True)  # `[B, n_caches]`
                    sentinel_vec = self.sentinel(dec_feat)  # `[B, 1, 1]`
                    e_cache = torch.cat([e_cache, sentinel_vec.squeeze(2)], dim=1)
                    aw_cache = F.softmax(e_cache, dim=-1)  # `[B, n_caches + 1]`
                    gate_cache = aw_cache[:, -1].unsqueeze(1).unsqueeze(2)
                    aw_cache = aw_cache[:, :-1]
                    cv_cache = torch.matmul(aw_cache.unsqueeze(1), cache_keys)
                    aw_cache = aw_cache.unsqueeze(2)
                    lm_feat_cache = self.cf_linear_lm_feat(cv_cache)
                    gated_lm_feat = gate_cache * lm_feat_cache
                    if self.lm_fusion_type == 'cache_cold_dot_generate':
                        lm_feat = self.cf_linear_lm_feat(lmout)
                        gate = torch.sigmoid(self.cf_linear_lm_gate(torch.cat([dec_feat, lm_feat], dim=-1)))
                        gated_lm_feat = torch.cat([gate * lm_feat, gated_lm_feat], dim=-1)

                out = self.output_bn(torch.cat([dec_feat, gated_lm_feat], dim=-1))
        else:
            out = self.output_bn(torch.cat([dout, cv], dim=-1))
        attn_v = torch.tanh(out)
        return attn_v, aw_cache, gated_lm_feat

    def greedy(self, eouts, elens, max_len_ratio, exclude_eos=False, speakers=None):
        """Greedy decoding in the inference stage (used only for evaluation during training).

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (list): A list of length `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            exclude_eos (bool):
            speakers (list):
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
        lmstate = (None, None)
        lm_feat = eouts.new_zeros(bs, 1, self.dec_n_units)

        # Start from <sos> (<eos> in case of the backward decoder)
        y = eouts.new_zeros(bs, 1).fill_(self.eos).long()

        if self.n_caches > 0:
            assert bs == 1
            if speakers[0] != self.prev_spk:
                self.reset_global_cache()
            else:
                lmstate = self.lmstate_final
            # NOTE: speech cache is not used here
            self.prev_spk = speakers[0]

        best_hyps_tmp, aws_tmp = [], []
        ylens = np.zeros((bs,), dtype=np.int64)
        eos_flags = [False] * bs
        fifo_cache_ids_local = []
        fifo_cache_lm_key_local = []
        for t in range(int(math.floor(max_xlen * max_len_ratio)) + 1):
            # Recurrency (1st)
            y_emb = self.embed(y)
            dec_in = attn_v if self.input_feeding else cv
            if 'update' in self.lm_fusion_type:
                dec_in = torch.cat([dec_in, lm_feat], dim=-1)
            if self.loop_type in ['conditional', 'rnmt']:
                dstates = self.recurrency_step1(y_emb, dec_in, dstates)
            elif self.loop_type in ['normal', 'lmdecoder']:
                dstates = self.recurrency(y_emb, dec_in, dstates['dstate'])

            # Update RNNLM states for cold fusion
            lmout = None
            if self.rnnlm is not None:
                lmout, lmstate = self.rnnlm.decode(self.rnnlm.encode(y), lmstate)
                if self.cache_fusion:
                    fifo_cache_lm_key_local.append(lmout)  # features for prediction

            # Score
            cv, aw = self.score(eouts, elens, eouts, dstates['dout_score'], aw)

            # Recurrency (2nd, only for the internal decoder)
            if self.loop_type in ['conditional', 'rnmt']:
                dstates = self.recurrency_step2(cv, dstates)

            # Generate
            memory = None
            if self.cache_fusion:
                if self.fifo_cache_lm_key is None:
                    memory = eouts.new_zeros(1, 1, self.rnnlm.n_units)
                else:
                    memory = self.fifo_cache_lm_key
            attn_v, _, lm_feat = self.generate(cv, dstates['dout_gen'], lmout, cache_keys=memory)
            logits_t = self.output(attn_v)

            # Pick up 1-best
            y = logits_t.detach().argmax(-1)
            best_hyps_tmp += [y]
            if self.score.n_heads > 1:
                aws_tmp += [aw[0]]
            else:
                aws_tmp += [aw]

            if self.cache_fusion:
                fifo_cache_ids_local += [y]  # predicted label (not previous label)

            # Count lengths of hypotheses
            for b in range(bs):
                if not eos_flags[b]:
                    if y[b].item() == self.eos:
                        eos_flags[b] = True
                    ylens[b] += 1
                    # NOTE: include <eos>

            # Break if <eos> is outputed in all mini-bs
            if sum(eos_flags) == bs:
                break

        # RNNLM state carry over
        self.lmstate_final = lmstate

        if self.cache_fusion:
            fifo_cache_lm_key_local = torch.cat(fifo_cache_lm_key_local, dim=1)
            self.fifo_cache_lm_key = torch.cat([self.fifo_cache_lm_key, fifo_cache_lm_key_local], dim=1)
            self.fifo_cache_ids += fifo_cache_ids_local
            # Truncate
            self.fifo_cache_lm_key = self.fifo_cache_lm_key[:, -self.n_caches:]
            self.fifo_cache_ids = self.fifo_cache_ids[-self.n_caches:]

        # Concatenate in L dimension
        best_hyps_tmp = tensor2np(torch.cat(best_hyps_tmp, dim=1))
        aws_tmp = tensor2np(torch.stack(aws_tmp, dim=1))

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
                best_hyps = [best_hyps[b][1:] if eos_flags[b] else best_hyps[b] for b in range(bs)]
            else:
                best_hyps = [best_hyps[b][:-1] if eos_flags[b] else best_hyps[b] for b in range(bs)]

        return best_hyps, aws

    def beam_search(self, eouts, elens, params, rnnlm, rnnlm_rev=None, ctc_log_probs=None,
                    nbest=1, exclude_eos=False, idx2token=None, refs=None,
                    ensemble_eouts=None, ensemble_elens=None, ensemble_decoders=[],
                    speakers=None):
        """Beam search decoding in the inference stage.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (list): A list of length `[B]`
            params (dict):
                beam_width (int): size of beam
                max_len_ratio (int): maximum sequence length of tokens
                min_len_ratio (float): minimum sequence length of tokens
                length_penalty (float): length penalty
                coverage_penalty (float): coverage penalty
                coverage_threshold (float): threshold for coverage penalty
                lm_weight (float): weight of RNNLM score
                n_caches (int):
            rnnlm (torch.nn.Module):
            rnnlm_rev (torch.nn.Module):
            ctc_log_probs (torch.FloatTensor):
            nbest (int):
            exclude_eos (bool):
            idx2token (): converter from index to token
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

        oracle = params['recog_oracle']
        beam_width = params['recog_beam_width']
        ctc_weight = params['recog_ctc_weight']
        lp_weight = params['recog_length_penalty']
        cp_weight = params['recog_coverage_penalty']
        cp_threshold = params['recog_coverage_threshold']
        lm_weight = params['recog_rnnlm_weight']
        gnmt_decoding = params['recog_gnmt_decoding']
        asr_state_carry_over = params['recog_asr_state_carry_over']
        lm_state_carry_over = params['recog_rnnlm_state_carry_over']
        n_caches = params['recog_n_caches']
        cache_theta_sp = params['recog_cache_theta_speech']
        cache_lambda_sp = params['recog_cache_lambda_speech']
        cache_theta_lm = params['recog_cache_theta_lm']
        cache_lambda_lm = params['recog_cache_lambda_lm']
        cache_type = params['recog_cache_type']
        concat_prev_n_utterances = params['recog_concat_prev_n_utterances']

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

        nbest_hyps, aws, scores, scores_cp = [], [], [], []
        eos_flags = []
        for b in range(bs):
            # Initialization per utterance
            dstates = self.init_dec_state(1, self.n_layers, eouts[b:b + 1], elens[b:b + 1])
            cv = eouts.new_zeros(1, 1, self.dec_n_units if self.input_feeding else self.enc_n_units)
            self.score.reset()
            lm_hxs, lm_cxs = None, None

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

            if speakers[b] != self.prev_spk:
                self.reset_global_cache()
            else:
                if lm_state_carry_over:
                    lm_hxs, lm_cxs = self.lmstate_final
                if asr_state_carry_over:
                    dstates = self.dstates_final
            self.prev_spk = speakers[b]

            complete = []
            beam = [{'hyp': [self.eos],
                     'score': 0,
                     'scores': [0],
                     'scores_cp': [0],
                     'score_attn': 0,
                     'score_ctc': 0,
                     'score_lm': 0,
                     'dstates': dstates,
                     'cv': cv,
                     'aws': [None],
                     'lm_hxs': lm_hxs,
                     'lm_cxs': lm_cxs,
                     'cp_prev': 0,
                     'ensemble_dstates': ensemble_dstates,
                     'ensemble_cv': ensemble_cv,
                     'ensemble_aws':[[None] for _ in range(n_models)],
                     'ctc_state':  ctc_prefix_score.initial_state() if ctc_weight > 0 and ctc_log_probs is not None else None,
                     'ctc_score': 0,
                     'cache_idx': [],
                     'cache_sp_key': [],
                     'cache_lm_key': [],
                     'cache_id_hist': [],
                     'cache_sp_attn_hist': [],
                     'cache_lm_attn_hist': [],
                     }]
            if oracle:
                assert refs is not None
                ylen_max = len(refs[b])
            else:
                ylen_max = int(math.floor(elens[b] * params['recog_max_len_ratio'])) + 1
            for t in range(ylen_max):
                new_beam = []
                for i_beam in range(len(beam)):
                    prev_idx = ([self.eos] + refs[b])[t] if oracle else beam[i_beam]['hyp'][-1]

                    # Recurrency (1st) for the main model
                    y = eouts.new_zeros(1, 1).fill_(prev_idx).long()
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
                                        eouts[b:b + 1, :elens[b]],
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

                    lmout, lmstate = None, None
                    if self.rnnlm is not None:
                        # Update RNNLM states for cold fusion
                        y_lm = eouts.new_zeros(1, 1).fill_(prev_idx).long()
                        lmout, lmstate = self.rnnlm.decode(
                            self.rnnlm.encode(y_lm),
                            (beam[i_beam]['lm_hxs'], beam[i_beam]['lm_cxs']))
                    elif lm_weight > 0:
                        # Update RNNLM states for shallow fusion
                        y_lm = eouts.new_zeros(1, 1).fill_(prev_idx).long()
                        lmout, lmstate = rnnlm.decode(
                            rnnlm.encode(y_lm),
                            (beam[i_beam]['lm_hxs'], beam[i_beam]['lm_cxs']))

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
                    memory = None
                    if self.cache_fusion:
                        if self.fifo_cache_lm_key is None:
                            memory = eouts.new_zeros(1, 1, self.rnnlm.n_units)
                        else:
                            memory = self.fifo_cache_lm_key
                    attn_v, aw_cache, lm_feat = self.generate(cv, dstates['dout_gen'], lmout, cache_keys=memory)
                    probs = F.softmax(self.output(attn_v).squeeze(1), dim=1)

                    # Generate for RNNLM
                    if lm_weight > 0:
                        if self.rnnlm is not None:
                            lm_probs = F.softmax(self.rnnlm.generate(lmout).squeeze(1), dim=-1)
                        else:
                            lm_probs = F.softmax(rnnlm.generate(lmout).squeeze(1), dim=-1)

                    # Cache decoding
                    is_cache = len(self.fifo_cache_ids + beam[i_beam]['cache_idx']) > 0
                    cache_idx = None
                    cache_sp_attn = None
                    cache_lm_attn = None
                    cache_probs_sp = torch.zeros_like(probs)
                    cache_probs_lm = torch.zeros_like(probs)
                    if n_caches > 0 and is_cache:
                        # Compute inner-product over caches
                        cache_idx = (self.fifo_cache_ids + beam[i_beam]['cache_idx'])[-n_caches:]
                        if cache_type in ['speech', 'joint']:
                            cache_sp_key = beam[i_beam]['cache_sp_key'][-n_caches:]
                            if len(cache_sp_key) > 0:
                                cache_sp_key = torch.cat(cache_sp_key, dim=1)
                                if self.fifo_cache_sp_key is not None:
                                    cache_sp_key = torch.cat([self.fifo_cache_sp_key, cache_sp_key], dim=1)
                            else:
                                # For the first token
                                cache_sp_key = self.fifo_cache_sp_key
                            # Truncate
                            cache_sp_key = cache_sp_key[:, -n_caches:]  # `[1, L, enc_n_units]`
                            cache_sp_attn = F.softmax(cache_theta_sp * torch.matmul(
                                cache_sp_key, cv.transpose(2, 1)), dim=1)  # `[1, L, 1]`
                            # cache_sp_attn = F.softmax(cache_theta_sp * torch.matmul(
                            #     cache_sp_key, dstates['dout_gen'].transpose(2, 1)), dim=1)  # `[1, L, 1]`
                            # Sum all probabilities
                            for c in set(beam[i_beam]['cache_idx']):
                                for offset in [i for i, key in enumerate(cache_idx) if key == c]:
                                    cache_probs_sp[0, c] += cache_sp_attn[0, offset, 0]
                            probs = (1 - cache_lambda_sp) * probs + cache_lambda_sp * cache_probs_sp

                        if cache_type in ['lm', 'joint'] and lm_weight > 0:
                            cache_lm_key = beam[i_beam]['cache_lm_key'][-n_caches:]
                            if len(cache_lm_key) > 0:
                                cache_lm_key = torch.cat(cache_lm_key, dim=1)
                                if self.fifo_cache_lm_key is not None:
                                    cache_lm_key = torch.cat([self.fifo_cache_lm_key, cache_lm_key], dim=1)
                            else:
                                # For the first token
                                cache_lm_key = self.fifo_cache_lm_key
                            # Truncate
                            cache_lm_key = cache_lm_key[:, -n_caches:]  # `[1, L, lm_n_units]`
                            cache_lm_attn = F.softmax(cache_theta_lm * torch.matmul(
                                cache_lm_key, lmout.transpose(2, 1)), dim=1)  # `[1, L, 1]`
                            # Sum all probabilities
                            for c in set(beam[i_beam]['cache_idx']):
                                for offset in [i for i, key in enumerate(cache_idx) if key == c]:
                                    cache_probs_lm[0, c] += cache_lm_attn[0, offset, 0]
                            lm_probs = (1 - cache_lambda_lm) * lm_probs + cache_lambda_lm * cache_probs_lm

                        if self.cache_fusion:
                            cache_lm_attn = aw_cache

                    log_probs = torch.log(probs)
                    # Generate for the ensemble
                    if n_models > 0:
                        for i_e, dec in enumerate(ensemble_decoders):
                            attn_v, _ = dec.generate(ensemble_cv[i_e],
                                                     ensemble_dstates[i_e]['dout_gen'],
                                                     lmout)
                            log_probs += F.log_softmax(dec.output(attn_v).squeeze(1), dim=1)
                        # re-normalize
                        log_probs /= (n_models + 1)
                        # TODO(hirofumi): cache

                    # Pick up the top-k scores
                    log_probs_topk, ids_topk = torch.topk(log_probs, k=beam_width, dim=1, largest=True, sorted=True)
                    scores_attn = beam[i_beam]['score_attn'] + log_probs_topk
                    local_scores = scores_attn.clone()

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
                                cp = aw_mat.sum() / self.score.n_heads
                            else:
                                cp = torch.where(aw_mat > cp_threshold, aw_mat,
                                                 torch.zeros_like(aw_mat)).sum() / self.score.n_heads
                            local_scores += cp * cp_weight
                    else:
                        cp = torch.zeros((), dtype=torch.float32)
                        aw_mat = None

                    local_scores *= (1 - ctc_weight)

                    # Add RNNLM score
                    if lm_weight > 0:
                        lm_log_probs = torch.log(lm_probs)
                        scores_lm = beam[i_beam]['score_lm'] + lm_log_probs[0, ids_topk[0]]
                        score_lm_norm = scores_lm / lp  # normalize by length
                        local_scores += score_lm_norm * lm_weight
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
                        if top_idx == self.eos and len(beam[i_beam]['hyp']) - 1 < elens[b] * params['recog_min_len_ratio']:
                            continue

                        score_attn = scores_attn[0, k].item()
                        score_ctc = scores_ctc[k].item()
                        score_lm = scores_lm[k].item()
                        score_t = score_attn * (1 - ctc_weight) + score_ctc * ctc_weight + score_lm * lm_weight

                        new_beam.append(
                            {'hyp': beam[i_beam]['hyp'] + [top_idx],
                             'score': local_scores[0, k].item(),
                             #  'scores': beam[i_beam]['scores'] + [score_t],
                             'scores': beam[i_beam]['scores'] + [local_scores[0, k].item()],
                             'scores_cp': beam[i_beam]['scores_cp'] + [cp * cp_weight],
                             'score_attn': score_attn,  # total score
                             'score_cp': cp,
                             'score_ctc': score_ctc,  # total score
                             'score_lm': score_lm,  # total score
                             'dstates': dstates,
                             'cv': attn_v if self.input_feeding else cv,
                             'aws': beam[i_beam]['aws'] + [aw],
                             'lm_hxs': lmstate[0][:] if lmstate is not None else None,
                             'lm_cxs': lmstate[1][:] if lmstate is not None else None,
                             'cp_prev': cp,
                             'ensemble_dstates': ensemble_dstates,
                             'ensemble_cv': ensemble_cv,
                             'ensemble_aws': ensemble_aws,
                             'ctc_state': ctc_states[joint_ids_topk[0, k]] if ctc_log_probs is not None else None,
                             'ctc_score': ctc_scores[joint_ids_topk[0, k]] if ctc_log_probs is not None else None,
                             'cache_idx': beam[i_beam]['cache_idx'] + [top_idx],
                             'cache_sp_key': beam[i_beam]['cache_sp_key'] + [cv],
                             #  'cache_sp_key': beam[i_beam]['cache_sp_key'] + [dstates['dout_gen']],
                             'cache_lm_key': beam[i_beam]['cache_lm_key'] + [lmout],
                             'cache_id_hist': beam[i_beam]['cache_id_hist'] + [cache_idx] if is_cache else beam[i_beam]['cache_id_hist'],
                             'cache_sp_attn_hist': beam[i_beam]['cache_sp_attn_hist'] + [cache_sp_attn] if is_cache else beam[i_beam]['cache_sp_attn_hist'],
                             'cache_lm_attn_hist': beam[i_beam]['cache_lm_attn_hist'] + [cache_lm_attn] if is_cache else beam[i_beam]['cache_lm_attn_hist'],
                             })

                new_beam = sorted(new_beam, key=lambda x: x['score'], reverse=True)

                # Remove complete hypotheses
                not_complete = []
                for cand in new_beam[:beam_width]:
                    if oracle:
                        if t == len(refs[b]):
                            complete += [cand]
                        else:
                            not_complete += [cand]
                    else:
                        if cand['hyp'][-1] == self.eos and cand['hyp'].count(self.eos) >= concat_prev_n_utterances + 2:
                            complete += [cand]
                        else:
                            not_complete += [cand]

                # Pruning
                if len(complete) >= beam_width:
                    complete = complete[:beam_width]
                    break
                beam = not_complete[:beam_width]

            # Pruning
            if len(complete) == 0:
                complete = beam[:]
            elif len(complete) < nbest and nbest > 1:
                complete.extend(beam[:nbest - len(complete)])

            # backward LM rescoring
            if rnnlm_rev is not None and lm_weight > 0:
                for cand in complete:
                    # Initialize
                    lm_rev_hxs, lm_rev_cxs = None, None
                    score_lm_rev = 0
                    lp = 1

                    # Append <eos>
                    if cand['hyp'][-1] != self.eos:
                        cand['hyp'].append(self.eos)

                    if lp_weight > 0 and gnmt_decoding:
                        lp = (math.pow(5 + (len(cand['hyp']) - 1 + 1), lp_weight)) / math.pow(6, lp_weight)
                    for t in range(len(cand['hyp'][1:])):
                        y_lm = eouts.new_zeros(1, 1).fill_(cand['hyp'][-1 - t]).long()
                        lm_out_rev, (lm_rev_hxs, lm_rev_cxs) = rnnlm_rev.decode(
                            rnnlm_rev.encode(y_lm), (lm_hxs, lm_rev_cxs))
                        lm_log_probs = F.log_softmax(rnnlm_rev.generate(lm_out_rev).squeeze(1), dim=-1)
                        score_lm_rev += lm_log_probs[0, cand['hyp'][-2 - t]]
                    score_lm_rev /= lp  # normalize
                    cand['score'] += score_lm_rev * lm_weight
                    cand['score_lm_rev'] = score_lm_rev * lm_weight

            # Sort by score
            complete = sorted(complete, key=lambda x: x['score'], reverse=True)

            # N-best list
            if self.backward:
                # Reverse the order
                nbest_hyps += [[np.array(complete[n]['hyp'][1:][::-1]) for n in range(nbest)]]
                if self.score.n_heads > 1:
                    aws += [[complete[n]['aws'][0, 1:][::-1] for n in range(nbest)]]
                else:
                    aws += [[complete[n]['aws'][1:][::-1] for n in range(nbest)]]
                scores += [[complete[n]['scores'][1:][::-1] for n in range(nbest)]]
                scores_cp += [[complete[n]['scores_cp'][1:][::-1] for n in range(nbest)]]
            else:
                nbest_hyps += [[np.array(complete[n]['hyp'][1:]) for n in range(nbest)]]
                if self.score.n_heads > 1:
                    aws += [[complete[n]['aws'][0, 1:] for n in range(nbest)]]
                else:
                    aws += [[complete[n]['aws'][1:] for n in range(nbest)]]
                scores += [[complete[n]['scores'][1:] for n in range(nbest)]]
                scores_cp += [[complete[n]['scores_cp'][1:] for n in range(nbest)]]

            # Check <eos>
            eos_flag = [True if complete[n]['hyp'][-1] == self.eos else False for n in range(nbest)]
            eos_flags.append(eos_flag)

            if refs is not None and idx2token is not None:
                logger.info('Ref: %s' % idx2token(refs[b]))
            for n in range(nbest):
                if idx2token is not None:
                    logger.info('Hyp: %s' % idx2token(nbest_hyps[0][n]))
                logger.info('log prob (hyp): %.7f' % complete[n]['score'])
                logger.info('log prob (hyp, att): %.7f' % (complete[n]['score_attn'] * (1 - ctc_weight)))
                logger.info('log prob (hyp, cp): %.7f' % (complete[n]['score_cp'] * cp_weight))
                if ctc_weight > 0 and ctc_log_probs is not None:
                    logger.info('log prob (hyp, ctc): %.7f' % (complete[n]['score_ctc'] * ctc_weight))
                if lm_weight > 0:
                    logger.info('log prob (hyp, lm): %.7f' % (complete[n]['score_lm'] * lm_weight))
                    if rnnlm_rev is not None:
                        logger.info('log prob (hyp, lm rev): %.7f' % (complete[n]['score_lm_rev']))
                if n_caches > 0:
                    logger.info('Cache: %d' % (len(self.fifo_cache_ids) + len(complete[0]['cache_idx'])))

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

        # Store ASR/RNNLM state
        self.dstates_final = complete[0]['dstates']
        self.lmstate_final = (complete[0]['lm_hxs'], complete[0]['lm_cxs'])

        # Store in cache
        cache_id_hist = None
        cache_sp_attn_hist = None
        cache_lm_attn_hist = None
        is_local_cache = len(complete[0]['cache_idx']) > 0
        if n_caches > 0 and is_local_cache:
            hyp_len = len(complete[0]['hyp'][1:])
            self.fifo_cache_ids = (self.fifo_cache_ids + complete[0]['cache_idx'])[:n_caches]

            cache_id_hist = complete[0]['cache_id_hist']
            if cache_type in ['speech', 'joint']:
                cache_sp_key = complete[0]['cache_sp_key'][-n_caches:]
                cache_sp_key = torch.cat(cache_sp_key, dim=1)
                if self.fifo_cache_sp_key is not None:
                    cache_sp_key = torch.cat([self.fifo_cache_sp_key, cache_sp_key], dim=1)
                # Truncate
                self.fifo_cache_sp_key = cache_sp_key[:, -n_caches:]  # `[1, L, enc_n_units]`
                cache_sp_attn_hist = torch.zeros(
                    (1, complete[0]['cache_sp_attn_hist'][-1].size(1), hyp_len), dtype=torch.float32)
                for i, p in enumerate(complete[0]['cache_sp_attn_hist']):
                    if p.size(1) < n_caches:
                        cache_sp_attn_hist[0, :p.size(1), i] = p[0, :, 0].cpu()
                    else:
                        cache_sp_attn_hist[0, :n_caches - (hyp_len - 1 - i), i] = p[0, (hyp_len - 1 - i):, 0].cpu()

            if cache_type in ['lm', 'joint']:
                assert lm_weight > 0 or self.cache_fusion
                cache_lm_key = complete[0]['cache_lm_key'][-n_caches:]
                cache_lm_key = torch.cat(cache_lm_key, dim=1)
                if self.fifo_cache_lm_key is not None:
                    cache_lm_key = torch.cat([self.fifo_cache_lm_key, cache_lm_key], dim=1)
                # Truncate
                self.fifo_cache_lm_key = cache_lm_key[:, -n_caches:]  # `[1, L, lm_n_units]`
                cache_lm_attn_hist = torch.zeros((1, complete[0]['cache_lm_attn_hist'][-1].size(1), hyp_len),
                                                 dtype=torch.float32)  # `[B, n_keys, n_values]`
                for i, p in enumerate(complete[0]['cache_lm_attn_hist']):
                    if p.size(1) < n_caches:
                        cache_lm_attn_hist[0, :p.size(1), i] = p[0, :, 0].cpu()
                    else:
                        cache_lm_attn_hist[0, :n_caches - (hyp_len - 1 - i), i] = p[0, (hyp_len - 1 - i):, 0].cpu()

        if cache_type in ['speech', 'joint']:
            return nbest_hyps, aws, scores, scores_cp, (cache_sp_attn_hist, cache_id_hist)
        else:
            return nbest_hyps, aws, scores, scores_cp, (cache_lm_attn_hist, cache_id_hist)

    def reset_global_cache(self):
        """Reset global cache when the speaker/session is changed."""
        self.fifo_cache_ids = []
        self.fifo_cache_sp_key = None
        self.fifo_cache_lm_key = None

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
