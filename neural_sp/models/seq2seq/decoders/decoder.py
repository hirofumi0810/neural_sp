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
from neural_sp.models.linear import Embedding
from neural_sp.models.linear import LinearND
from neural_sp.models.seq2seq.decoders.attention import AttentionMechanism
from neural_sp.models.seq2seq.decoders.ctc_beam_search_decoder import BeamSearchDecoder
from neural_sp.models.seq2seq.decoders.ctc_greedy_decoder import GreedyDecoder
from neural_sp.models.seq2seq.decoders.multihead_attention import MultiheadAttentionMechanism
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list
from neural_sp.models.torch_utils import tensor2np

random.seed(1)

logger = logging.getLogger("decoding")


class Decoder(nn.Module):
    """RNN decoder.

    Args:
        enc_nunits (int):
        sos (int): index for <sos>
        eos (int): index for <eos>
        pad (int): index for <pad>
        enc_nunits (int):
        attn_type ():
        attn_dim ():
        attn_sharpening_factor ():
        attn_sigmoid_smoothing ():
        attn_conv_out_channels ():
        attn_conv_kernel_size ():
        attn_nheads ():
        dropout_att ():
        rnn_type (str): lstm or gru
        nunits (int): the number of units in each RNN layer
        nlayers (int): the number of RNN layers
        residual (bool):
        emb_dim (int): the dimension of the embedding in target spaces.
        tie_embedding (bool):
        vocab (int): the number of nodes in softmax layer
        logits_temp (float): a parameter for smoothing the softmax layer in outputing probabilities
        dropout (float): the probability to drop nodes in the RNN layer
        dropout_emb (float): the probability to drop nodes of the embedding layer
        ss_prob (float): scheduled sampling probability
        lsm_prob (float): label smoothing probability
        layer_norm (bool): layer normalization
        init_with_enc (bool):
        ctc_weight (float):
        ctc_fc_list (list):
        input_feeding (bool):
        backward (bool): decode in the backward order
        rnnlm_cold_fusion (torch.nn.Module):
        cold_fusion (str): the type of cold fusion
            prob: probability from RNNLM
            hidden: hidden states of RNNLM
        internal_lm (bool):
        rnnlm_init ():
        lmobj_weight (float):
        share_lm_softmax (bool):

    """

    def __init__(self,
                 sos,
                 eos,
                 pad,
                 enc_nunits,
                 attn_type,
                 attn_dim,
                 attn_sharpening_factor,
                 attn_sigmoid_smoothing,
                 attn_conv_out_channels,
                 attn_conv_kernel_size,
                 attn_nheads,
                 dropout_att,
                 rnn_type,
                 nunits,
                 nlayers,
                 residual,
                 emb_dim,
                 tie_embedding,
                 vocab,
                 logits_temp,
                 dropout,
                 dropout_emb,
                 ss_prob,
                 lsm_prob,
                 layer_norm,
                 fl_weight,
                 fl_gamma,
                 init_with_enc=False,
                 ctc_weight=0.,
                 ctc_fc_list=[],
                 input_feeding=False,
                 backward=False,
                 rnnlm_cold_fusion=False,
                 cold_fusion='hidden',
                 internal_lm=False,
                 rnnlm_init=False,
                 lmobj_weight=0.,
                 share_lm_softmax=False,
                 global_weight=1.0,
                 mtl_per_batch=False,
                 vocab_char=None):

        super(Decoder, self).__init__()

        self.sos = sos
        self.eos = eos
        self.pad = pad
        self.rnn_type = rnn_type
        assert rnn_type in ['lstm', 'gru']
        self.enc_nunits = enc_nunits
        self.nunits = nunits
        self.nlayers = nlayers
        self.residual = residual
        self.logits_temp = logits_temp
        self.dropout = dropout
        self.dropout_emb = dropout_emb
        self.ss_prob = ss_prob
        self.lsm_prob = lsm_prob
        self.layer_norm = layer_norm
        self.fl_weight = fl_weight
        self.fl_gamma = fl_gamma
        self.init_with_enc = init_with_enc
        self.ctc_weight = ctc_weight
        self.ctc_fc_list = ctc_fc_list
        self.backward = backward
        self.rnnlm_cf = rnnlm_cold_fusion
        self.cold_fusion = cold_fusion
        self.internal_lm = internal_lm
        self.rnnlm_init = rnnlm_init
        self.lmobj_weight = lmobj_weight
        self.share_lm_softmax = share_lm_softmax
        self.global_weight = global_weight
        self.mtl_per_batch = mtl_per_batch

        if ctc_weight > 0 and not backward:
            # Fully-connected layers for CTC
            if len(ctc_fc_list) > 0:
                fc_layers = OrderedDict()
                for i in range(len(ctc_fc_list)):
                    input_dim = enc_nunits if i == 0 else ctc_fc_list[i - 1]
                    fc_layers['fc' + str(i)] = LinearND(input_dim, ctc_fc_list[i], dropout=dropout)
                fc_layers['fc' + str(len(ctc_fc_list))] = LinearND(ctc_fc_list[-1], vocab, dropout=0)
                self.output_ctc = nn.Sequential(fc_layers)
            else:
                self.output_ctc = LinearND(enc_nunits, vocab)
            self.decode_ctc_greedy = GreedyDecoder(blank_index=0)
            self.decode_ctc_beam = BeamSearchDecoder(blank_index=0)
            self.warpctc_loss = warpctc_pytorch.CTCLoss(size_average=True)

        if ctc_weight < global_weight:
            # Attention layer
            if attn_nheads > 1:
                self.score = MultiheadAttentionMechanism(
                    enc_nunits=self.enc_nunits,
                    dec_nunits=nunits,
                    attn_type=attn_type,
                    attn_dim=attn_dim,
                    sharpening_factor=attn_sharpening_factor,
                    sigmoid_smoothing=attn_sigmoid_smoothing,
                    conv_out_channels=attn_conv_out_channels,
                    conv_kernel_size=attn_conv_kernel_size,
                    nheads=attn_nheads,
                    dropout=dropout_att)
            else:
                self.score = AttentionMechanism(
                    enc_nunits=self.enc_nunits,
                    dec_nunits=nunits,
                    attn_type=attn_type,
                    attn_dim=attn_dim,
                    sharpening_factor=attn_sharpening_factor,
                    sigmoid_smoothing=attn_sigmoid_smoothing,
                    conv_out_channels=attn_conv_out_channels,
                    conv_kernel_size=attn_conv_kernel_size,
                    dropout=dropout_att)

            # for decoder initialization with pre-trained RNNLM
            if rnnlm_init:
                assert internal_lm
                assert rnnlm_init.predictor.vocab == vocab
                assert rnnlm_init.predictor.nunits == nunits
                assert rnnlm_init.predictor.nlayers == 1  # TODO(hirofumi): on-the-fly

            # for MTL with RNNLM objective
            if lmobj_weight > 0:
                if internal_lm and not share_lm_softmax:
                    self.output_lmobj = LinearND(nunits, vocab)

            # Decoder
            self.rnn = torch.nn.ModuleList()
            self.dropout = torch.nn.ModuleList()
            if rnn_type == 'lstm':
                rnn_cell = nn.LSTMCell
            elif rnn_type == 'gru':
                rnn_cell = nn.GRUCell
            if internal_lm:
                assert nlayers >= 2
                self.rnn_inlm = rnn_cell(emb_dim, nunits)
                self.dropout_inlm = nn.Dropout(p=dropout)
                self.rnn += [rnn_cell(nunits + enc_nunits, nunits)]
            else:
                self.rnn += [rnn_cell(emb_dim + enc_nunits, nunits)]
            self.dropout += [nn.Dropout(p=dropout)]
            for l in range(1, nlayers):
                self.rnn += [rnn_cell(nunits, nunits)]
                self.dropout += [nn.Dropout(p=dropout)]

            # cold fusion
            if rnnlm_cold_fusion:
                self.cf_linear_dec_feat = LinearND(nunits + enc_nunits, nunits)
                if cold_fusion == 'hidden':
                    self.cf_linear_lm_feat = LinearND(rnnlm_cold_fusion.nunits, nunits)
                elif cold_fusion == 'prob':
                    self.cf_linear_lm_feat = LinearND(rnnlm_cold_fusion.vocab, nunits)
                else:
                    raise ValueError(cold_fusion)
                self.cf_linear_lm_gate = LinearND(nunits * 2, nunits)
                self.output_bn = LinearND(nunits * 2, nunits)

                # fix RNNLM parameters
                for p in self.rnnlm_cf.parameters():
                    p.requires_grad = False
            else:
                self.output_bn = LinearND(nunits + enc_nunits, nunits)

            self.output = LinearND(nunits, vocab)

            # Embedding
            self.embed = Embedding(vocab=vocab,
                                   emb_dim=emb_dim,
                                   dropout=dropout_emb,
                                   ignore_index=pad)

            # Optionally tie weights as in:
            # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
            # https://arxiv.org/abs/1608.05859
            # and
            # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
            # https://arxiv.org/abs/1611.01462
            if tie_embedding:
                if nunits != emb_dim:
                    raise ValueError('When using the tied flag, nunits must be equal to emb_dim.')
                self.output.fc.weight = self.embed.embed.weight

    def forward(self, eouts, elens, ys, device_id, task='all'):
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
        obserbation = {'loss': None,
                       'loss_att': None, 'loss_ctc': None, 'loss_lmobj': None,
                       'acc_att': None, 'acc_lmobj': None,
                       'ppl_att': None, 'ppl_lmobj': None}
        loss = torch.zeros((1,), dtype=torch.float32).cuda()

        # CTC loss
        if self.ctc_weight > 0 and (not self.mtl_per_batch or (self.mtl_per_batch and 'ctc' in task)):
            loss_ctc = self.forward_ctc(eouts, elens, ys, device_id)
            obserbation['loss_ctc'] = loss_ctc.item()
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight

        # LM objective
        if self.lmobj_weight > 0 and 'lmobj' in task:
            loss_lmobj, acc_lmobj, ppl_lmobj = self.forward_lmobj(ys, device_id)
            obserbation['loss_lmobj'] = loss_lmobj.item()
            obserbation['acc_lmobj'] = acc_lmobj
            obserbation['ppl_lmobj'] = ppl_lmobj
            if self.mtl_per_batch:
                loss += loss_lmobj
            else:
                loss += loss_lmobj * self.lmobj_weight

        # XE loss
        if self.global_weight - self.ctc_weight > 0 and 'ctc' not in task and 'lmobj' not in task:
            loss_att, acc_att, ppl_att = self.forward_att(eouts, elens, ys, device_id)
            obserbation['loss_att'] = loss_att.item()
            obserbation['acc_att'] = acc_att
            obserbation['ppl_att'] = ppl_att
            if self.mtl_per_batch:
                loss += loss_att
            else:
                loss += loss_att * (self.global_weight - self.ctc_weight)

        obserbation['loss'] = loss.item()
        return loss, obserbation

    def forward_ctc(self, eouts, elens, ys, device_id):
        """Compute CTC loss.

        Args:
            eouts (FloatTensor): `[B, T, dec_units]`
            elens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            device_id (int):
        Returns:
            loss (FloatTensor): `[B, L, vocab]`

        """
        logits = self.output_ctc(eouts).transpose(0, 1).cpu()  # time-major

        # Compute the auxiliary CTC loss
        assert not self.backward
        enc_lens_ctc = np2tensor(np.fromiter(elens, dtype=np.int32), -1).int()
        ys_ctc = [np2tensor(np.fromiter(y, dtype=np.int64)).long() for y in ys]  # always fwd
        y_lens = np2tensor(np.fromiter([y.size(0) for y in ys_ctc], dtype=np.int32), -1).int()
        ys_ctc = torch.cat(ys_ctc, dim=0).int()
        # NOTE: Concatenate all elements in ys for warpctc_pytorch
        # NOTE: do not copy to GPUs here

        # Compute CTC loss
        loss = self.warpctc_loss(logits, ys_ctc, enc_lens_ctc, y_lens)
        # NOTE: ctc loss has already been normalized by bs
        # NOTE: index 0 is reserved for blank in warpctc_pytorch

        if device_id >= 0:
            loss = loss.cuda(device_id)

        # Label smoothing for CTC
        if self.lsm_prob > 0 and self.ctc_weight == 1:
            loss = loss * (1 - self.lsm_prob) + kldiv_lsm_ctc(
                logits, y_lens=elens,
                lsm_prob=self.lsm_prob, size_average=True) * self.lsm_prob

        return loss

    def forward_lmobj(self, ys, device_id):
        """Compute XE loss for LM objective.

        Args:
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            device_id (int):
        Returns:
            loss (FloatTensor): `[1]`
            acc (float):
            ppl (float):

        """
        bs = len(ys)

        # Append <sos> and <eos>
        sos = torch.zeros((1,)).fill_(self.sos).long().cuda(device_id)
        eos = torch.zeros((1,)).fill_(self.eos).long().cuda(device_id)
        if self.backward:
            ys = [np2tensor(np.fromiter(y[::-1], dtype=np.int64), device_id).long() for y in ys]
            ys_in = [torch.cat([eos, y], dim=0) for y in ys]
            ys_out = [torch.cat([y, sos], dim=0) for y in ys]
        else:
            ys = [np2tensor(np.fromiter(y, dtype=np.int64), device_id).long() for y in ys]
            ys_in = [torch.cat([sos, y], dim=0) for y in ys]
            ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        ys_in_pad = pad_list(ys_in, self.pad)
        ys_out_pad = pad_list(ys_out, -1)

        # Initialization
        dout, dstate = self.init_dec_state(bs, self.nlayers, device_id)
        _dout, _dstate = self.init_dec_state(bs, 1, device_id)  # for internal LM
        context = torch.zeros((bs, 1, self.enc_nunits), dtype=torch.float32).cuda(device_id)

        # Pre-computation of embedding
        ys_emb = self.embed(ys_in_pad)

        logits = []
        for t in range(ys_in_pad.size(1)):
            y_emb = ys_emb[:, t:t + 1]

            # Recurrency
            dout, dstate, _dout, _dstate = self.recurrency(y_emb, context, dstate, _dstate)

            # Generate
            if self.internal_lm:
                if self.share_lm_softmax:
                    logits_t = self.output(_dout)
                else:
                    logits_t = self.output_lmobj(_dout)
            else:
                attentional_t = self.generate(context, dout)
                logits_t = self.output(attentional_t)
            logits.append(logits_t)

        # Compute XE loss for RNNLM objective
        logits = torch.cat(logits, dim=1) / self.logits_temp
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

    def forward_att(self, eouts, elens, ys, device_id):
        """Compute XE loss for the sequence-to-sequence model.

        Args:
            eouts (FloatTensor): `[B, T, dec_units]`
            elens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            device_id (int):
        Returns:
            loss (FloatTensor): `[B, L, vocab]`
            acc (float):
            ppl (float):

        """
        bs, _, enc_nunits = eouts.size()

        # Append <sos> and <eos>
        sos = eouts.new_zeros(1).fill_(self.sos).long()
        eos = eouts.new_zeros(1).fill_(self.eos).long()
        if self.backward:
            ys = [np2tensor(np.fromiter(y[::-1], dtype=np.int64), device_id).long() for y in ys]
            ys_in = [torch.cat([eos, y], dim=0) for y in ys]
            ys_out = [torch.cat([y, sos], dim=0) for y in ys]
        else:
            ys = [np2tensor(np.fromiter(y, dtype=np.int64), device_id).long() for y in ys]
            ys_in = [torch.cat([sos, y], dim=0) for y in ys]
            ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        ys_in_pad = pad_list(ys_in, self.pad)
        ys_out_pad = pad_list(ys_out, -1)

        # Initialization
        dout, dstate = self.init_dec_state(bs, self.nlayers, device_id, eouts, elens)
        _dout, _dstate = self.init_dec_state(bs, 1, device_id, eouts, elens)  # for internal LM
        context = eouts.new_zeros(bs, 1, enc_nunits)
        self.score.reset()
        aw = None
        rnnlm_state = None

        # Pre-computation of embedding
        ys_emb = self.embed(ys_in_pad)
        if self.rnnlm_cf:
            ys_lm_emb = self.rnnlm_cf.embed(ys_in_pad)
            # ys_lm_emb = [self.rnnlm_cf.embed(ys_in_pad[:, t:t + 1])
            #              for t in range(ys_in_pad.size(1))]
            # ys_lm_emb = torch.cat(ys_lm_emb, dim=1)

        logits = []
        for t in range(ys_in_pad.size(1)):
            # Sample for scheduled sampling
            is_sample = t > 0 and self.ss_prob > 0 and random.random() < self.ss_prob
            if is_sample:
                y_emb = self.embed(torch.argmax(logits[-1].detach(), dim=-1))
            else:
                y_emb = ys_emb[:, t:t + 1]

            # Recurrency
            dout, dstate, _dout, _dstate = self.recurrency(y_emb, context, dstate, _dstate)

            # Update RNNLM states for cold fusion
            if self.rnnlm_cf:
                if is_sample:
                    y_lm_emb = self.rnnlm_cf.embed(np.argmax(logits[-1].detach(), axis=2).cuda(device_id))
                else:
                    y_lm_emb = ys_lm_emb[:, t:t + 1]
                logits_lm_t, lm_out, rnnlm_state = self.rnnlm_cf.predict(y_lm_emb, rnnlm_state)
            else:
                logits_lm_t, lm_out = None, None

            # Score
            context, aw = self.score(eouts, elens, dout, aw)

            # Generate
            attentional_t = self.generate(context, dout, logits_lm_t, lm_out)
            if self.rnnlm_init and self.internal_lm:
                # Residual connection
                attentional_t += _dout
            logits.append(self.output(attentional_t))

        # Compute XE sequence loss
        logits = torch.cat(logits, dim=1) / self.logits_temp
        if self.lsm_prob > 0:
            # Label smoothing
            y_lens = [y.size(0) for y in ys_out]
            loss = cross_entropy_lsm(
                logits, ys=ys_out_pad, y_lens=y_lens,
                lsm_prob=self.lsm_prob, size_average=True)
        else:
            loss = F.cross_entropy(
                logits.view((-1, logits.size(2))),
                ys_out_pad.view(-1),  # long
                ignore_index=-1, size_average=False) / bs
        ppl = math.exp(loss.item())

        # Focal loss
        if self.fl_weight > 0:
            y_lens = [y.size(0) for y in ys_out]
            fl = focal_loss(logits, ys=ys_out_pad, y_lens=y_lens,
                            gamma=self.fl_gamma, size_average=True)
            loss = loss * (1 - self.fl_weight) + fl * self.fl_weight

        # Compute token-level accuracy in teacher-forcing
        pad_pred = logits.view(ys_out_pad.size(0), ys_out_pad.size(1), logits.size(-1)).argmax(2)
        mask = ys_out_pad != -1
        numerator = torch.sum(pad_pred.masked_select(mask) == ys_out_pad.masked_select(mask))
        denominator = torch.sum(mask)
        acc = float(numerator) * 100 / float(denominator)

        return loss, acc, ppl

    def init_dec_state(self, batch_size, nlayers, device_id, eouts=None, elens=None):
        """Initialize decoder state.

        Args:
            eouts (FloatTensor): `[B, T, dec_units]`
            elens (list): A list of length `[B]`
            nlayers (int):
        Returns:
            dout (FloatTensor): `[B, 1, dec_units]`
            dstate (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of FloatTensor):
                cx_list (list of FloatTensor):

        """
        if self.init_with_enc:
            if eouts.size(-1) == self.nunits:
                # unidirectinal encoder
                dout = torch.cat([eouts[b:b + 1, elens[b] - 1:elens[b]]
                                  for b in range(len(elens))], dim=0)
            else:
                raise NotImplementedError()
                # TODO(hirofumi): add bridge layer
                # bidirectional encoder
                dout = torch.cat([eouts[b:b + 1, 0:1, self.nunits:]
                                  for b in range(len(elens))], dim=0)
                # NOTE: initialize with reverse direction
            dout = torch.tanh(dout)
            hx_list = [dout.clone().squeeze(1)] * self.nlayers
            cx_list = [dout.clone().squeeze(1)] * self.nlayers if self.rnn_type == 'lstm' else None
        else:
            dout = torch.zeros((batch_size, 1, self.nunits), dtype=torch.float32).cuda(device_id)
            zero_state = torch.zeros((batch_size, self.nunits), dtype=torch.float32).cuda(device_id)
            hx_list = [zero_state] * self.nlayers
            cx_list = [zero_state] * self.nlayers if self.rnn_type == 'lstm' else None

        return dout, (hx_list, cx_list)

    def recurrency(self, y_emb, context, dstate, _dstate=None):
        """Recurrency function.

        Args:
            y_emb (FloatTensor): `[B, 1, emb_dim]`
            context (FloatTensor): `[B, 1, enc_nunits]`
            dstate (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of FloatTensor):
                cx_list (list of FloatTensor):
            _dstate (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of FloatTensor):
                cx_list (list of FloatTensor):
        Returns:
            dout (FloatTensor): `[B, 1, nunits]`
            dstate (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of FloatTensor):
                cx_list (list of FloatTensor):
            _dout (FloatTensor): `[B, 1, nunits]`
            _dstate (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of FloatTensor):
                cx_list (list of FloatTensor):

        """
        hx_list, cx_list = dstate
        y_emb = y_emb.squeeze(1)
        context = context.squeeze(1)

        if self.internal_lm:
            hx_lm, cx_lm = _dstate
            if self.rnn_type == 'lstm':
                hx_lm[0], cx_lm[0] = self.rnn_inlm(y_emb, (hx_lm[0], cx_lm[0]))
                _h_lm = torch.cat([self.dropout_inlm(hx_lm[0]), context], dim=-1)
                hx_list[0], cx_list[0] = self.rnn[0](_h_lm, (hx_list[0], cx_list[0]))
            elif self.rnn_type == 'gru':
                hx_lm = self.rnn_inlm(y_emb, hx_lm)
                _h_lm = torch.cat([self.dropout_inlm(hx_lm), context], dim=-1)
                hx_list[0] = self.rnn[0](_h_lm, hx_list[0])
            _dout = self.dropout[0](hx_lm[0]).unsqueeze(1)
            _dstate = (hx_lm, cx_lm)
        else:
            if self.rnn_type == 'lstm':
                hx_list[0], cx_list[0] = self.rnn[0](torch.cat([y_emb, context], dim=-1), (hx_list[0], cx_list[0]))
            elif self.rnn_type == 'gru':
                hx_list[0] = self.rnn[0](torch.cat([y_emb, context], dim=-1), hx_list[0])
            _dout = None

        for l in range(1, self.nlayers):
            hx_lower = self.dropout[l - 1](hx_list[l - 1])
            if self.rnn_type == 'lstm':
                hx_list[l], cx_list[l] = self.rnn[l](hx_lower, (hx_list[l], cx_list[l]))
            elif self.rnn_type == 'gru':
                hx_list[l] = self.rnn[l](hx_lower, hx_list[l])

            # Residual connection
            if self.residual:
                hx_list[l] += hx_lower

        dout = self.dropout[-1](hx_list[-1]).unsqueeze(1)
        return dout, (hx_list, cx_list), _dout, _dstate

    def generate(self, context, dout, logits_lm_t=None, lm_out=None):
        """Generate function.

        Args:
            context (FloatTensor): `[B, 1, enc_nunits]`
            dout (FloatTensor): `[B, 1, dec_units]`
            logits_lm_t (FloatTensor): `[B, 1, vocab]`
            lm_out (FloatTensor): `[B, 1, lm_nunits]`
        Returns:
            logits_t (FloatTensor): `[B, 1, vocab]`

        """
        if self.rnnlm_cf:
            # cold fusion
            if self.cold_fusion == 'hidden':
                lm_feat = self.cf_linear_lm_feat(lm_out)
            elif self.cold_fusion == 'prob':
                lm_feat = self.cf_linear_lm_feat(logits_lm_t)
            dec_feat = self.cf_linear_dec_feat(torch.cat([dout, context], dim=-1))
            gate = F.sigmoid(self.cf_linear_lm_gate(torch.cat([dec_feat, lm_feat], dim=-1)))
            gated_lm_feat = gate * lm_feat
            logits_t = self.output_bn(torch.cat([dec_feat, gated_lm_feat], dim=-1))
        else:
            logits_t = self.output_bn(torch.cat([dout, context], dim=-1))
        return torch.tanh(logits_t)

    def greedy(self, eouts, elens, max_len_ratio, exclude_eos=False):
        """Greedy decoding in the inference stage.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (list): A list of length `[B]`
            max_len_ratio (int): the maximum sequence length of tokens
            exclude_eos (bool):
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`

        """
        bs, enc_time, enc_nunits = eouts.size()
        device_id = eouts.get_device()

        # Initialization
        dout, dstate = self.init_dec_state(bs, self.nlayers, device_id, eouts, elens)
        _dout, _dstate = self.init_dec_state(bs, 1, device_id, eouts, elens)
        context = eouts.new_zeros(bs, 1, enc_nunits)
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
        y_lens = np.zeros((bs,), dtype=np.int32)
        eos_flags = [False] * bs
        for t in range(int(math.floor(enc_time * max_len_ratio)) + 1):
            # Recurrency
            y_emb = self.embed(y)
            dout, dstate, _dout, _dstate = self.recurrency(y_emb, context, dstate, _dstate)

            # Update RNNLM states for cold fusion
            if self.rnnlm_cf:
                y_lm = self.rnnlm_cf.embed(y)
                logits_lm_t, lm_out, rnnlm_state = self.rnnlm_cf.predict(y_lm, rnnlm_state)
            else:
                logits_lm_t, lm_out = None, None

            # Score
            context, aw = self.score(eouts, elens, dout, aw)

            # Generate
            attentional_t = self.generate(context, dout, logits_lm_t, lm_out)
            if self.rnnlm_init and self.internal_lm:
                # Residual connection
                attentional_t += _dout
            logits_t = self.output(attentional_t)

            # Pick up 1-best
            device_id = logits_t.get_device()
            y = np.argmax(logits_t.squeeze(1).detach(), axis=1).cuda(device_id).unsqueeze(1)
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
                    y_lens[b] += 1
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
            best_hyps = [best_hyps_tmp[b, :y_lens[b]][::-1] for b in range(bs)]
            aws = [aws_tmp[b, :y_lens[b]][::-1] for b in range(bs)]
        else:
            best_hyps = [best_hyps_tmp[b, :y_lens[b]] for b in range(bs)]
            aws = [aws_tmp[b, :y_lens[b]] for b in range(bs)]

        # Exclude <eos> (<sos> in case of the backward decoder)
        if exclude_eos:
            if self.backward:
                best_hyps = [best_hyps[b][1:] if eos_flags[b]
                             else best_hyps[b] for b in range(bs)]
            else:
                best_hyps = [best_hyps[b][:-1] if eos_flags[b]
                             else best_hyps[b] for b in range(bs)]

        return best_hyps, aws

    def beam_search(self, eouts, elens, params, rnnlm, nbest=1,
                    exclude_eos=False, id2token=None, refs=None):
        """Beam search decoding in the inference stage.

        Args:
            eouts (FloatTensor): `[B, T, dec_units]`
            elens (list): A list of length `[B]`
            params (dict):
                beam_width (int): the size of beam
                max_len_ratio (int): the maximum sequence length of tokens
                min_len_ratio (float): the minimum sequence length of tokens
                length_penalty (float): length penalty
                coverage_penalty (float): coverage penalty
                coverage_threshold (float): threshold for coverage penalty
                rnnlm_weight (float): the weight of RNNLM score
            rnnlm (torch.nn.Module):
            nbest (int):
            exclude_eos (bool):
            id2token (): converter from index to token
            refs ():
        Returns:
            nbest_hyps (list): A list of length `[B]`, which contains list of n hypotheses
            aws (list): A list of length `[B]`, which contains arrays of size `[L, T]`
            scores (list):

        """
        bs, _, enc_nunits = eouts.size()
        device_id = eouts.get_device()

        # For cold fusion
        if params['rnnlm_weight'] > 0 and not self.cold_fusion:
            assert self.rnnlm_cf
            self.rnnlm_cf.eval()

        # For shallow fusion
        if rnnlm is not None:
            rnnlm.eval()

        if self.backward:
            sos, eos = self.eos, self.sos
        else:
            sos, eos = self.sos, self.eos

        nbest_hyps, aws, scores = [], [], []
        eos_flags = []
        for b in range(bs):
            # Initialization per utterance
            dout, (hx_list, cx_list) = self.init_dec_state(1, self.nlayers, device_id, eouts[b:b + 1], elens[b:b + 1])
            _dout, _dstate = self.init_dec_state(1, 1, device_id, eouts[b:b + 1], elens[b:b + 1])
            context = eouts.new_zeros(1, 1, enc_nunits)
            self.score.reset()

            complete = []
            beam = [{'hyp': [sos],
                     'score': 0,
                     'scores': [0],
                     'score_raw': 0,
                     'dout': dout,
                     'hx_list': hx_list,
                     'cx_list': cx_list,
                     'context': context,
                     'aws': [None],
                     'rnnlm_hx_list': None,
                     'rnnlm_cx_list': None,
                     'prev_cov': 0,
                     '_dout': _dout,
                     '_dstate': _dstate}]
            for t in range(int(math.floor(elens[b] * params['max_len_ratio'])) + 1):
                new_beam = []
                for i_beam in range(len(beam)):
                    # Recurrency
                    y = eouts.new_zeros(1, 1).fill_(beam[i_beam]['hyp'][-1]).long()
                    y_emb = self.embed(y)
                    dout, (hx_list, cx_list), _dout, _dstate = self.recurrency(
                        y_emb, beam[i_beam]['context'],
                        (beam[i_beam]['hx_list'], beam[i_beam]['cx_list']),
                        beam[i_beam]['_dstate'])

                    # Score
                    context, aw = self.score(eouts[b:b + 1, :elens[b]],
                                             elens[b:b + 1],
                                             dout,
                                             beam[i_beam]['aws'][-1])

                    if self.rnnlm_cf:
                        # Update RNNLM states for cold fusion
                        y_lm = eouts.new_zeros(1, 1).fill_(beam[i_beam]['hyp'][-1]).long()
                        y_lm_emb = self.rnnlm_cf.embed(y_lm).squeeze(1)
                        logits_lm_t, lm_out, rnnlm_state = self.rnnlm_cf.predict(
                            y_lm_emb, (beam[i_beam]['rnnlm_hx_list'], beam[i_beam]['rnnlm_cx_list']))
                    elif rnnlm is not None:
                        # Update RNNLM states for shallow fusion
                        y_lm = eouts.new_zeros(1, 1).fill_(beam[i_beam]['hyp'][-1]).long()
                        y_lm_emb = rnnlm.embed(y_lm).squeeze(1)
                        logits_lm_t, lm_out, rnnlm_state = rnnlm.predict(
                            y_lm_emb, (beam[i_beam]['rnnlm_hx_list'], beam[i_beam]['rnnlm_cx_list']))
                    else:
                        logits_lm_t, lm_out, rnnlm_state = None, None, None

                    # Generate
                    attentional_t = self.generate(context, dout, logits_lm_t, lm_out)
                    if self.rnnlm_init and self.internal_lm:
                        # Residual connection
                        attentional_t += _dout
                    logits_t = self.output(attentional_t)

                    # Path through the softmax layer & convert to log-scale
                    log_probs = F.log_softmax(logits_t.squeeze(1), dim=1)  # log-prob-level
                    # log_probs = logits_t.squeeze(1)  # logits-level
                    # NOTE: `[1 (B), 1, vocab]` -> `[1 (B), vocab]`

                    # Pick up the top-k scores
                    log_probs_topk, indices_topk = torch.topk(
                        log_probs, k=params['beam_width'], dim=1, largest=True, sorted=True)

                    for k in range(params['beam_width']):
                        # Exclude short hypotheses
                        if indices_topk[0, k].item() == eos and len(beam[i_beam]['hyp']) < elens[b] * params['min_len_ratio']:
                            continue

                        # Add length penalty
                        score_raw = beam[i_beam]['score_raw'] + log_probs_topk[0, k].item()
                        score = score_raw + params['length_penalty']

                        # Add coverage penalty
                        if params['coverage_penalty'] > 0:
                            # Recompute converage penalty in each step
                            score -= beam[i_beam]['prev_cov'] * params['coverage_penalty']
                            aw_stack = torch.stack(beam[i_beam]['aws'][1:] + [aw], dim=-1)
                            cov_sum = aw_stack.detach().cpu().numpy()
                            if params['coverage_threshold'] == 0:
                                cov_sum = np.sum(cov_sum) / self.score.nheads
                            else:
                                cov_sum = np.sum(
                                    cov_sum[np.where(cov_sum > params['coverage_threshold'])[0]]) / self.score.nheads
                            score += cov_sum * params['coverage_penalty']
                        else:
                            cov_sum = 0

                        # Add RNNLM score
                        if params['rnnlm_weight'] > 0:
                            lm_log_probs = F.log_softmax(logits_lm_t.squeeze(1), dim=1)
                            assert log_probs.size() == lm_log_probs.size()
                            score += lm_log_probs[0, indices_topk[0, k].item()].item() * params['rnnlm_weight']

                        new_beam.append(
                            {'hyp': beam[i_beam]['hyp'] + [indices_topk[0, k].item()],
                             'score': score,
                             'scores': beam[i_beam]['scores'] + [score],
                             'score_raw': score_raw,
                             'score_lm': 0,  # TODO(hirofumi):
                             'score_lp': 0,  # TODO(hirofumi):
                             'score_cp': 0,  # TODO(hirofumi):
                             'hx_list': hx_list[:],
                             'cx_list': cx_list[:] if cx_list is not None else None,
                             'dout': dout,
                             'context': context,
                             'aws': beam[i_beam]['aws'] + [aw],
                             'rnnlm_hx_list': rnnlm_state[0][:] if rnnlm_state is not None else None,
                             'rnnlm_cx_list': rnnlm_state[1][:] if rnnlm_state is not None else None,
                             'prev_cov': cov_sum,
                             '_dout': _dout,
                             '_dstate': _dstate[:]})

                new_beam = sorted(new_beam, key=lambda x: x['score'], reverse=True)

                # Remove complete hypotheses
                not_complete = []
                for cand in new_beam[:params['beam_width']]:
                    if cand['hyp'][-1] == eos:
                        complete += [cand]
                    else:
                        not_complete += [cand]

                if len(complete) >= params['beam_width']:
                    complete = complete[:params['beam_width']]
                    break

                beam = not_complete[:params['beam_width']]

            # Sort by score
            if len(complete) == 0:
                complete = beam
            elif len(complete) < nbest and nbest > 1:
                complete.extend(beam[:nbest - len(complete)])
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
            else:
                nbest_hyps += [[np.array(complete[n]['hyp'][1:]) for n in range(nbest)]]
                if self.score.nheads > 1:
                    aws += [[complete[n]['aws'][0, 1:] for n in range(nbest)]]
                else:
                    aws += [[complete[n]['aws'][1:] for n in range(nbest)]]
                scores += [[complete[n]['scores'][1:] for n in range(nbest)]]
            # scores += [[complete[n]['score_raw'] for n in range(nbest)]]

            # Check <eos>
            eos_flag = [True if complete[n]['hyp'][-1] == eos else False for n in range(nbest)]
            eos_flags.append(eos_flag)

            if id2token is not None:
                if refs is not None:
                    logger.info('Ref: %s' % refs[b].lower())
                for n in range(nbest):
                    logger.info('Hyp: %s' % id2token(nbest_hyps[0][n]))
            if refs is not None:
                logger.info('log prob (ref): ')
            for n in range(nbest):
                logger.info('log prob (hyp): %.3f' % complete[n]['score'])
                logger.info('log prob (hyp, raw): %.3f' % complete[n]['score_raw'])

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

        return nbest_hyps, aws, scores

    def decode_ctc(self, eouts, x_lens, beam_width=1, rnnlm=None):
        """Decoding by the CTC layer in the inference stage.

            This is only used for Joint CTC-Attention model.
        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            beam_width (int): the size of beam
            rnnlm ():
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`

        """
        logits_ctc = self.output_ctc(eouts)
        if beam_width == 1:
            best_hyps = self.decode_ctc_greedy(tensor2np(logits_ctc), x_lens)
        else:
            best_hyps = self.decode_ctc_beam(F.log_softmax(logits_ctc, dim=-1),
                                             x_lens, beam_width, rnnlm)
            # TODO(hirofumi): decoding paramters

        return best_hyps

    def ctc_posteriors(self, eouts, x_lens, temperature, topk):
        # Path through the softmax layer
        logits_ctc = self.output_ctc(eouts)
        ctc_probs = F.softmax(logits_ctc / temperature, dim=-1)
        if topk is None:
            topk = ctc_probs.size(-1)
        _, indices_topk = torch.topk(ctc_probs.sum(1), k=topk, dim=-1, largest=True, sorted=True)
        return tensor2np(ctc_probs), tensor2np(indices_topk)
