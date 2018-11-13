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
import six
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
try:
    import warpctc_pytorch
except:
    raise ImportError('Install warpctc_pytorch.')

from neural_sp.models.criterion import cross_entropy_lsm
from neural_sp.models.linear import Embedding
from neural_sp.models.linear import LinearND
from neural_sp.models.seq2seq.attention.attention import AttentionMechanism
from neural_sp.models.seq2seq.attention.multihead_attention import MultiheadAttentionMechanism
from neural_sp.models.seq2seq.decoders.ctc_beam_search_decoder import BeamSearchDecoder
from neural_sp.models.seq2seq.decoders.ctc_greedy_decoder import GreedyDecoder
from neural_sp.models.utils import np2var
from neural_sp.models.utils import pad_list
from neural_sp.models.utils import var2np

random.seed(1)

logger = logging.getLogger("decoding").getChild("decoder")


class Decoder(nn.Module):
    """RNN decoder.

    Args:
        attention (torch.nn.Module):
        sos (int): index for <sos>
        eos (int): index for <eos>
        pad (int): index for <pad>
        enc_num_units (int):
        rnn_type (str): lstm or gru
        num_units (int): the number of units in each RNN layer
        num_layers (int): the number of RNN layers
        residual (bool):
        emb_dim (int): the dimension of the embedding in target spaces.
        num_classes (int): the number of nodes in softmax layer
        logits_temp (float): a parameter for smoothing the softmax layer in outputing probabilities
        dropout_hidden (float): the probability to drop nodes in the RNN layer
        dropout_emb (float): the probability to drop nodes of the embedding layer
        ss_prob (float): the probability of scheduled sampling
        lsm_prob (float): the probability of label smoothing
        init_with_enc (bool):
        ctc_weight (float):
        backward (bool): decode in the backward order
        rnnlm_cold_fusion (torch.nn.Module):
        cold_fusion (str): the type of cold fusion
            prob: probability from RNNLM
            hidden: hidden states of RNNLM
        internal_lm (bool):
        rnnlm_init ():
        rnnlm_task_weight (float):
        share_lm_softmax (bool):

    """

    def __init__(self,
                 attention,
                 sos,
                 eos,
                 pad,
                 enc_num_units,
                 rnn_type,
                 num_units,
                 num_layers,
                 residual,
                 emb_dim,
                 num_classes,
                 logits_temp,
                 dropout_hidden,
                 dropout_emb,
                 ss_prob,
                 lsm_prob,
                 init_with_enc=False,
                 ctc_weight=0.,
                 ctc_fc_list=[],
                 backward=False,
                 rnnlm_cold_fusion=False,
                 cold_fusion='hidden',
                 internal_lm=False,
                 rnnlm_init=False,
                 rnnlm_task_weight=0.,
                 share_lm_softmax=False):

        super(Decoder, self).__init__()

        self.score = attention
        self.sos = sos
        self.eos = eos
        self.pad = pad
        self.rnn_type = rnn_type
        assert rnn_type in ['lstm', 'gru']
        self.num_units = num_units
        self.num_layers = num_layers
        self.residual = residual
        self.logits_temp = logits_temp
        self.dropout_hidden = dropout_hidden
        self.dropout_emb = dropout_emb
        self.ss_prob = ss_prob
        self.lsm_prob = lsm_prob
        self.init_with_enc = init_with_enc
        self.ctc_weight = ctc_weight
        self.ctc_fc_list = ctc_fc_list
        self.backward = backward
        self.rnnlm_cf = rnnlm_cold_fusion
        self.cold_fusion = cold_fusion
        self.internal_lm = internal_lm
        self.rnnlm_init = rnnlm_init
        self.rnnlm_task_weight = rnnlm_task_weight
        self.share_lm_softmax = share_lm_softmax

        if ctc_weight > 0:
            # Fully-connected layers for CTC
            if len(ctc_fc_list) > 0:
                fc_layers = OrderedDict()
                ctc_fc_list += [num_classes]
                for i in six.moves.range(len(ctc_fc_list)):
                    input_dim = enc_num_units if i == 0 else ctc_fc_list[i - 1]
                    fc_layers['fc' + str(i)] = LinearND(
                        input_dim, ctc_fc_list[i],
                        dropout=dropout_hidden if i < len(ctc_fc_list) - 1 else 0)
                self.output_ctc = nn.Sequential(fc_layers)
            else:
                self.output_ctc = LinearND(enc_num_units, num_classes)
            self.decode_ctc_greedy = GreedyDecoder(blank_index=0)
            self.decode_ctc_beam = BeamSearchDecoder(blank_index=0)
            self.warpctc_loss = warpctc_pytorch.CTCLoss(size_average=True)

        if ctc_weight < 1:
            # for decoder initialization with pre-trained RNNLM
            if rnnlm_init:
                assert internal_lm
                assert rnnlm_init.predictor.num_classes == num_classes
                assert rnnlm_init.predictor.num_units == num_units
                assert rnnlm_init.predictor.num_layers == 1  # TODO(hirofumi): on-the-fly

            # for MTL with RNNLM objective
            if rnnlm_task_weight > 0:
                assert internal_lm
                if not share_lm_softmax:
                    self.output_rnnlm = LinearND(num_units, num_classes)

            # Attention
            assert isinstance(attention, AttentionMechanism) or isinstance(attention, MultiheadAttentionMechanism)

            # Decoder
            if internal_lm:
                if rnn_type == 'lstm':
                    self.lstm_internal_lm = nn.LSTMCell(emb_dim, num_units)
                    self.dropout_internal_lm = nn.Dropout(p=dropout_hidden)
                    self.lstm_l0 = nn.LSTMCell(num_units + enc_num_units, num_units)
                elif rnn_type == 'gru':
                    self.gru_internal_lm = nn.GRUCell(emb_dim, num_units)
                    self.dropout_internal_lm = nn.Dropout(p=dropout_hidden)
                    self.gru_l0 = nn.GRUCell(num_units + enc_num_units, num_units)
            else:
                if rnn_type == 'lstm':
                    self.lstm_l0 = nn.LSTMCell(emb_dim + enc_num_units, num_units)
                elif rnn_type == 'gru':
                    self.gru_l0 = nn.GRUCell(emb_dim + enc_num_units, num_units)
            self.dropout_l0 = nn.Dropout(p=dropout_hidden)

            for i_l in six.moves.range(1, num_layers):
                if rnn_type == 'lstm':
                    rnn_i = nn.LSTMCell(num_units, num_units)
                elif rnn_type == 'gru':
                    rnn_i = nn.GRUCell(num_units, num_units)
                setattr(self, rnn_type + '_l' + str(i_l), rnn_i)
                setattr(self, 'dropout_l' + str(i_l), nn.Dropout(p=dropout_hidden))

            # cold fusion
            if rnnlm_cold_fusion:
                self.cf_linear_dec_feat = LinearND(num_units + enc_num_units, num_units)
                if cold_fusion == 'hidden':
                    self.cf_linear_lm_feat = LinearND(rnnlm_cold_fusion.num_units, num_units)
                elif cold_fusion == 'prob':
                    self.cf_linear_lm_feat = LinearND(rnnlm_cold_fusion.num_classes, num_units)
                else:
                    raise ValueError(cold_fusion)
                self.cf_linear_lm_gate = LinearND(num_units * 2, num_units)
                self.output_bn = LinearND(num_units * 2, num_units)

                # fix RNNLM parameters
                for p in self.rnnlm_cf.parameters():
                    p.requires_grad = False
            else:
                self.output_bn = LinearND(num_units + enc_num_units, num_units)

            self.output = LinearND(num_units, num_classes)

            # Embedding
            self.embed = Embedding(num_classes=num_classes,
                                   emb_dim=emb_dim,
                                   dropout=dropout_emb,
                                   ignore_index=pad)

    def forward(self, enc_out, enc_lens, ys):
        """Compute XE loss.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, dec_num_units]`
            enc_lens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains Variables of size `[L]`
        Returns:
            logits (torch.autograd.Variable, float): A tensor of size
                `[B, L, num_classes]`
            aw (torch.autograd.Variable, float): A tensor of size
                `[B, L, T, num_heads]`
            logits_lm (torch.autograd.Variable, float): A tensor of size
                `[B, L, num_classes]`

        """
        device_id = enc_out.get_device()

        # Compute the auxiliary CTC loss
        if self.ctc_weight > 0:
            enc_lens_ctc = np2var(np.fromiter(enc_lens, dtype=np.int32), -1).int()
            y_lens = np2var(np.fromiter([y.size(0) for y in ys], dtype=np.int32), -1).int()
            # NOTE: do not copy to GPUs here

            # Concatenate all elements in ys for warpctc_pytorch
            ys_ctc = torch.cat(ys, dim=0).int()

            # Compute CTC loss
            loss_ctc = self.warpctc_loss(self.output_ctc(enc_out).transpose(0, 1).cpu(),  # time-major
                                         ys_ctc.cpu(), enc_lens_ctc, y_lens)
            # NOTE: ctc loss has already been normalized by batch_size
            # NOTE: index 0 is reserved for blank in warpctc_pytorch

            if device_id >= 0:
                loss_ctc = loss_ctc.cuda(device_id)
            loss = loss_ctc * self.ctc_weight
        else:
            loss_ctc = Variable(enc_out.new(1,).fill_(0.))
            loss = Variable(enc_out.new(1,).fill_(0.))

        if self.ctc_weight == 1:
            loss_acc = {'loss': loss.item(),
                        'loss_att': 0,
                        'loss_ctc': loss_ctc.item(),
                        'loss_lm': 0,
                        'acc': 0}
            return loss, loss_acc

        # Append <sos> and <eos>
        sos = Variable(enc_out.new(1,).fill_(self.sos).long())
        eos = Variable(enc_out.new(1,).fill_(self.eos).long())
        if self.backward:
            ys_in = [torch.cat([eos, y], dim=0) for y in ys]
            ys_out = [torch.cat([y, sos], dim=0) for y in ys]
        else:
            ys_in = [torch.cat([sos, y], dim=0) for y in ys]
            ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        ys_in_pad = pad_list(ys_in, self.pad)
        ys_out_pad = pad_list(ys_out, -1)

        # Initialization
        dec_out, dec_state = self._init_dec_state(enc_out, enc_lens, self.num_layers)
        _dec_out, _dec_state = self._init_dec_state(enc_out, enc_lens, 1)  # for internal LM
        self.score.reset()
        att_weight_step = None
        rnnlm_state = None

        # Pre-computation of embedding
        ys_emb = self.embed(ys_in_pad)
        if self.rnnlm_cf:
            ys_lm_emb = self.rnnlm_cf.embed(ys_in_pad)
            # ys_lm_emb = [self.rnnlm_cf.embed(ys_in_pad[:, t:t + 1])
            #              for t in six.moves.range(ys_in_pad.size(1))]
            # ys_lm_emb = torch.cat(ys_lm_emb, dim=1)

        logits_att, logits_lm = [], []
        for t in six.moves.range(ys_in_pad.size(1)):
            is_sample = t > 0 and self.ss_prob > 0 and random.random() < self.ss_prob

            # Score
            context_vec, att_weight_step = self.score(enc_out, enc_lens, dec_out, att_weight_step)

            # Update RNNLM states for cold fusion
            if self.rnnlm_cf:
                if is_sample:
                    y_lm_emb = self.rnnlm_cf.embed(np.argmax(logits_att[-1].detach(), axis=2).cuda(device_id))
                else:
                    y_lm_emb = ys_lm_emb[:, t:t + 1]
                logits_rnnlm_step, rnnlm_out, rnnlm_state = self.rnnlm_cf.predict(y_lm_emb, rnnlm_state)
            else:
                logits_rnnlm_step, rnnlm_out = None, None

            # Generate
            logits_att_t = self.generate(context_vec, dec_out, logits_rnnlm_step, rnnlm_out)

            # Residual connection
            if self.rnnlm_init and self.internal_lm:
                logits_att_t += _dec_out

            logits_att_t = self.output(logits_att_t)
            logits_att.append(logits_att_t)

            if t == ys_in_pad.size(1) - 1:
                break

            # Sample for scheduled sampling
            if is_sample:
                y_emb = self.embed(np.argmax(logits_att[-1].detach(), axis=2).cuda(device_id))
            else:
                y_emb = ys_emb[:, t + 1:t + 2]

            # Recurrency
            dec_out, dec_state, _dec_out, _dec_state = self.recurrency(
                y_emb, context_vec, dec_state, _dec_state)
            if self.rnnlm_task_weight > 0:
                if self.share_lm_softmax:
                    logits_rnnlm_step = self.output(_dec_out)
                else:
                    logits_rnnlm_step = self.output_rnnlm(_dec_out)
                logits_lm.append(logits_rnnlm_step)

        logits_att = torch.cat(logits_att, dim=1) / self.logits_temp

        # Compute XE sequence loss
        if self.lsm_prob > 0:
            # Label smoothing
            y_lens = [y.size(0) for y in ys_out]
            loss_att = cross_entropy_lsm(
                logits_att, ys=ys_out_pad, y_lens=y_lens,
                lsm_prob=self.lsm_prob, size_average=True)
        else:
            loss_att = F.cross_entropy(
                input=logits_att.view((-1, logits_att.size(2))),
                target=ys_out_pad.view(-1),  # long
                ignore_index=-1, size_average=False) / len(enc_out)
        loss += loss_att * (1 - self.ctc_weight)

        # Compute XE loss for RNNLM objective
        if self.rnnlm_task_weight > 0:
            logits_lm = torch.cat(logits_lm, dim=1)
            loss_lm = F.cross_entropy(
                input=logits_lm.view((-1, logits_lm.size(2))),
                target=ys_out_pad[:, 1:].contiguous().view(-1),
                ignore_index=-1, size_average=True)
            loss += loss_lm * self.rnnlm_task_weight
        else:
            loss_lm = Variable(enc_out.new(1,).fill_(0.))

        # Compute token-level accuracy in teacher-forcing
        pad_pred = logits_att.view(ys_out_pad.size(0), ys_out_pad.size(1), logits_att.size(-1)).argmax(2)
        mask = ys_out_pad != -1
        numerator = torch.sum(pad_pred.masked_select(mask) == ys_out_pad.masked_select(mask))
        denominator = torch.sum(mask)
        acc = float(numerator) / float(denominator)

        loss_acc = {'loss': loss.item(),
                    'loss_att': loss_att.item(),
                    'loss_ctc': loss_ctc.item(),
                    'loss_lm': loss_lm.item(),
                    'acc': acc}
        return loss, loss_acc

    def _init_dec_state(self, enc_out, enc_lens, num_layers):
        """Initialize decoder state.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, dec_num_units]`
            enc_lens (list): A list of length `[B]`
            num_layers (int):
        Returns:
            dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, dec_num_units]`
            dec_state (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):

        """
        batch_size = enc_out.size(0)

        if self.init_with_enc:
            if enc_out.size(-1) == self.num_units:
                # unidirectinal encoder
                dec_out = torch.cat([enc_out[b:b + 1, enc_lens[b] - 1:enc_lens[b]]
                                     for b in six.moves.range(len(enc_lens))], dim=0)
            else:
                raise NotImplementedError()
                # TODO(hirofumi): add bridge layer
                # bidirectional encoder
                dec_out = torch.cat([enc_out[b:b + 1, 0:1, self.num_units:]
                                     for b in six.moves.range(len(enc_lens))], dim=0)
                # NOTE: initialize with reverse direction
            dec_out = torch.tanh(dec_out)
            hx_list = [dec_out.clone().squeeze(1)] * self.num_layers
            cx_list = [dec_out.clone().squeeze(1)] * self.num_layers if self.rnn_type == 'lstm' else None
        else:
            dec_out = Variable(enc_out.new(batch_size, 1, self.num_units).fill_(0.),
                               volatile=not self.training)
            zero_state = Variable(enc_out.new(batch_size, self.num_units).fill_(0.),
                                  volatile=not self.training)
            hx_list = [zero_state] * self.num_layers
            cx_list = [zero_state] * self.num_layers if self.rnn_type == 'lstm' else None

        return dec_out, (hx_list, cx_list)

    def recurrency(self, y_emb, context_vec, dec_state, _dec_state):
        """Recurrency function.

        Args:
            y_emb (torch.autograd.Variable, float): A tensor of size
                `[B, 1, emb_dim]`
            context_vec (torch.autograd.Variable, float): A tensor of size
                `[B, 1, enc_num_units]`
            dec_state (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):
            _dec_state (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):
        Returns:
            dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, num_units]`
            dec_state (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):
            _dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, num_units]`
            _dec_state (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):

        """
        hx_list, cx_list = dec_state
        hx_lm, cx_lm = _dec_state
        y_emb = y_emb.squeeze(1)
        context_vec = context_vec.squeeze(1)

        if self.internal_lm:
            if self.rnn_type == 'lstm':
                hx_lm[0], cx_lm[0] = self.lstm_internal_lm(y_emb, (hx_lm[0], cx_lm[0]))
                hx_lm[0] = self.dropout_internal_lm(hx_lm[0])
                _h_lm = torch.cat([hx_lm[0], context_vec], dim=-1)
                hx_list[0], cx_list[0] = self.lstm_l0(_h_lm, (hx_list[0], cx_list[0]))
            elif self.rnn_type == 'gru':
                hx_lm = self.gru_internal_lm(y_emb, hx_lm)
                hx_lm = self.dropout_internal_lm(hx_lm)
                _h_lm = torch.cat([hx_lm, context_vec], dim=-1)
                hx_list[0] = self.gru_l0(_h_lm, hx_list[0])
        else:
            if self.rnn_type == 'lstm':
                hx_list[0], cx_list[0] = self.lstm_l0(torch.cat([y_emb, context_vec], dim=-1), (hx_list[0], cx_list[0]))
            elif self.rnn_type == 'gru':
                hx_list[0] = self.gru_l0(torch.cat([y_emb, context_vec], dim=-1), hx_list[0])
        hx_list[0] = self.dropout_l0(hx_list[0])

        for i_l in six.moves.range(1, self.num_layers):
            if self.rnn_type == 'lstm':
                hx_list[i_l], cx_list[i_l] = getattr(self, 'lstm_l' + str(i_l))(
                    hx_list[i_l - 1], (hx_list[i_l], cx_list[i_l]))
            elif self.rnn_type == 'gru':
                hx_list[i_l] = getattr(self, 'gru_l' + str(i_l))(
                    hx_list[i_l - 1], hx_list[i_l])
            hx_list[i_l] = getattr(self, 'dropout_l' + str(i_l))(hx_list[i_l])

            # Residual connection
            if self.residual:
                hx_list[i_l] += hx_list[i_l - 1]

        dec_out = hx_list[-1].unsqueeze(1)
        _dec_out = hx_lm[0].unsqueeze(1)
        return dec_out, (hx_list, cx_list), _dec_out, (hx_lm, cx_lm)

    def generate(self, context_vec, dec_out, logits_rnnlm_step, rnnlm_out):
        """Generate function.

        Args:
            context_vec (torch.autograd.Variable, float): A tensor of size
                `[B, 1, enc_num_units]`
            dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, dec_num_units]`
            logits_rnnlm_step (torch.autograd.Variable, float): A tensor of size
                `[B, 1, num_classes]`
            rnnlm_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, lm_num_units]`
        Returns:
            logits_t (torch.autograd.Variable, float): A tensor of size
                `[B, 1, num_classes]`

        """
        if self.rnnlm_cf:
            # cold fusion
            if self.cold_fusion == 'hidden':
                lm_feat = self.cf_linear_lm_feat(rnnlm_out)
            elif self.cold_fusion == 'prob':
                lm_feat = self.cf_linear_lm_feat(logits_rnnlm_step)
            dec_feat = self.cf_linear_dec_feat(torch.cat([dec_out, context_vec], dim=-1))
            gate = F.sigmoid(self.cf_linear_lm_gate(torch.cat([dec_feat, lm_feat], dim=-1)))
            gated_lm_feat = gate * lm_feat
            logits_t = self.output_bn(torch.cat([dec_feat, gated_lm_feat], dim=-1))
        else:
            logits_t = self.output_bn(torch.cat([dec_out, context_vec], dim=-1))

        # TODO(hirofumi): add non-linearity
        # logits_t = F.tanh(logits_t)

        return logits_t

    def greedy(self, enc_out, enc_lens, max_len_ratio, exclude_eos=False):
        """Greedy decoding in the inference stage.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, encoder_num_units]`
            enc_lens (list): A list of length `[B]`
            max_len_ratio (int): the maximum sequence length of tokens
            exclude_eos (bool):
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`

        """
        batch_size, enc_time, enc_num_units = enc_out.size()

        # Initialization
        dec_out, dec_state = self._init_dec_state(enc_out, enc_lens, self.num_layers)
        _dec_out, _dec_state = self._init_dec_state(enc_out, enc_lens, 1)
        self.score.reset()
        att_weight_step = None
        rnnlm_state = None

        if self.backward:
            sos, eos = self.eos, self.sos
        else:
            sos, eos = self.sos, self.eos

        # Start from <sos> (<eos> in case of the backward decoder)
        y = Variable(enc_out.new(batch_size, 1).fill_(sos).long(), volatile=True)

        _best_hyps, _aw = [], []
        y_lens = np.zeros((batch_size,), dtype=np.int32)
        eos_flags = [False] * batch_size
        for t in six.moves.range(int(math.floor(enc_time * max_len_ratio)) + 1):
            # Score
            context_vec, att_weight_step = self.score(enc_out, enc_lens, dec_out, att_weight_step)

            # Update RNNLM states for cold fusion
            if self.rnnlm_cf:
                y_lm = self.rnnlm_cf.embed(y)
                logits_rnnlm_step, rnnlm_out, rnnlm_state = self.rnnlm_cf.predict(y_lm, rnnlm_state)
            else:
                logits_rnnlm_step, rnnlm_out = None, None

            # Generate
            logits_t = self.generate(context_vec, dec_out, logits_rnnlm_step, rnnlm_out)

            # residual connection
            if self.rnnlm_init and self.internal_lm:
                logits_t += _dec_out

            if self.share_lm_softmax or self.rnnlm_init:
                logits_t = self.output_bn(logits_t)
            logits_t = self.output(logits_t)

            # Pick up 1-best
            device_id = logits_t.get_device()
            y = np.argmax(logits_t.squeeze(1).detach(), axis=1).cuda(device_id).unsqueeze(1)
            _best_hyps += [y]
            _aw += [att_weight_step]

            # Count lengths of hypotheses
            for b in six.moves.range(batch_size):
                if not eos_flags[b]:
                    if y[b].item() == eos:
                        eos_flags[b] = True
                    y_lens[b] += 1
                    # NOTE: include <eos>

            # Break if <eos> is outputed in all mini-batch_size
            if sum(eos_flags) == batch_size:
                break

            # Recurrency
            y_emb = self.embed(y)
            dec_out, dec_state, _dec_out, _dec_state = self.recurrency(
                y_emb, context_vec, dec_state, _dec_state)

        # Concatenate in L dimension
        _best_hyps = torch.cat(_best_hyps, dim=1)
        _aw = torch.stack(_aw, dim=1)

        # Convert to numpy
        _best_hyps = var2np(_best_hyps)
        _aw = var2np(_aw)

        if self.score.num_heads > 1:
            _aw = _aw[:, :, :, 0]
            # TODO(hirofumi): fix for MHA

        # Truncate by the first <eos> (<sos> in case of the backward decoder)
        if self.backward:
            # Reverse the order
            best_hyps = [_best_hyps[b, :y_lens[b]][::-1] for b in six.moves.range(batch_size)]
            aw = [_aw[b, :y_lens[b]][::-1] for b in six.moves.range(batch_size)]
        else:
            best_hyps = [_best_hyps[b, :y_lens[b]] for b in six.moves.range(batch_size)]
            aw = [_aw[b, :y_lens[b]] for b in six.moves.range(batch_size)]

        # Exclude <eos> (<sos> in case of the backward decoder)
        if exclude_eos:
            if self.backward:
                best_hyps = [best_hyps[b][1:] if eos_flags[b]
                             else best_hyps[b] for b in six.moves.range(batch_size)]
            else:
                best_hyps = [best_hyps[b][:-1] if eos_flags[b]
                             else best_hyps[b] for b in six.moves.range(batch_size)]

        return best_hyps, aw

    def beam_search(self, enc_out, enc_lens, params, rnnlm, exclude_eos=False):
        """Beam search decoding in the inference stage.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, dec_num_units]`
            enc_lens (list): A list of length `[B]`
            params (dict):
                beam_width (int): the size of beam
                max_len_ratio (int): the maximum sequence length of tokens
                min_len_ratio (float): the minimum sequence length of tokens
                length_penalty (float): length penalty
                coverage_penalty (float): coverage penalty
                coverage_threshold (float): threshold for coverage penalty
                rnnlm_weight (float): the weight of RNNLM score
            rnnlm (torch.nn.Module):
            exclude_eos (bool):
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`

        """
        batch_size = enc_out.size(0)

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

        best_hyps, aw = [], []
        y_lens = np.zeros((batch_size,), dtype=np.int32)
        eos_flags = [False] * batch_size
        for b in six.moves.range(batch_size):
            # Initialization per utterance
            dec_out, (hx_list, cx_list) = self._init_dec_state(enc_out[b:b + 1], enc_lens[b:b + 1], self.num_layers)
            _dec_out, _dec_state = self._init_dec_state(enc_out[b:b + 1], enc_lens[b:b + 1], 1)
            context_vec = Variable(enc_out.new(1, 1, enc_out.size(-1)).fill_(0.), volatile=True)
            self.score.reset()

            complete = []
            beam = [{'hyp': [sos],
                     'score': 0,  # log 1
                     'dec_out': dec_out,
                     'hx_list': hx_list,
                     'cx_list': cx_list,
                     'context_vec': context_vec,
                     'att_weight_steps': [None],
                     'rnnlm_hx_list': None,
                     'rnnlm_cx_list': None,
                     'prev_cov': 0,
                     '_dec_out': _dec_out,
                     '_dec_state': _dec_state}]
            for t in six.moves.range(int(math.floor(enc_lens[b] * params['max_len_ratio'])) + 1):
                new_beam = []
                for i_beam in six.moves.range(len(beam)):
                    if t > 0:
                        # Recurrency
                        y = Variable(enc_out.new(1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                        y_emb = self.embed(y)
                        dec_out, (hx_list, cx_list), _dec_out, _dec_state = self.recurrency(
                            y_emb, beam[i_beam]['context_vec'],
                            (beam[i_beam]['hx_list'], beam[i_beam]['cx_list']),
                            beam[i_beam]['_dec_state'])
                    else:
                        dec_out = beam[i_beam]['dec_out']

                    # Score
                    context_vec, att_weight_step = self.score(enc_out[b:b + 1, :enc_lens[b]],
                                                              enc_lens[b:b + 1],
                                                              dec_out,
                                                              beam[i_beam]['att_weight_steps'][-1])

                    if self.rnnlm_cf:
                        # Update RNNLM states for cold fusion
                        y_lm = Variable(enc_out.new(1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                        y_lm_emb = self.rnnlm_cf.embed(y_lm).squeeze(1)
                        logits_rnnlm_step, rnnlm_out, rnnlm_state = self.rnnlm_cf.predict(
                            y_lm_emb, (beam[i_beam]['rnnlm_hx_list'], beam[i_beam]['rnnlm_cx_list']))
                    elif rnnlm is not None:
                        # Update RNNLM states for shallow fusion
                        y_lm = Variable(enc_out.new(1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                        y_lm_emb = rnnlm.embed(y_lm).squeeze(1)
                        logits_rnnlm_step, rnnlm_out, rnnlm_state = rnnlm.predict(
                            y_lm_emb, (beam[i_beam]['rnnlm_hx_list'], beam[i_beam]['rnnlm_cx_list']))
                    else:
                        logits_rnnlm_step, rnnlm_out, rnnlm_state = None, None, None

                    # Generate
                    logits_t = self.generate(context_vec, dec_out, logits_rnnlm_step, rnnlm_out)

                    # residual connection
                    if self.rnnlm_init and self.internal_lm:
                        if t == 0:
                            logits_t += beam[i_beam]['_dec_out']
                        else:
                            logits_t += _dec_out

                    if self.share_lm_softmax or self.rnnlm_init:
                        logits_t = self.output_bn(logits_t)
                    logits_t = self.output(logits_t)

                    # Path through the softmax layer & convert to log-scale
                    log_probs = F.log_softmax(logits_t.squeeze(1), dim=1)
                    # log_probs = logits_t.squeeze(1)
                    # NOTE: `[1 (B), 1, num_classes]` -> `[1 (B), num_classes]`

                    # Pick up the top-k scores
                    log_probs_topk, indices_topk = torch.topk(
                        log_probs, k=params['beam_width'], dim=1, largest=True, sorted=True)

                    for k in six.moves.range(params['beam_width']):
                        # Exclude short hypotheses
                        if indices_topk[0, k].item() == eos and len(beam[i_beam]['hyp']) < enc_lens[b] * params['min_len_ratio']:
                            continue

                        # Add length penalty
                        score = beam[i_beam]['score'] + log_probs_topk[0, k].item() + params['length_penalty']

                        # Add coverage penalty
                        if params['coverage_penalty'] > 0:
                            # Recompute converage penalty in each step
                            score -= beam[i_beam]['prev_cov'] * params['coverage_penalty']

                            att_weight_stack = torch.stack(
                                beam[i_beam]['att_weight_steps'][1:] + [att_weight_step], dim=1)

                            if self.score.num_heads > 1:
                                cov_sum = att_weight_stack[0, :, :, 0].detach().cpu().numpy()
                                # TODO(hirofumi): fix for MHA
                            else:
                                cov_sum = att_weight_stack.detach().cpu().numpy()
                            if params['coverage_threshold'] == 0:
                                cov_sum = np.sum(cov_sum)
                            else:
                                cov_sum = np.sum(cov_sum[np.where(cov_sum > params['coverage_threshold'])[0]])
                            score += cov_sum * params['coverage_penalty']
                        else:
                            cov_sum = 0

                        # Add RNNLM score
                        if params['rnnlm_weight'] > 0:
                            lm_log_probs = F.log_softmax(logits_rnnlm_step.squeeze(1), dim=1)
                            assert log_probs.size() == lm_log_probs.size()
                            score += lm_log_probs[0, indices_topk[0, k].item()].item() * params['rnnlm_weight']

                        new_beam.append(
                            {'hyp': beam[i_beam]['hyp'] + [indices_topk[0, k].item()],
                             'score': score,
                             'hx_list': hx_list[:],
                             'cx_list': cx_list[:],
                             'dec_out': dec_out,
                             'context_vec': context_vec,
                             'att_weight_steps': beam[i_beam]['att_weight_steps'] + [att_weight_step],
                             'rnnlm_hx_list': rnnlm_state[0][:] if rnnlm_state is not None else None,
                             'rnnlm_cx_list': rnnlm_state[1][:] if rnnlm_state is not None else None,
                             'prev_cov': cov_sum,
                             '_dec_out': _dec_out,
                             '_dec_state': _dec_state[:]})

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

            if len(complete) == 0:
                complete = beam

            complete = sorted(complete, key=lambda x: x['score'], reverse=True)
            best_hyps += [np.array(complete[0]['hyp'][1:])]
            aw += [complete[0]['att_weight_steps'][1:]]
            y_lens[b] = len(complete[0]['hyp'][1:])
            if complete[0]['hyp'][-1] == eos:
                eos_flags[b] = True

        logger.info('log prob (hyp): %.3f' % complete[0]['score'])
        logger.info('log prob (ref): ')

        # Concatenate in L dimension
        for b in six.moves.range(len(aw)):
            aw[b] = var2np(torch.stack(aw[b], dim=1).squeeze(0))
            if self.score.num_heads > 1:
                aw[b] = aw[b][:, :, 0]
                # TODO(hirofumi): fix for MHA

        # Reverse the order
        if self.backward:
            best_hyps = [best_hyps[b][::-1] for b in six.moves.range(batch_size)]
            aw = [aw[b][::-1] for b in six.moves.range(batch_size)]

        # Exclude <eos> (<sos> in case of the backward decoder)
        if exclude_eos:
            if self.backward:
                best_hyps = [best_hyps[b][1:] if eos_flags[b]
                             else best_hyps[b] for b in six.moves.range(batch_size)]
            else:
                best_hyps = [best_hyps[b][:-1] if eos_flags[b]
                             else best_hyps[b] for b in six.moves.range(batch_size)]

        return best_hyps, aw

    def decode_ctc(self, enc_out, x_lens, beam_width=1, rnnlm=None):
        """Decoding by the CTC layer in the inference stage.

            This is only used for Joint CTC-Attention model.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, encoder_num_units]`
            beam_width (int): the size of beam
            rnnlm ():
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            perm_idx (list): A list of length `[B]`

        """
        # Path through the softmax layer
        batch_size, max_time = enc_out.size()[:2]
        enc_out = enc_out.view(batch_size * max_time, -1).contiguous()
        logits_ctc = self.output_ctc(enc_out)
        logits_ctc = logits_ctc.view(batch_size, max_time, -1)

        if beam_width == 1:
            best_hyps = self.decode_ctc_greedy(var2np(logits_ctc), x_lens)
        else:
            best_hyps = self.decode_ctc_beam(var2np(F.log_softmax(logits_ctc, dim=-1)),
                                             x_lens, beam_width, rnnlm)
            # TODO(hirofumi: decoding paramters

        return best_hyps
