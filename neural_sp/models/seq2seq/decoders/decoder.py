#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN decoder (including CTC loss calculation)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import copy
import math
import numpy as np
import random
import six
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _assert_no_grad
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
            0 means that decoder inputs are represented by one-hot vectors.
        num_classes (int): the number of nodes in softmax layer (excluding <sos> and <eos> classes)
        logits_temp (float): a parameter for smoothing the softmax layer in outputing probabilities
        dropout_dec (float): the probability to drop nodes in the RNN layer
        dropout_emb (float): the probability to drop nodes of the embedding layer
        ss_prob (float): the probability of scheduled sampling
        lsm_prob (float): the probability of label smoothing
        lsm_type (str): uniform or unigram
        ctc_weight (float):
        backward (bool): decode in the backward order
        rnnlm_cf (torch.nn.Module):
        cold_fusion_type (str): the type of cold fusion
            prob: probability from RNNLM
            hidden: hidden states of RNNLM
        internal_lm ():
        rnnlm_init (torch.nn.Module):
        rnnlm_weight (float): the weight for the auxiliary XE loss of RNNLM objective
        share_softmax (bool):

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
                 dropout_dec,
                 dropout_emb,
                 ss_prob,
                 lsm_prob,
                 lsm_type,
                 ctc_weight=0,
                 ctc_fc_list=[],
                 backward=False,
                 rnnlm_cf=None,
                 cold_fusion_type=False,
                 internal_lm=False,
                 rnnlm_init=None,
                 rnnlm_weight=0,
                 share_softmax=False):

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
        self.dropout_dec = dropout_dec
        self.dropout_emb = dropout_emb
        self.ss_prob = ss_prob
        self.lsm_prob = lsm_prob
        self.lsm_type = lsm_type
        self.ctc_weight = ctc_weight
        self.ctc_fc_list = ctc_fc_list
        self.backward = backward
        self.rnnlm_cf = rnnlm_cf
        self.cold_fusion_type = cold_fusion_type
        self.internal_lm = internal_lm
        self.rnnlm_init = rnnlm_init
        self.rnnlm_weight = rnnlm_weight
        self.share_softmax = share_softmax

        if ctc_weight > 0:
            # Fully-connected layers for CTC
            if len(ctc_fc_list) > 0:
                fc_layers = OrderedDict()
                for i in range(len(ctc_fc_list)):
                    input_dim = enc_num_units if i == 0 else ctc_fc_list[i - 1]
                    fc_layers['fc' + str(i)] = LinearND(input_dim, ctc_fc_list[i],
                                                        dropout=dropout_dec)
                fc_layers['output'] = LinearND(ctc_fc_list[-1], num_classes)
                self.output_ctc = nn.Sequential(fc_layers)
            else:
                self.output_ctc = LinearND(enc_num_units, num_classes)
            self.decode_ctc_greedy = GreedyDecoder(blank_index=0)
            self.decode_ctc_beam = BeamSearchDecoder(blank_index=0)

        if ctc_weight == 1:
            return None

        # for decoder initialization with pre-trained RNNLM
        if rnnlm_init is not None:
            assert internal_lm
            assert rnnlm_init.predictor.num_classes == num_classes
            assert rnnlm_init.predictor.num_units == num_units
            assert rnnlm_init.predictor.num_layers == 1  # on-the-fly

        # for MTL with RNNLM objective
        if rnnlm_weight > 0:
            assert internal_lm
            if not share_softmax:
                self.output_rnnlm = LinearND(num_units, num_classes)

        # Attention
        assert isinstance(attention, AttentionMechanism) or isinstance(attention, MultiheadAttentionMechanism)

        # Decoder
        if internal_lm:
            if rnn_type == 'lstm':
                self.lstm_internal_lm = nn.LSTMCell(emb_dim, num_units)
                self.lstm_l0 = nn.LSTMCell(enc_num_units + num_units, num_units)
            elif rnn_type == 'gru':
                self.gru_internal_lm = nn.GRUCell(emb_dim, num_units)
                self.gru_l0 = nn.GRUCell(enc_num_units + num_units, num_units)
            self.dropout_internal_lm = nn.Dropout(p=dropout_dec)
        else:
            if rnn_type == 'lstm':
                self.lstm_l0 = nn.LSTMCell(emb_dim + enc_num_units, num_units)
            elif rnn_type == 'gru':
                self.gru_l0 = nn.GRUCell(emb_dim + enc_num_units, num_units)
        self.dropout_l0 = nn.Dropout(p=dropout_dec)

        for i_l in six.moves.range(1, num_layers):
            if rnn_type == 'lstm':
                rnn_i = nn.LSTMCell(num_units, num_units)
            elif rnn_type == 'gru':
                rnn_i = nn.GRUCell(num_units, num_units)
            setattr(self, rnn_type + '_l' + str(i_l), rnn_i)
            setattr(self, 'dropout_l' + str(i_l), nn.Dropout(p=dropout_dec))

        # cold fusion
        if rnnlm_cf is not None:
            assert cold_fusion_type in ['hidden', 'prob']
            self.cf_fc_dec_feat = LinearND(num_units + enc_num_units, num_units)
            if cold_fusion_type == 'hidden':
                self.cf_fc_lm_feat = LinearND(rnnlm_cf.num_units, num_units)
            elif cold_fusion_type == 'prob':
                # probability projection
                self.cf_fc_lm_feat = LinearND(rnnlm_cf.num_classes, num_units)
            self.cf_fc_lm_gate = LinearND(num_units * 2, num_units)
            self.output_bottle = LinearND(num_units * 2, num_units)
            self.output = LinearND(num_units, num_classes)

            # fix RNNLM parameters
            for name, param in self.rnnlm_cf.named_parameters():
                param.requires_grad = False
        else:
            self.output = LinearND(num_units + enc_num_units, num_classes)

        # Embedding
        self.emb = Embedding(num_classes=num_classes,
                             emb_dim=emb_dim,
                             dropout=dropout_emb,
                             ignore_index=pad)

    def _init_dec_state(self, enc_out, num_layers):
        """Initialize decoder state.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, enc_num_units]`
        Returns:
            dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, dec_num_units]`
            dec_state (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):

        """
        batch_size = enc_out.size(0)
        dec_out = Variable(enc_out.data.new(batch_size, 1, self.num_units).fill_(0.),
                           volatile=not self.training)
        zero_state = Variable(enc_out.data.new(batch_size, self.num_units).fill_(0.),
                              volatile=not self.training)
        hx_list = [zero_state] * self.num_layers
        cx_list = [zero_state] * self.num_layers if self.rnn_type == 'lstm' else None
        return dec_out, (hx_list, cx_list)

    def forward(self, enc_out, x_lens, ys):
        """Decoding in the training stage.　Compute XE loss.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, enc_num_units]`
            x_lens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains Variables of size `[L]`
        Returns:
            logits (torch.autograd.Variable, float): A tensor of size
                `[B, L, num_classes]`
            aw (torch.autograd.Variable, float): A tensor of size
                `[B, L, T, num_heads]`
            logits_lm (torch.autograd.Variable, float): A tensor of size
                `[B, L, num_classes]`

        """
        # Compute the auxiliary CTC loss
        if self.ctc_weight > 0:
            logits_ctc = self.output_ctc(enc_out)
            loss_ctc = self._compute_ctc_loss(logits_ctc, x_lens, ys)
            device_id = enc_out.get_device()
            if device_id >= 0:
                loss_ctc = loss_ctc.cuda(device_id)
            loss = loss_ctc * self.ctc_weight
        else:
            loss_ctc = 0
            loss = 0.

        if self.ctc_weight == 1:
            loss_acc = {'loss': loss,
                        'loss_att': 0,
                        'loss_ctc': loss_ctc,
                        'loss_lm': 0,
                        'acc': 0}
            return loss_acc

        # Append <sos> and <eos>
        sos = Variable(enc_out.data.new(1,).fill_(self.sos).long())
        eos = Variable(enc_out.data.new(1,).fill_(self.eos).long())
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        ys_in_pad = pad_list(ys_in, self.pad)
        ys_out_pad = pad_list(ys_out, -1)

        # Initialization
        dec_out, dec_state = self._init_dec_state(enc_out, self.num_layers)
        _dec_out, _dec_state = self._init_dec_state(enc_out, 1)  # for internal LM
        self.score.reset()
        aw_t = None
        rnnlm_state = None

        # Pre-computation of embedding
        ys_emb = self.emb(ys_in_pad)
        if self.rnnlm_cf is not None:
            ys_lm_emb = [self.rnnlm_cf.emb(ys_in_pad[:, t:t + 1])
                         for t in six.moves.range(ys_in_pad.size(1))]
            ys_lm_emb = torch.cat(ys_lm_emb, dim=1)

        logits_att, logits_lm = [], []
        for t in six.moves.range(ys_in_pad.size(1)):
            is_sample = t > 0 and self.ss_prob > 0 and random.random() < self.ss_prob

            # Score
            cv, aw_t = self.score(enc_out, x_lens, dec_out, aw_t)

            # Update RNNLM states for cold fusion
            if self.rnnlm_cf is not None:
                if is_sample:
                    y_lm_emb = self.rnnlm_cf.emb(torch.max(logits_att[-1], dim=2)[1]).detach()
                else:
                    y_lm_emb = ys_lm_emb[:, t:t + 1]
                logits_lm_t, lm_out, rnnlm_state = self.rnnlm_cf.predict(y_lm_emb, rnnlm_state)
            else:
                logits_lm_t, lm_out = None, None

            # Generate
            logits_att_t = self._generate(cv, dec_out, logits_lm_t, lm_out)

            # Residual connection
            if self.rnnlm_init is not None and self.internal_lm:
                logits_att_t += _dec_out

            if self.share_softmax or self.rnnlm_init is not None:
                logits_att_t = self.output_bottle(logits_att_t)
            logits_att_t = self.output(logits_att_t)
            if self.rnnlm_cf is not None:
                logits_att_t = F.relu(logits_att_t)
            logits_att.append(logits_att_t)

            if t == ys_in_pad.size(1) - 1:
                break

            # Sample for scheduled sampling
            if is_sample:
                y_emb = self.emb(torch.max(logits_att[-1], dim=2)[1]).detach()
            else:
                y_emb = ys_emb[:, t + 1:t + 2]

            # Recurrency
            dec_out, dec_state, _dec_out, _dec_state = self._recurrency(
                y_emb, cv, dec_state, _dec_state)
            if self.rnnlm_weight > 0:
                if self.share_softmax:
                    logits_lm_t = self.output(_dec_out)
                else:
                    logits_lm_t = self.output_rnnlm(_dec_out)
                logits_lm.append(logits_lm_t)

        logits_att = torch.cat(logits_att, dim=1) / self.logits_temp

        # Compute XE sequence loss
        if self.lsm_prob > 0:
            # Label smoothing
            y_lens = [y.size(0) for y in ys_out]
            loss_att = cross_entropy_lsm(
                logits_att, ys=ys_out_pad, y_lens=y_lens,
                lsm_prob=self.lsm_prob, lsm_type=self.lsm_type,
                size_average=True)
        else:
            loss_att = F.cross_entropy(
                input=logits_att.view((-1, logits_att.size(2))),
                target=ys_out_pad.view(-1),  # long
                ignore_index=-1, size_average=False) / len(enc_out)
        loss += loss_att * (1 - self.ctc_weight)

        # Compute XE loss for RNNLM objective
        if self.rnnlm_weight > 0:
            logits_lm = torch.cat(logits_lm, dim=1)
            loss_lm = F.cross_entropy(
                input=logits_lm.view((-1, logits_lm.size(2))),
                target=ys_out_pad[:, 1:].contiguous().view(-1),
                ignore_index=-1, size_average=True)
            loss += loss_lm * self.rnnlm_weight
        else:
            loss_lm = 0

        # Compute token-level accuracy in teacher-forcing
        pad_pred = logits_att.data.view(ys_out_pad.size(0), ys_out_pad.size(1), logits_att.size(-1)).max(2)[1]
        mask = ys_out_pad.data != -1
        numerator = torch.sum(pad_pred.masked_select(mask) == ys_out_pad.data.masked_select(mask))
        denominator = torch.sum(mask)
        acc = float(numerator) / float(denominator)

        loss_acc = {'loss': loss,
                    'loss_att': loss_att,
                    'loss_ctc': loss_ctc,
                    'loss_lm': loss_lm,
                    'acc': acc}
        return loss_acc

    def _compute_ctc_loss(self, logits, x_lens, ys):
        """Compute CTC loss.

        Args:
            logits (torch.autograd.Variable, float): A tensor of size
                `[B, T, enc_num_units]`
            x_lens (list): A list of length `[B]`
            ys (torch.autograd.Variable, long): A tensor of size `[B, L]`,
                which does not include <sos> nor <eos>
        Returns:
            loss (torch.autograd.Variable, float): A tensor of size `[1]`

        """
        x_lens = np2var(np.fromiter(x_lens, dtype=np.int32), -1).int()
        y_lens = np2var(np.fromiter([y.size(0) for y in ys], dtype=np.int32), -1).int()
        # NOTE: do not copy to GPUs here

        # Concatenate all elements in ys for warpctc_pytorch
        ys_ctc = torch.cat(ys, dim=0).int()

        # Compute CTC loss
        loss = warpctc(logits.transpose(0, 1).cpu(),  # time-major
                       ys_ctc.cpu(), x_lens, y_lens)
        # NOTE: ctc loss has already been normalized by batch_size
        # NOTE: index 0 is reserved for blank in warpctc_pytorch

        return loss

    def _recurrency(self, y_emb, cv, dec_state, _dec_state):
        """Recurrency function.

        Args:
            y_emb (torch.autograd.Variable, float): A tensor of size
                `[B, 1, emb_dim]`
            cv (torch.autograd.Variable, float): A tensor of size
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
        cv = cv.squeeze(1)

        if self.internal_lm:
            if self.rnn_type == 'lstm':
                hx_lm[0], cx_lm[0] = self.lstm_internal_lm(y_emb, (hx_lm[0], cx_lm[0]))
                hx_lm[0] = self.dropout_internal_lm(hx_lm[0])
                _h_lm = torch.cat([cv, hx_lm[0]], dim=-1)
                hx_list[0], cx_list[0] = self.lstm_l0(_h_lm, (hx_list[0], cx_list[0]))
            elif self.rnn_type == 'gru':
                hx_lm = self.gru_internal_lm(y_emb, hx_lm)
                hx_lm = self.dropout_internal_lm(hx_lm)
                _h_lm = torch.cat([cv, hx_lm], dim=-1)
                hx_list[0] = self.gru_l0(_h_lm, hx_list[0])
        else:
            if self.rnn_type == 'lstm':
                hx_list[0], cx_list[0] = self.lstm_l0(torch.cat([y_emb, cv], dim=-1), (hx_list[0], cx_list[0]))
            elif self.rnn_type == 'gru':
                hx_list[0] = self.gru_l0(torch.cat([y_emb, cv], dim=-1), hx_list[0])
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

    def _generate(self, cv, dec_out, logits_lm_t, lm_out):
        """Generate function.

        Args:
            cv (torch.autograd.Variable, float): A tensor of size
                `[B, 1, enc_num_units]`
            dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, dec_num_units]`
            logits_lm_t (torch.autograd.Variable, float): A tensor of size
                `[B, 1, num_classes]`
            lm_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, lm_num_units]`
        Returns:
            logits_t (torch.autograd.Variable, float): A tensor of size
                `[B, 1, num_classes]`

        """
        if self.rnnlm_cf is not None:
            # cold fusion
            if self.cold_fusion_type == 'prob':
                lm_feat = self.cf_fc_lm_feat(logits_lm_t)
            else:
                lm_feat = self.cf_fc_lm_feat(lm_out)
            dec_feat = self.cf_fc_dec_feat(torch.cat([dec_out, cv], dim=-1))
            gate = F.sigmoid(self.cf_fc_lm_gate(torch.cat([dec_feat, lm_feat], dim=-1)))
            gated_lm_feat = gate * lm_feat
            logits_t = self.output_bottle(torch.cat([dec_feat, gated_lm_feat], dim=-1))
        else:
            logits_t = torch.cat([dec_out, cv], dim=-1)
        return logits_t

    def greedy(self, enc_out, x_lens, max_len_ratio, exclude_eos=False):
        """Greedy decoding in the inference stage.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, encoder_num_units]`
            x_lens (list): A list of length `[B]`
            max_len_ratio (int): the maximum sequence length of tokens
            exclude_eos (bool):
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`

        """
        batch_size, enc_time, enc_num_units = enc_out.size()

        # Initialization
        dec_out, dec_state = self._init_dec_state(enc_out, self.num_layers)
        _dec_out, _dec_state = self._init_dec_state(enc_out, 1)
        self.score.reset()
        aw_t = None
        rnnlm_state = None

        # Start from <sos>
        y = Variable(enc_out.data.new(batch_size, 1).fill_(self.sos).long(), volatile=True)

        _best_hyps, _aw = [], []
        y_lens = np.zeros((batch_size,), dtype=np.int32)
        eos_flags = [False] * batch_size
        for t in six.moves.range(int(math.floor(enc_time * max_len_ratio)) + 1):
            # Score
            cv, aw_t = self.score(enc_out, x_lens, dec_out, aw_t)

            # Update RNNLM states for cold fusion
            if self.rnnlm_cf is not None:
                y_lm = self.rnnlm_cf.emb(y)
                logits_lm_t, lm_out, rnnlm_state = self.rnnlm_cf.predict(y_lm, rnnlm_state)
            else:
                logits_lm_t, lm_out = None, None

            # Generate
            logits_t = self._generate(cv, dec_out, logits_lm_t, lm_out)

            # residual connection
            if self.rnnlm_init is not None and self.internal_lm:
                logits_t += _dec_out

            if self.share_softmax or self.rnnlm_init is not None:
                logits_t = self.output_bottle(logits_t)
            logits_t = self.output(logits_t)
            if self.rnnlm_cf is not None:
                logits_t = F.relu(logits_t)

            # Pick up 1-best
            y = torch.max(logits_t.squeeze(1), dim=1)[1].unsqueeze(1)
            _best_hyps += [y]
            _aw += [aw_t]

            # Count lengths of hypotheses
            for b in six.moves.range(batch_size):
                if not eos_flags[b]:
                    if y.data.cpu().numpy()[b] == self.eos:
                        eos_flags[b] = True
                    y_lens[b] += 1
                    # NOTE: include <eos>

            # Break if <eos> is outputed in all mini-batch_size
            if sum(eos_flags) == batch_size:
                break

            # Recurrency
            y_emb = self.emb(y)
            dec_out, dec_state, _dec_out, _dec_state = self._recurrency(
                y_emb, cv, dec_state, _dec_state)

        # Concatenate in L dimension
        _best_hyps = torch.cat(_best_hyps, dim=1)
        _aw = torch.stack(_aw, dim=1)

        # Convert to numpy
        _best_hyps = var2np(_best_hyps)
        _aw = var2np(_aw)

        if self.score.num_heads > 1:
            _aw = _aw[:, :, :, 0]
            # TODO(hirofumi): fix for MHA

        # Truncate by the first <eos>
        if self.backward:
            # Reverse the order
            best_hyps = [_best_hyps[b, :y_lens[b]][::-1] for b in six.moves.range(batch_size)]
            aw = [_aw[b, :y_lens[b]][::-1] for b in six.moves.range(batch_size)]
        else:
            best_hyps = [_best_hyps[b, :y_lens[b]] for b in six.moves.range(batch_size)]
            aw = [_aw[b, :y_lens[b]] for b in six.moves.range(batch_size)]

        # Exclude <eos>
        if exclude_eos:
            best_hyps = [best_hyps[b][:-1] if eos_flags[b]
                         else best_hyps[b] for b in six.moves.range(batch_size)]

        return best_hyps, aw

    def beam_search(self, enc_out, x_lens, params, rnnlm=None, exclude_eos=False):
        """Beam search decoding in the inference stage.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, enc_num_units]`
            x_lens (list): A list of length `[B]`
            params (dict):
                beam_width (int): the size of beam
                max_len_ratio (int): the maximum sequence length of tokens
                min_len_ratio (float): the minimum sequence length of tokens
                len_penalty (float): length penalty
                cov_penalty (float): coverage penalty
                cov_threshold (float): threshold for coverage penalty
                rnnlm_weight (float): the weight of RNNLM score
            rnnlm (torch.nn.Module):
            exclude_eos (bool):
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`

        """
        batch_size = enc_out.size(0)

        # For cold fusion
        if params['rnnlm_weight'] > 0 and not self.cold_fusion_type:
            assert self.rnnlm_cf is not None
            assert not self.rnnlm_cf.training

        # For shallow fusion
        if rnnlm is not None:
            assert not rnnlm.training

        best_hyps, aw = [], []
        y_lens = np.zeros((batch_size,), dtype=np.int32)
        eos_flags = [False] * batch_size
        for b in six.moves.range(batch_size):
            # Initialization per utterance
            dec_out, dec_state = self._init_dec_state(enc_out[b:b + 1], self.num_layers)
            _dec_out, _dec_state = self._init_dec_state(enc_out[b:b + 1], 1)
            cv = Variable(enc_out.data.new(1, 1, enc_out.size(-1)).fill_(0.), volatile=True)
            self.score.reset()

            complete = []
            beam = [{'hyp': [self.sos],
                     'score': 0,  # log 1
                     'dec_out': dec_out,
                     'dec_state': dec_state,
                     'cv': cv,
                     'aw_t_list': [None],
                     'rnnlm_state': None,
                     'prev_cov': 0,
                     '_dec_out': _dec_out,
                     '_dec_state': _dec_state}]
            for t in six.moves.range(int(math.floor(x_lens[b] * params['max_len_ratio'])) + 1):
                new_beam = []
                for i_beam in six.moves.range(len(beam)):
                    if t > 0:
                        # Recurrency
                        y = Variable(enc_out.data.new(
                            1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                        y_emb = self.emb(y)
                        dec_out, dec_state, _dec_out, _dec_state = self._recurrency(
                            y_emb, beam[i_beam]['cv'],
                            beam[i_beam]['dec_state'], beam[i_beam]['_dec_state'])
                    else:
                        dec_out = beam[i_beam]['dec_out']

                    # Score
                    cv, aw_t = self.score(enc_out[b:b + 1, :x_lens[b]],
                                          x_lens[b:b + 1],
                                          dec_out,
                                          beam[i_beam]['aw_t_list'][-1])

                    if self.rnnlm_cf is not None:
                        # Update RNNLM states for cold fusion
                        y_lm = Variable(enc_out.data.new(
                            1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                        y_lm_emb = self.rnnlm_cf.emb(y_lm)
                        logits_lm_t, lm_out, rnnlm_state = self.rnnlm_cf.predict(
                            y_lm_emb, beam[i_beam]['rnnlm_state'])
                    elif rnnlm is not None:
                        # Update RNNLM states for shallow fusion
                        y_lm = Variable(enc_out.data.new(
                            1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                        y_lm_emb = rnnlm.emb(y_lm)
                        logits_lm_t, lm_out, rnnlm_state = rnnlm.predict(
                            y_lm_emb, beam[i_beam]['rnnlm_state'])
                    else:
                        logits_lm_t, lm_out, rnnlm_state = None, None, None

                    # Generate
                    logits_t = self._generate(cv, dec_out, logits_lm_t, lm_out)

                    # residual connection
                    if self.rnnlm_init is not None and self.internal_lm:
                        if t == 0:
                            logits_t += beam[i_beam]['_dec_out']
                        else:
                            logits_t += _dec_out

                    if self.share_softmax or self.rnnlm_init is not None:
                        logits_t = self.output_bottle(logits_t)
                    logits_t = self.output(logits_t)
                    if self.rnnlm_cf is not None:
                        logits_t = F.relu(logits_t)

                    # Path through the softmax layer & convert to log-scale
                    log_probs = F.log_softmax(logits_t.squeeze(1), dim=1)
                    # log_probs = logits_t.squeeze(1)
                    # NOTE: `[1 (B), 1, num_classes]` -> `[1 (B), num_classes]`

                    # Pick up the top-k scores
                    log_probs_topk, indices_topk = torch.topk(
                        log_probs, k=params['beam_width'], dim=1, largest=True, sorted=True)

                    for k in six.moves.range(params['beam_width']):
                        # Exclude short hypotheses
                        if indices_topk[0, k].data[0] == self.eos and len(beam[i_beam]['hyp']) < x_lens[b] * params['min_len_ratio']:
                            continue

                        # Add length penalty
                        score = beam[i_beam]['score'] + log_probs_topk.data[0, k] + params['len_penalty']

                        # Add coverage penalty
                        if params['cov_penalty'] > 0:
                            # Recompute converage penalty in each step
                            score -= beam[i_beam]['prev_cov'] * params['cov_penalty']

                            aw_stack = torch.stack(beam[i_beam]['aw_t_list'][1:] + [aw_t], dim=1)

                            if self.score.num_heads > 1:
                                cov_sum = aw_stack.data[0, :, :, 0].cpu().numpy()
                                # TODO(hirofumi): fix for MHA
                            else:
                                cov_sum = aw_stack.data[0].cpu().numpy()
                            if params['cov_threshold'] == 0:
                                cov_sum = np.sum(cov_sum)
                            else:
                                cov_sum = np.sum(cov_sum[np.where(cov_sum > params['cov_threshold'])[0]])
                            score += cov_sum * params['cov_penalty']
                        else:
                            cov_sum = 0

                        # Add RNNLM score
                        if params['rnnlm_weight'] > 0:
                            lm_log_probs = F.log_softmax(logits_lm_t.squeeze(1), dim=1)
                            assert log_probs.size() == lm_log_probs.size()
                            score += lm_log_probs.data[0, indices_topk.data[0, k]] * params['rnnlm_weight']

                        new_beam.append(
                            {'hyp': beam[i_beam]['hyp'] + [indices_topk.data[0, k]],
                             'score': score,
                             'dec_state': copy.deepcopy(dec_state),
                             'dec_out': dec_out,
                             'cv': cv,
                             'aw_t_list': beam[i_beam]['aw_t_list'] + [aw_t],
                             'rnnlm_state': copy.deepcopy(rnnlm_state),
                             'prev_cov': cov_sum,
                             '_dec_out': _dec_out,
                             '_dec_state': _dec_state})

                new_beam = sorted(new_beam, key=lambda x: x['score'], reverse=True)

                # Remove complete hypotheses
                not_complete = []
                for cand in new_beam[:params['beam_width']]:
                    if cand['hyp'][-1] == self.eos:
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
            aw += [complete[0]['aw_t_list'][1:]]
            y_lens[b] = len(complete[0]['hyp'][1:])
            if complete[0]['hyp'][-1] == self.eos:
                eos_flags[b] = True

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

        # Exclude <eos>
        if exclude_eos:
            best_hyps = [best_hyps[b][:-1] if eos_flags[b]
                         else best_hyps[b] for b in six.moves.range(batch_size)]

        return best_hyps, aw

    def decode_ctc(self, enc_out, x_lens, beam_width=1, task_index=0):
        """Decoding by the CTC layer in the inference stage.

            This is only used for Joint CTC-Attention model.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, encoder_num_units]`
            beam_width (int): the size of beam
            task_index (int): the index of a task
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
                                             x_lens, beam_width=beam_width)

        return best_hyps


class _CTC(warpctc_pytorch._CTC):
    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens):
        is_cuda = True if acts.is_cuda else False
        acts = acts.contiguous()
        loss_func = warpctc_pytorch.gpu_ctc if is_cuda else warpctc_pytorch.cpu_ctc
        grads = torch.zeros(acts.size()).type_as(acts)
        minibatch_size = acts.size(1)
        costs = torch.zeros(minibatch_size).cpu()
        loss_func(acts,
                  grads,
                  labels,
                  label_lens,
                  act_lens,
                  minibatch_size,
                  costs)
        # modified only here from original
        costs = torch.FloatTensor([costs.sum()]) / acts.size(1)
        ctx.grads = Variable(grads)
        ctx.grads /= ctx.grads.size(1)

        return costs


def warpctc(acts, labels, act_lens, label_lens):
    """Chainer like CTC Loss

    acts: Tensor of (seqLength x batch x outputDim) containing output from network
    labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
    act_lens: Tensor of size (batch) containing size of each output sequence from the network
    act_lens: Tensor of (batch) containing label length of each example

    """
    assert len(labels.size()) == 1  # labels must be 1 dimensional
    _assert_no_grad(labels)
    _assert_no_grad(act_lens)
    _assert_no_grad(label_lens)
    return _CTC.apply(acts, labels, act_lens, label_lens)
