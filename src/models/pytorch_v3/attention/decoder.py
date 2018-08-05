#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""RNN decoder (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import numpy as np
import random
import six

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from src.models.pytorch_v3.attention.attention_layer import AttentionMechanism
from src.models.pytorch_v3.attention.attention_layer import MultiheadAttentionMechanism
from src.models.pytorch_v3.criterion import cross_entropy_lsm
from src.models.pytorch_v3.linear import Embedding
from src.models.pytorch_v3.linear import LinearND
from src.models.pytorch_v3.utils import pad_list
from src.models.pytorch_v3.utils import var2np


class Decoder(torch.nn.Module):
    """RNN decoder.

    Args:
        score_fn ():
        sos ():
        eos ():
        enc_n_units ():
        rnn_type (string): lstm or gru
        n_units (int): the number of units in each layer
        n_layers (int): the number of layers
        emb_dim ():
        bottle_dim ():
        generate_feat ():
        n_classes ():
        dropout_dec (float): the probability to drop nodes in the decoder
        dropout_emb (float):
        residual ():
        backward ():
        rnnlm_cf ():
        cf_type ():
        rnnlm_init ():
        lm_weight ():
        internal_lm ():
        share_softmax ():

    """

    def __init__(self,
                 score_fn,
                 sos,
                 eos,
                 enc_n_units,
                 rnn_type,
                 n_units,
                 n_layers,
                 residual,
                 emb_dim,
                 bottle_dim,
                 generate_feat,
                 n_classes,
                 logits_temp,
                 dropout_dec,
                 dropout_emb,
                 ss_prob,
                 lsm_prob,
                 lsm_type,
                 backward,
                 rnnlm_cf,
                 cf_type,
                 internal_lm,
                 rnnlm_init,
                 lm_weight,
                 share_softmax):

        super(Decoder, self).__init__()

        self.score = score_fn
        assert isinstance(score_fn, AttentionMechanism) or isinstance(score_fn, MultiheadAttentionMechanism)
        self.sos = sos
        self.eos = eos
        self.rnn_type = rnn_type
        self.n_units = n_units
        self.n_layers = n_layers
        self.residual = residual
        self.bottle_dim = bottle_dim
        self.generate_feat = generate_feat
        self.logits_temp = logits_temp
        self.dropout_dec = dropout_dec
        self.dropout_emb = dropout_emb
        self.ss_prob = ss_prob
        self.lsm_prob = lsm_prob
        self.lsm_type = lsm_type
        self.backward = backward
        self.rnnlm_cf = rnnlm_cf
        self.cf_type = cf_type
        self.internal_lm = internal_lm
        self.rnnlm_init = rnnlm_init
        self.lm_weight = lm_weight
        self.share_softmax = share_softmax
        self.pad_index = -1024

        # for decoder initialization with pre-trained RNNLM
        if rnnlm_init is not None:
            assert internal_lm
            assert rnnlm_init.predictor.n_classes == n_classes
            assert rnnlm_init.predictor.n_units == n_units
            assert rnnlm_init.predictor.n_layers == 1  # on-the-fly

        # for MTL with RNNLM objective
        if lm_weight > 0:
            assert internal_lm
            if not share_softmax:
                self.rnnlm_output = LinearND(n_units, n_classes)

        # Decoder
        if internal_lm:
            if rnn_type == 'lstm':
                self.lstm_lm = torch.nn.LSTMCell(emb_dim, n_units)
                self.lstm_l0 = torch.nn.LSTMCell(enc_n_units + n_units, n_units)
            elif rnn_type == 'gru':
                self.gru_lm = torch.nn.GRUCell(emb_dim, n_units)
                self.gru_l0 = torch.nn.GRUCell(enc_n_units + n_units, n_units)
            else:
                raise ValueError('rnn_type must be "lstm" or "gru".')
            self.dropout_lm = torch.nn.Dropout(p=dropout_dec)
        else:
            if rnn_type == 'lstm':
                self.lstm_l0 = torch.nn.LSTMCell(emb_dim + enc_n_units, n_units)
            elif rnn_type == 'gru':
                self.gru_l0 = torch.nn.GRUCell(emb_dim + enc_n_units, n_units)
            else:
                raise ValueError('rnn_type must be "lstm" or "gru".')
        self.dropout_l0 = torch.nn.Dropout(p=dropout_dec)

        for i_l in six.moves.range(1, n_layers):
            if rnn_type == 'lstm':
                rnn_i = torch.nn.LSTMCell(n_units, n_units)
            elif rnn_type == 'gru':
                rnn_i = torch.nn.GRUCell(n_units, n_units)
            else:
                raise ValueError('rnn_type must be "lstm" or "gru".')
            setattr(self, rnn_type + '_l' + str(i_l), rnn_i)
            setattr(self, 'dropout_l' + str(i_l), torch.nn.Dropout(p=dropout_dec))

        # cold fusion
        if rnnlm_cf is not None:
            assert cf_type in ['hidden', 'prob']
            if generate_feat == 'sc':
                self.cf_fc_dec_feat = LinearND(n_units + enc_n_units, n_units)
            if cf_type == 'hidden':
                self.cf_fc_lm_feat = LinearND(rnnlm_cf.n_units, n_units)
            elif cf_type == 'prob':
                # probability projection
                self.cf_fc_lm_feat = LinearND(rnnlm_cf.n_classes, n_units)
            self.cf_fc_lm_gate = LinearND(n_units * 2, n_units)
            self.fc_bottle = LinearND(n_units * 2, n_units)
            self.output = LinearND(n_units, n_classes)

            # fix RNNLM parameters
            for name, param in self.rnnlm_cf.named_parameters():
                param.requires_grad = False
        else:
            if generate_feat == 'sc' and (share_softmax or rnnlm_init is not None):
                self.fc_bottle = LinearND(n_units + enc_n_units, n_units)
                self.output = LinearND(n_units, n_classes)
            elif generate_feat == 'sc':
                self.output = LinearND(n_units + enc_n_units, n_classes)
            else:
                self.output = LinearND(n_units, n_classes)

        # Embedding
        self.emb = Embedding(n_classes=n_classes,
                             emb_dim=emb_dim,
                             dropout=dropout_emb,
                             ignore_index=eos)

    def _init_dec_state(self, enc_out, n_layers):
        """Initialize decoder state.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, enc_n_units]`
        Returns:
            dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, dec_n_units]`
            dec_state (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):

        """
        batch = enc_out.size(0)
        dec_out = Variable(enc_out.data.new(
            batch, 1, self.n_units).fill_(0.), volatile=not self.training)
        zero_state = Variable(enc_out.data.new(
            batch, self.n_units).fill_(0.), volatile=not self.training)
        hx_list = [zero_state] * self.n_layers
        cx_list = [zero_state] * self.n_layers if self.rnn_type == 'lstm' else None
        return dec_out, (hx_list, cx_list)

    def forward(self, enc_out, x_lens, ys):
        """Decoding in the training stage.　Compute XE loss.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, enc_n_units]`
            x_lens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains Variables of size `[L]`
        Returns:
            logits (torch.autograd.Variable, float): A tensor of size
                `[B, L, n_classes]`
            aw (torch.autograd.Variable, float): A tensor of size
                `[B, L, T, n_heads]`
            logits_lm (torch.autograd.Variable, float): A tensor of size
                `[B, L, n_classes]`

        """
        sos = Variable(enc_out.data.new(1,).fill_(self.sos).long())
        eos = Variable(enc_out.data.new(1,).fill_(self.eos).long())

        # Append <SOS> and <EOS>
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # Convert list to Variable
        ys_in_pad = pad_list(ys_in, self.eos)  # pad with <EOS>
        ys_out_pad = pad_list(ys_out, self.pad_index)

        batch, enc_time, enc_n_units = enc_out.size()

        # Initialization
        dec_out, dec_state = self._init_dec_state(enc_out, self.n_layers)
        internal_lm_out, internal_lm_state = self._init_dec_state(enc_out, 1)
        self.score.reset()
        aw_t = None
        lm_state = None

        # Pre-computation of embedding
        ys_emb = self.emb(ys_in_pad)
        if self.rnnlm_cf is not None:
            ys_lm_emb = [self.rnnlm_cf.emb(ys_in_pad[:, t:t + 1]) for t in six.moves.range(ys_in_pad.size(1))]
            ys_lm_emb = torch.cat(ys_lm_emb, dim=1)

        logits, logits_lm = [], []
        for t in six.moves.range(ys_in_pad.size(1)):
            is_sample = self.ss_prob > 0 and t > 0 and random.random() < self.ss_prob

            # Update RNNLM states for cold fusion
            if self.rnnlm_cf is not None:
                if is_sample:
                    y_lm_emb = self.rnnlm_cf.emb(torch.max(logits[-1], dim=2)[1]).detach()
                else:
                    y_lm_emb = ys_lm_emb[:, t:t + 1]
                logits_t_lm, lm_out, lm_state = self.rnnlm_cf.predict(y_lm_emb, lm_state)
            else:
                logits_t_lm, lm_out = None, None

            # Score
            cv, aw_t = self.score(enc_out, x_lens, dec_out, aw_t)

            # Generate
            logits_t = self.generate(cv, dec_out, logits_t_lm, lm_out)

            # residual connection
            if self.rnnlm_init is not None and self.internal_lm:
                logits_t += internal_lm_out

            logits_t = self.output(logits_t)
            if self.rnnlm_cf is not None:
                logits_t = F.relu(logits_t)
            logits += [logits_t]

            if t == ys_in_pad.size(1) - 1:
                break

            # Sample for scheduled sampling
            if is_sample:
                y_emb = self.emb(torch.max(logits[-1], dim=2)[1]).detach()
            else:
                y_emb = ys_emb[:, t:t + 1]

            # Recurrency
            dec_out, dec_state, internal_lm_out, internal_lm_state = self.recurrency(
                y_emb, cv, dec_state, internal_lm_state)
            if self.lm_weight > 0:
                if self.share_softmax:
                    logits_lm_t = self.output(internal_lm_out)
                else:
                    logits_lm_t = self.rnnlm_lo(internal_lm_out)
                logits_lm.append(logits_lm_t)

        logits = torch.cat(logits, dim=1)

        # Output smoothing
        if self.logits_temp != 1:
            logits /= self.logits_temp

        # Compute XE sequence loss
        if self.lsm_prob > 0:
            # Label smoothing
            y_lens = [y.size(0) for y in ys_out]
            loss = cross_entropy_lsm(
                logits, ys=ys_out_pad, y_lens=y_lens,
                lsm_prob=self.lsm_prob, lsm_type=self.lsm_type,
                size_average=True)
        else:
            loss = F.cross_entropy(
                input=logits.view((-1, logits.size(2))),
                target=ys_out_pad.view(-1),  # long
                ignore_index=self.pad_index, size_average=False) / len(enc_out)

        # Compute XE loss for RNNLM objective
        if self.lm_weight > 0:
            logits_lm = torch.cat(logits_lm, dim=1)
            loss_lm = F.cross_entropy(
                input=logits_lm.view((-1, logits_lm.size(2))),
                target=ys_out_pad.contiguous().view(-1),
                ignore_index=self.pad_index, size_average=True)
            loss += loss_lm * self.lm_weight

        # Compute token-level accuracy in teacher-forcing
        pad_pred = logits.data.view(ys_out_pad.size(
            0), ys_out_pad.size(1), logits.size(-1)).max(2)[1]
        mask = ys_out_pad.data != self.pad_index
        numerator = torch.sum(pad_pred.masked_select(
            mask) == ys_out_pad.data.masked_select(mask))
        denominator = torch.sum(mask)
        acc = float(numerator) / float(denominator)

        return loss, acc

    def recurrency(self, y_emb, cv, dec_state, internal_lm_state):
        """Recurrency function.

        Args:
            y_emb (torch.autograd.Variable, float): A tensor of size
                `[B, 1, emb_dim]`
            cv (torch.autograd.Variable, float): A tensor of size
                `[B, 1, enc_n_units]`
            dec_state (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):
            internal_lm_state (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):
        Returns:
            dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, n_units]`
            dec_state (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):
            hx_lm (torch.autograd.Variable, float): A tensor of size
                `[B, 1, n_units]`
            internal_lm_state (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):

        """
        hx_list, cx_list = dec_state
        hx_lm, cx_lm = internal_lm_state
        y_emb = y_emb.squeeze(1)
        cv = cv.squeeze(1)

        if self.internal_lm:
            if self.rnn_type == 'lstm':
                hx_lm, cx_lm = self.lstm_lm(y_emb, (hx_lm, cx_lm))
                hx_lm = self.dropout_lm(hx_lm)
                _h_lm = torch.cat([cv, hx_lm], dim=-1)
                hx_list[0], cx_list[0] = self.lstm_l0(_h_lm, (hx_list[0], cx_list[0]))
            elif self.rnn_type == 'gru':
                hx_lm = self.gru_lm(y_emb, hx_lm)
                hx_lm = self.dropout_lm(hx_lm)
                _h_lm = torch.cat([cv, hx_lm], dim=-1)
                hx_list[0] = self.gru_l0(_h_lm, hx_list[0])
        else:
            if self.rnn_type == 'lstm':
                hx_list[0], cx_list[0] = self.lstm_l0(torch.cat([y_emb, cv], dim=-1), (hx_list[0], cx_list[0]))
            elif self.rnn_type == 'gru':
                hx_list[0] = self.gru_l0(torch.cat([y_emb, cv], dim=-1), hx_list[0])
        hx_list[0] = self.dropout_l0(hx_list[0])

        for i_l in six.moves.range(1, self.n_layers):
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
        return dec_out, (hx_list, cx_list), hx_lm, (hx_lm, cx_lm)

    def generate(self, cv, dec_out, logits_t_lm, lm_out):
        """Generate function.

        Args:
            cv (torch.autograd.Variable, float): A tensor of size
                `[B, 1, enc_n_units]`
            dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, dec_n_units]`
            logits_t_lm (torch.autograd.Variable, float): A tensor of size
                `[B, 1, n_classes]`
            lm_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, lm_n_units]`
        Returns:
            logits_t (torch.autograd.Variable, float): A tensor of size
                `[B, 1, n_classes]`

        """
        if self.rnnlm_cf is not None:
            # cold fusion
            if self.cf_type == 'prob':
                lm_feat = self.cf_fc_lm_feat(logits_t_lm)
            else:
                lm_feat = self.cf_fc_lm_feat(lm_out)
            if self.generate_feat == 's':
                dec_feat = dec_out
            elif self.generate_feat == 'sc':
                dec_feat = self.cf_fc_dec_feat(torch.cat([dec_out, cv], dim=-1))
            gate = F.sigmoid(self.cf_fc_lm_gate(torch.cat([dec_feat, lm_feat], dim=-1)))
            gated_lm_feat = gate * lm_feat
            logits_t = self.fc_bottle(torch.cat([dec_feat, gated_lm_feat], dim=-1))
        else:
            if self.generate_feat == 's':
                logits_t = dec_out
            elif self.generate_feat == 'sc':
                logits_t = torch.cat([dec_out, cv], dim=-1)

        return logits_t

    def greedy(self, enc_out, x_lens, max_len_ratio, exclude_eos):
        """Greedy decoding in the inference stage.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, encoder_n_units]`
            x_lens (list): A list of length `[B]`
            max_len_ratio (int): the maximum sequence length of tokens
            exclude_eos (bool):
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`

        """
        batch, enc_time, enc_n_units = enc_out.size()

        # Initialization
        dec_out, dec_state = self._init_dec_state(enc_out, self.n_layers)
        internal_lm_out, internal_lm_state = self._init_dec_state(enc_out, 1)
        self.score.reset()
        aw_t = None
        lm_state = None

        # Start from <SOS>
        y = Variable(enc_out.data.new(batch, 1).fill_(self.sos).long(), volatile=True)

        _best_hyps, _aw = [], []
        y_lens = np.zeros((batch,), dtype=np.int32)
        eos_flags = [False] * batch
        for t in six.moves.range(math.floor(enc_time * max_len_ratio) + 1):
            if self.rnnlm_cf is not None:
                # Update RNNLM states for cold fusion
                y_lm = self.rnnlm_cf.emb(y)
                logits_t_lm, lm_out, lm_state = self.rnnlm_cf.predict(y_lm, lm_state)
            else:
                logits_t_lm, lm_out = None, None

            # Score
            cv, aw_t = self.score(enc_out, x_lens, dec_out, aw_t)

            # Generate
            logits_t = self.generate(cv, dec_out, logits_t_lm, lm_out)

            # residual connection
            if self.rnnlm_init is not None and self.internal_lm:
                logits_t += internal_lm_out

            logits_t = self.output(logits_t)
            if self.rnnlm_cf is not None:
                logits_t = F.relu(logits_t)

            # Pick up 1-best
            y = torch.max(logits_t.squeeze(1), dim=1)[1].unsqueeze(1)
            _best_hyps += [y]
            _aw += [aw_t]

            # Count lengths of hypotheses
            for b in six.moves.range(batch):
                if not eos_flags[b]:
                    if y.data.cpu().numpy()[b] == self.eos:
                        eos_flags[b] = True
                    y_lens[b] += 1
                    # NOTE: include <EOS>

            # Break if <EOS> is outputed in all mini-batch
            if sum(eos_flags) == batch:
                break

            # Recurrency
            y_emb = self.emb(y)
            dec_out, dec_state, internal_lm_out, internal_lm_state = self.recurrency(
                y_emb, cv, dec_state, internal_lm_state)

        # Concatenate in L dimension
        _best_hyps = torch.cat(_best_hyps, dim=1)
        _aw = torch.stack(_aw, dim=1)

        # Convert to numpy
        _best_hyps = var2np(_best_hyps)
        _aw = var2np(_aw)

        if self.score.n_heads > 1:
            _aw = _aw[:, :, :, 0]
            # TODO(hirofumi): fix for MHA

        # Truncate by the first <EOS>
        if self.backward:
            # Reverse the order
            best_hyps = [_best_hyps[b, :y_lens[b]][::-1] for b in six.moves.range(batch)]
            aw = [_aw[b, :y_lens[b]][::-1] for b in six.moves.range(batch)]
        else:
            best_hyps = [_best_hyps[b, :y_lens[b]] for b in six.moves.range(batch)]
            aw = [_aw[b, :y_lens[b]] for b in six.moves.range(batch)]

        # Exclude <EOS>
        if exclude_eos:
            best_hyps = [best_hyps[b][:-1] if eos_flags[b]
                         else best_hyps[b] for b in six.moves.range(batch)]

        return best_hyps, aw

    def beam_search(self, enc_out, x_lens, beam_width,
                    min_len_ratio, max_len_ratio,
                    len_penalty, cov_penalty, cov_threshold,
                    rnnlm, lm_weight, exclude_eos):
        """Beam search decoding in the inference stage.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, enc_n_units]`
            x_lens (list): A list of length `[B]`
            beam_width (int): the size of beam
            max_len_ratio (int): the maximum sequence length of tokens
            min_len_ratio (float): the minimum sequence length of tokens
            len_penalty (float): length penalty
            cov_penalty (float): coverage penalty
            cov_threshold (float): threshold for coverage penalty
            rnnlm ():
            lm_weight (float): the weight of RNNLM score
            exclude_eos (bool):
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`

        """
        batch_size = enc_out.size(0)

        # For shallow fusion
        if lm_weight > 0 and not self.cf_type:
            assert self.rnnlm is not None
            assert not self.rnnlm.training

        # For cold fusion
        if self.rnnlm_cf is not None:
            assert self.rnnlm is not None
            assert not self.rnnlm.training

        best_hyps, aw = [], []
        y_lens = np.zeros((batch_size,), dtype=np.int32)
        eos_flags = [False] * batch_size
        for b in six.moves.range(batch_size):
            # Initialization per utterance
            dec_out, dec_state = self._init_dec_state(enc_out[b:b + 1], self.n_layers)
            internal_lm_out, internal_lm_state = self._init_dec_state(enc_out[b:b + 1], 1)
            cv = Variable(enc_out.data.new(1, 1, enc_out.size(-1)).fill_(0.), volatile=True)
            self.score.reset()

            complete = []
            beam = [{'hyp': [self.sos],
                     'score': 0,  # log 1
                     'dec_out': dec_out,
                     'dec_state': dec_state,
                     'cv': cv,
                     'aw_t_list': [None],
                     'lm_state': None,
                     'prev_cov': 0,
                     'internal_lm_out': internal_lm_out,
                     'internal_lm_state': internal_lm_state}]
            for t in six.moves.range(math.floor(x_lens[b] * max_len_ratio) + 1):
                new_beam = []
                for i_beam in six.moves.range(len(beam)):
                    if self.rnnlm_cf is not None:
                        # Update RNNLM states for cold fusion
                        y_lm = Variable(enc_out.data.new(
                            1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                        y_lm_emb = self.rnnlm_cf.emb(y_lm)
                        logits_t_lm, lm_out, lm_state = self.rnnlm_cf.predict(
                            y_lm_emb, beam[i_beam]['lm_state'])
                    elif self.rnnlm is not None:
                        # Update RNNLM states for shallow fusion
                        y_lm = Variable(enc_out.data.new(
                            1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                        y_lm_emb = self.rnnlm.emb(y_lm)
                        logits_t_lm, lm_out, lm_state = self.rnnlm.predict(
                            y_lm_emb, beam[i_beam]['lm_state'])
                    else:
                        logits_t_lm, lm_out = None, None

                    if t == 0:
                        dec_out = beam[i_beam]['dec_out']
                    else:
                        y = Variable(enc_out.data.new(
                            1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                        y_emb = self.emb(y)

                        # Recurrency
                        dec_out, dec_state, internal_lm_out, internal_lm_state = self.recurrency(
                            y_emb, beam[i_beam]['cv'], beam[i_beam]['dec_state'], beam[i_beam]['internal_lm_state'])

                    # Score
                    cv, aw_t = self.score(enc_out[b:b + 1, :x_lens[b]],
                                          x_lens[b:b + 1],
                                          dec_out,
                                          beam[i_beam]['aw_t_list'][-1])

                    # Generate
                    logits_t = self.generate(cv, dec_out, logits_t_lm, lm_out)

                    # residual connection
                    if self.rnnlm_init is not None and self.internal_lm:
                        if t == 0:
                            logits_t += beam[i_beam]['internal_lm_out']
                        else:
                            logits_t += internal_lm_out

                    logits_t = self.output(logits_t)
                    if self.rnnlm_cf is not None:
                        logits_t = F.relu(logits_t)

                    # Path through the softmax layer & convert to log-scale
                    log_probs = F.log_softmax(logits_t.squeeze(1), dim=1)
                    # log_probs = logits_t.squeeze(1)
                    # NOTE: `[1 (B), 1, n_classes]` -> `[1 (B), n_classes]`

                    # Pick up the top-k scores
                    log_probs_topk, indices_topk = torch.topk(
                        log_probs, k=beam_width, dim=1, largest=True, sorted=True)

                    for k in six.moves.range(beam_width):
                        # Exclude short hypotheses
                        if indices_topk[0, k].data[0] == self.eos and len(beam[i_beam]['hyp']) < x_lens[b] * min_len_ratio:
                            continue

                        # Add length penalty
                        score = beam[i_beam]['score'] + log_probs_topk.data[0, k] + len_penalty

                        # Add coverage penalty
                        if cov_penalty > 0:
                            # Recompute converage penalty in each step
                            score -= beam[i_beam]['prev_cov'] * cov_penalty

                            aw_stack = torch.stack(beam[i_beam]['aw_t_list'][1:] + [aw_t], dim=1)

                            if self.score.n_heads > 1:
                                cov_sum = aw_stack.data[0, :, :, 0].cpu().numpy()
                                # TODO(hirofumi): fix for MHA
                            else:
                                cov_sum = aw_stack.data[0].cpu().numpy()
                            if cov_threshold == 0:
                                cov_sum = np.sum(cov_sum)
                            else:
                                cov_sum = np.sum(cov_sum[np.where(cov_sum > cov_threshold)[0]])
                            score += cov_sum * cov_penalty
                        else:
                            cov_sum = 0

                        # Add RNNLM score
                        if lm_weight > 0:
                            lm_log_probs = F.log_softmax(logits_t_lm.squeeze(1), dim=1)
                            assert log_probs.size() == lm_log_probs.size()
                            score += lm_log_probs.data[0, indices_topk.data[0, k]] * lm_weight
                        elif not self.cf_type:
                            lm_state = None

                        new_beam.append(
                            {'hyp': beam[i_beam]['hyp'] + [indices_topk.data[0, k]],
                             'score': score,
                             'dec_state': copy.deepcopy(dec_state),
                             'dec_out': dec_out,
                             'cv': cv,
                             'aw_t_list': beam[i_beam]['aw_t_list'] + [aw_t],
                             'lm_state': copy.deepcopy(lm_state),
                             'prev_cov': cov_sum})

                new_beam = sorted(new_beam, key=lambda x: x['score'], reverse=True)

                # Remove complete hypotheses
                not_complete = []
                for cand in new_beam[:beam_width]:
                    if cand['hyp'][-1] == self.eos:
                        complete += [cand]
                    else:
                        not_complete += [cand]

                if len(complete) >= beam_width:
                    complete = complete[:beam_width]
                    break

                beam = not_complete[:beam_width]

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
            if self.score.n_heads > 1:
                aw[b] = aw[b][:, :, 0]
                # TODO(hirofumi): fix for MHA

        # Reverse the order
        if self.backward:
            best_hyps = [best_hyps[b][::-1] for b in six.moves.range(batch_size)]
            aw = [aw[b][::-1] for b in six.moves.range(batch_size)]

        # Exclude <EOS>
        if exclude_eos:
            best_hyps = [best_hyps[b][:-1] if eos_flags[b]
                         else best_hyps[b] for b in six.moves.range(batch_size)]

        return best_hyps, aw
