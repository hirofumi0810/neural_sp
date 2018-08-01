#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""RNN decoder (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import random
import six

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from src.models.pytorch_v3.attention.attention_layer import AttentionMechanism
from src.models.pytorch_v3.attention.attention_layer import MultiheadAttentionMechanism
from src.models.pytorch_v3.linear import Embedding
from src.models.pytorch_v3.linear import LinearND
from src.models.pytorch_v3.utils import var2np


class Decoder(torch.nn.Module):
    """RNN decoder.

    Args:
        score_fn ():
        dec_in_size (int): the dimension of decoder inputs
        sos
        eos
        enc_n_units ():
        dec_in_size ():
        rnn_type (string): lstm or gru
        n_units (int): the number of units in each layer
        n_layers (int): the number of layers
        emb_dim ():
        bottle_dim ():
        generate_feat ():
        n_classes ():
        dropout (float):
        dropout_dec (float): the probability to drop nodes in the decoder
        dropout_emb (float):
        residual ():
        cov_weight ():
        backward ():
        lm_fusion ():
        rnnlm ():

    """

    def __init__(self,
                 score_fn,
                 sos,
                 eos,
                 enc_n_units,
                 rnn_type,
                 n_units,
                 n_layers,
                 emb_dim,
                 bottle_dim,
                 generate_feat,
                 n_classes,
                 dropout_dec,
                 dropout_emb,
                 residual,
                 cov_weight,
                 backward,
                 lm_fusion,
                 rnnlm):

        super(Decoder, self).__init__()

        self.score = score_fn
        assert isinstance(score_fn, AttentionMechanism) or isinstance(score_fn, MultiheadAttentionMechanism)
        self.sos = sos
        self.eos = eos
        self.rnn_type = rnn_type
        self.n_units = n_units
        self.n_layers = n_layers
        self.bottle_dim = bottle_dim
        self.generate_feat = generate_feat
        self.dropout_dec = dropout_dec
        self.dropout_emb = dropout_emb
        self.residual = residual
        self.cov_weight = cov_weight
        self.backward = backward
        self.lm_fusion = lm_fusion
        self.rnnlm = rnnlm

        # Decoder
        for i_l in six.moves.range(n_layers):
            dec_in_size = enc_n_units + emb_dim if i_l == 0 else n_units
            if rnn_type == 'lstm':
                rnn_i = torch.nn.LSTMCell(dec_in_size=dec_in_size,
                                          hidden_size=n_units, bias=True)
            elif rnn_type == 'gru':
                rnn_i = torch.nn.GRUCell(dec_in_size=dec_in_size,
                                         hidden_size=n_units, bias=True)
            else:
                raise ValueError('rnn_type must be "lstm" or "gru".')
            setattr(self, rnn_type + '_l' + str(i_l), rnn_i)

            # Dropout for hidden-hidden or hidden-output connection
            setattr(self, 'drop_l' + str(i_l), torch.nn.Dropout(p=dropout_dec))

        # RNNLM integration
        if 'cold_fusion' in lm_fusion:
            if lm_fusion == 'cold_fusion_prob':
                self.fc_cf_lm_logits = LinearND(rnnlm.n_classes, rnnlm.n_units)
            self.fc_cf_gate = LinearND(n_units + rnnlm.n_units, rnnlm.n_units)
            if bottle_dim == 0:
                self.fc_cf_gated_lm = LinearND(rnnlm.n_units, self.n_classes)
            else:
                self.fc_cf_gated_lm = LinearND(rnnlm.n_units, bottle_dim)

        # Output layer
        if bottle_dim > 0:
            self.fc_dec = LinearND(n_units, bottle_dim)
            if 'c' in generate_feat:
                self.fc_cv = LinearND(enc_n_units, bottle_dim)
            self.fc_bottle = LinearND(bottle_dim, n_classes)
        else:
            self.fc_dec = LinearND(n_units, n_classes)
            if 'c' in generate_feat:
                self.fc_cv = LinearND(enc_n_units, n_classes)

        # Embedding
        self.emb = Embedding(n_classes=n_classes,
                             emb_dim=emb_dim,
                             dropout=dropout_emb,
                             ignore_index=eos)

    def _init_dec_state(self, enc_out, x_lens):
        """Initialize decoder state.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, enc_n_units]`
            x_lens (list): A list of length `[B]`
        Returns:
            dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, decoder_num_units]`
            dec_state (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):

        """
        batch = enc_out.size(0)
        zero_state = Variable(enc_out.data.new(
            batch, self.n_units).fill_(0.), volatile=not self.training)

        dec_out = Variable(enc_out.data.new(
            batch, 1, self.n_units).fill_(0.), volatile=not self.training)

        hx_list = [zero_state] * self.n_layers
        cx_list = [zero_state] * self.n_layers if self.rnn_type == 'lstm' else None

        return dec_out, (hx_list, cx_list)

    def forward(self, enc_out, x_lens, ys):
        """Decoding in the training stage.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, enc_n_units]`
            x_lens (list): A list of length `[B]`
            ys (torch.autograd.Variable, long): A tensor of size `[B, L]`,
                which should be padded with <EOS>.
        Returns:
            logits (torch.autograd.Variable, float): A tensor of size
                `[B, L, n_classes]`
            aw (torch.autograd.Variable, float): A tensor of size
                `[B, L, T, n_heads]`
            logits_lm (torch.autograd.Variable, float): A tensor of size
                `[B, L, n_classes]`

        """
        batch, enc_time, enc_n_units = enc_out.size()

        # Initialization
        dec_out, dec_state = self._init_dec_state(enc_out, x_lens)
        self.score.reset()
        aw_t = None
        lm_state = None

        # Set RNNLM to the evaluation mode in cold fusion
        # if self.lm_fusion and self.lm_weight == 0:
        #     self.rnnlm.eval()
        # TODO(hirofumi): move this

        # Pre-computation of embedding
        ys_emb = self.emb(ys)
        if self.lm_fusion:
            ys_lm_emb = [self.rnnlm.emb(ys[:, t:t + 1]) for t in six.moves.range(ys.size(1))]
            ys_lm_emb = torch.cat(ys_lm_emb, dim=1)

        logits, aw, logits_lm = [], [], []
        for t in six.moves.range(ys.size(1)):
            # Sample for scheduled sampling
            is_sample = self.ss_prob > 0 and t > 0 and random.random() < self.ss_prob
            if is_sample:
                y_emb = self.emb(torch.max(logits[-1], dim=2)[1]).detach()
            else:
                y_emb = ys_emb[:, t:t + 1]

            # Update RNNLM states
            if self.lm_fusion:
                if is_sample:
                    y_lm_emb = self.rnnlm.emb(torch.max(logits[-1], dim=2)[1]).detach()
                else:
                    y_lm_emb = ys_lm_emb[:, t:t + 1]
                logits_t_lm, lm_out, lm_state = self.rnnlm.predict(y_lm_emb, lm_state)
                logits_lm += [logits_t_lm]

            # Score
            cv, aw_t = self.score(enc_out, x_lens, dec_out, aw_t)

            # Generate
            logits_t = self.fc_dec(dec_out)
            if 'c' in self.generate_feat:
                logits_t += self.fc_cv(cv)

            # RNNLM fusion
            if 'cold_fusion' in self.lm_fusion:
                if self.lm_fusion == 'cold_fusion_prob':
                    # Probability projection
                    lm_feat = self.fc_cf_lm_logits(logits_t_lm)
                else:
                    lm_feat = lm_out
                # Fine-grained gating
                gate = F.sigmoid(self.fc_cf_gate(torch.cat([dec_out, lm_feat], dim=-1)))
                logits_t += self.fc_cf_gated_lm(torch.mul(gate, lm_out))
            elif self.lm_fusion == 'logits_fusion':
                logits_t += logits_t_lm

            if self.bottle_dim > 0:
                logits_t = self.fc_bottle(F.tanh(logits_t))

            if self.lm_fusion == 'cold_fusion_prob':
                logits_t = F.relu(logits_t)

            logits += [logits_t]
            if self.cov_weight > 0:
                aw += [aw_t]

            if t == ys.size(1) - 1:
                break

            # Recurrency
            dec_in = torch.cat([y_emb, cv], dim=-1)
            dec_out, dec_state = self.recurrency(dec_in, dec_state)

        logits = torch.cat(logits, dim=1)
        if self.cov_weight > 0:
            aw = torch.stack(aw, dim=1)
        if self.lm_fusion:
            logits_lm = torch.cat(logits_lm, dim=1)

        return logits, aw, logits_lm

    def recurrency(self, dec_in, dec_state):
        """Recurrency function.

        Args:
            dec_in (torch.autograd.Variable, float): A tensor of size
                `[B, 1, emb_dim + enc_n_units]`
            dec_state (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):
        Returns:
            dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, n_units]`
            dec_state (tuple): A tuple of (hx_list, cx_list)
                hx_list (list of torch.autograd.Variable(float)):
                cx_list (list of torch.autograd.Variable(float)):

        """
        hx_list, cx_list = dec_state

        dec_in = dec_in.squeeze(1)
        # NOTE: exclude residual connection from decoder's inputs
        for i_l in six.moves.range(self.n_layers):
            if self.rnn_type == 'lstm':
                if i_l == 0:
                    hx_list[0], cx_list[0] = getattr(self, 'lstm_l0')(
                        dec_in, (hx_list[0], cx_list[0]))
                else:
                    hx_list[i_l], cx_list[i_l] = getattr(self, 'lstm_l' + str(i_l))(
                        hx_list[i_l - 1], (hx_list[i_l], cx_list[i_l]))
            elif self.rnn_type == 'gru':
                if i_l == 0:
                    hx_list[0] = getattr(self, 'gru_l0')(dec_in, hx_list[0])
                else:
                    hx_list[i_l] = getattr(self, 'gru_l' + str(i_l))(
                        hx_list[i_l - 1], hx_list[i_l])

            # Dropout for hidden-hidden or hidden-output connection
            hx_list[i_l] = getattr(self, 'drop_l' + str(i_l))(hx_list[i_l])

            # Residual connection
            if i_l > 0 and self.residual:
                hx_list[i_l] += sum(hx_list[i_l - 1])

        dec_out = hx_list[-1].unsqueeze(1)
        return dec_out, (hx_list, cx_list)

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
        dec_out, dec_state = self._init_dec_state(enc_out, x_lens)
        self.score.reset()
        aw_t = None
        lm_state = None

        # Set RNNLM to the evaluation mode in cold fusion
        # if self.lm_fusion and self.lm_weight == 0:
        #     self.rnnlm.eval()
        # TODO(hirofumi): move this

        # Start from <SOS>
        y = Variable(enc_out.data.new(batch, 1).fill_(self.sos).long(), volatile=True)

        _best_hyps, _aw = [], []
        y_lens = np.zeros((batch,), dtype=np.int32)
        eos_flags = [False] * batch
        for t in six.moves.range(enc_time * max_len_ratio + 1):
            # Update RNNLM states
            if self.lm_fusion:
                y_lm = self.rnnlm.emb(y)
                logits_t_lm, lm_out, lm_state = self.rnnlm.predict(y_lm, lm_state)

            y_emb = self.emb(y)

            # Score
            cv, aw_t = self.score(enc_out, x_lens, dec_out, aw_t)

            # Generate
            logits_t = self.fc_dec(dec_out)
            if 'c' in self.generate_feat:
                logits_t += self.fc_cv(cv)

            # RNNLM fusion
            if 'cold_fusion' in self.lm_fusion:
                if self.lm_fusion == 'cold_fusion_prob':
                    # Probability projection
                    lm_feat = self.fc_cf_lm_logits(logits_t_lm)
                else:
                    lm_feat = lm_out
                # Fine-grained gating
                gate = F.sigmoid(self.fc_cf_gate(torch.cat([dec_out, lm_feat], dim=-1)))
                logits_t += self.fc_cf_gated_lm(torch.mul(gate, lm_out))
            elif self.lm_fusion == 'logits_fusion':
                logits_t += logits_t_lm

            if self.bottle_dim > 0:
                logits_t = self.fc_bottle(F.tanh(logits_t))

            if self.lm_fusion == 'cold_fusion_prob':
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
            dec_in = torch.cat([y_emb, cv], dim=-1)
            dec_out, dec_state = self.reccurency(dec_in, dec_state)

        # Concatenate in L dimension
        _best_hyps = torch.cat(_best_hyps, dim=1)
        _aw = torch.stack(_aw, dim=1)

        # Convert to numpy
        _best_hyps = var2np(_best_hyps)
        _aw = var2np(_aw)

        if self.n_heads > 1:
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
                    lm_weight, exclude_eos):
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
            lm_weight (float): the weight of RNNLM score
            task (int): the index of a task
            dir (str): fwd or bwd
            exclude_eos (bool):
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`

        """
        batch_size = enc_out.size(0)

        # For shallow fusion
        if lm_weight > 0 and not self.lm_fusion:
            assert self.rnnlm is not None
            assert not self.rnnlm.training

        # For cold fusion
        if self.lm_fusion:
            assert self.rnnlm is not None
            assert not self.rnnlm.training

        best_hyps, aw = [], []
        y_lens = np.zeros((batch_size,), dtype=np.int32)
        eos_flags = [False] * batch_size
        for b in six.moves.range(batch_size):
            # Initialization per utterance
            dec_out, dec_state = self._init_dec_state(enc_out[b:b + 1], x_lens[b])
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
                     'prev_cov': 0}]
            for t in six.moves.range(x_lens[b] * max_len_ratio + 1):
                new_beam = []
                for i_beam in six.moves.range(len(beam)):
                    # Update RNNLM states
                    if lm_weight > 0 or self.lm_fusion:
                        y_rnnlm = Variable(enc_out.data.new(
                            1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                        y_rnnlm = self.rnnlm.embed(y_rnnlm)
                        logits_t_lm, lm_out, lm_state = self.rnnlm.predict(
                            y_rnnlm, beam[i_beam]['lm_state'])

                    y = Variable(enc_out.data.new(
                        1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                    y_emb = self.emb(y)

                    if t == 0:
                        dec_out = beam[i_beam]['dec_out']
                    else:
                        # Recurrency
                        dec_in = torch.cat([y_emb, beam[i_beam]['cv']], dim=-1)
                        dec_out, dec_state = self.reccurency(dec_in, beam[i_beam]['dec_state'])

                    # Score
                    cv, aw_t = self.score(enc_out[b:b + 1, :x_lens[b]],
                                          x_lens[b:b + 1],
                                          dec_out,
                                          beam[i_beam]['aw_t_list'][-1])

                    # Generate
                    logits_t = self.fc_dec(dec_out)
                    if 'c' in self.generate_feat:
                        logits_t += self.fc_cv(cv)

                    # RNNLM fusion
                    if 'cold_fusion' in self.lm_fusion:
                        if self.lm_fusion == 'cold_fusion_prob':
                            # Probability projection
                            lm_feat = self.fc_cf_lm_logits(logits_t_lm)
                        else:
                            lm_feat = lm_out
                        # Fine-grained gating
                        gate = F.sigmoid(self.fc_cf_gate(torch.cat([dec_out, lm_feat], dim=-1)))
                        logits_t += self.fc_cf_gated_lm(torch.mul(gate, lm_out))
                    elif self.lm_fusion == 'logits_fusion':
                        logits_t += logits_t_lm

                    if self.bottle_dim > 0:
                        logits_t = self.fc_bottle(F.tanh(logits_t))

                    if self.lm_fusion == 'cold_fusion_prob':
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

                            if self.n_heads > 1:
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
                        elif not self.lm_fusion:
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
            if self.n_heads > 1:
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
