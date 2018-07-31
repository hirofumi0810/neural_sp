#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""RNN decoder (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable

# from src.models.pytorch_v3.attention.attention_layer import AttentionMechanism
# from src.models.pytorch_v3.attention.attention_layer import MultiheadAttentionMechanism


class Decoder(torch.nn.Module):
    """RNN decoder.

    Args:
        score_fn ():
        dec_in_size (int): the dimension of decoder inputs
        rnn_type (string): lstm or gru
        n_units (int): the number of units in each layer
        n_layers (int): the number of layers
        dropout (float): the probability to drop nodes
        residual (bool):
        dense_residual (bool):

    """

    def __init__(self,
                 score_fn,
                 sos,
                 eos,
                 dec_in_size,
                 rnn_type,
                 n_units,
                 n_layers,
                 dropout_dec,
                 dropout_emb,
                 residual,
                 dense_residual,
                 coverage_weight,
                 rnnlm_fusion):

        super(Decoder, self).__init__()

        self.score = score_fn
        assert isinstance(score_fn, AttentionMechanism) or isinstance(score_fn, MultiheadAttentionMechanism)
        self.sos = sos
        self.eos = eos
        self.dec_in_size = dec_in_size
        self.rnn_type = rnn_type
        self.n_units = n_units
        self.n_layers = n_layers
        self.dropout = dropout
        self.res = residual
        self.dense_res = dense_residual
        self.coverage_weight = coverage_weight
        self.rnnlm_fusion = rnnlm_fusion

        # Decoder
        for i_l in range(n_layers):
            dec_in_size = dec_in_size if i_l == 0 else n_units
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

        # Output layer
        if bottle_dim > 0:
            self.fc_dec = LinearND(dec_nunits, bottle_dim)
            if 'c' in generate_feature:
                self.fc_cv = LinearND(enc_n_units, bottle_dim)
            if 'y' in generate_feature:
                self.fc_emb = LinearND(emb_dim, bottle_dim)
            self.fc_bottle = LinearND(bottle_dim, n_classes)
        else:
            self.fc_dec = LinearND(dec_nunits, n_classes)
            if 'c' in generate_feature:
                self.fc_cv = LinearND(enc_n_units, n_classes)
            if 'y' in generate_feature:
                self.fc_emb = LinearND(enc_n_units, n_classes)
            # NOTE: turn off dropout

        # Embedding
        if input_type == 'speech':
            self.emb = Embedding(n_classes=self.n_classes,
                                 emb_dim=emb_dim,
                                 dropout=dropout_emb,
                                 ignore_index=self.eos)
        elif input_type == 'text':
            self.emb = Embedding(n_classes=self.nclass_input,
                                 emb_dim=dec_in_size,
                                 dropout=dropout_emb,
                                 ignore_index=self.nclass_input - 1)

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
        cx_list = [zero_state] * self.n_layers　if self.rnn_type == 'lstm' else None

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
        aw_step = None
        cv = Variable(enc_out.data.new(batch, 1, enc_n_units).fill_(0.))
        lm_state = None

        # Set RNNLM to the evaluation mode in cold fusion
        # if self.rnnlm_fusion and self.rnnlm_weight == 0:
        #     self.rnnlm.eval()
        # TODO(hirofumi): move this

        # Pre-computation of embedding
        ys_emb = self.emb(ys)
        if self.rnnlm_fusion:
            ys_lm_emb = [self.rnnlm.emb(ys[:, t:t + 1]) for t in range(ys.size(1))]
            ys_lm_emb = torch.cat(ys_lm_emb, dim=1)

        logits, aw, logits_lm = [], [], []
        for t in range(ys.size(1)):
            # Sample for scheduled sampling
            is_sample = self.ss_prob > 0 and t > 0 and random.random() < self.ss_prob
            if is_sample:
                y = self.emb(torch.max(logits[-1], dim=2)[1]).detach()
            else:
                y = ys_emb[:, t:t + 1]

            # Update RNNLM states
            if self.rnnlm_fusion:
                if is_sample:
                    y_lm = self.rnnlm.emb(torch.max(logits[-1], dim=2)[1]).detach()
                else:
                    y_lm = ys_lm_emb[:, t:t + 1]
                logits_step_lm, lm_out, lm_state = self.rnnlm.predict(y_lm, lm_state)
                logits_lm += [logits_step_lm]

            if t > 0:
                # Recurrency
                dec_in = torch.cat([y, cv], dim=-1)
                if self.rnnlm_fusion in ['embedding_fusion', 'state_embedding_fusion']:
                    dec_in = torch.cat([dec_in, y_lm], dim=-1)
                if self.rnnlm_fusion in ['state_fusion', 'state_embedding_fusion']:
                    dec_in = torch.cat([dec_in, lm_out], dim=-1)
                dec_out, dec_state = self.recurrency(dec_in, dec_state)

            # Score
            cv, aw_step = self.score(enc_out, x_lens, dec_out, aw_step)

            # Generate
            logits_step = self.fc_dec(dec_out)
            if 'c' in self.generate_feature:
                logits_step += self.fc_cv(cv)
            if 'y' in self.generate_feature:
                logits_step += self.fc_emb(y)

            # RNNLM fusion
            if self.rnnlm_fusion == 'cold_fusion_simple':
                # Fine-grained gating
                gate = F.sigmoid(self.fc_lm_gate(
                    torch.cat([dec_out, lm_out], dim=-1)))
                logits_step += self.fc_lm(torch.mul(gate, lm_out))
            elif self.rnnlm_fusion == 'cold_fusion':
                # Prob injection
                rnnlm_feat = self.fc_lm_logits(logits_step_lm)
                # Fine-grained gating
                gate = F.sigmoid(self.fc_lm_gate(
                    torch.cat([dec_out, rnnlm_feat], dim=-1)))
                logits_step += self.fc_lm(torch.mul(gate, lm_out))
                if self.bottle_dim == 0:
                    logits_step = F.relu(logits_step)
            elif self.rnnlm_fusion == 'logits_fusion':
                logits_step += logits_step_lm

            if self.bottle_dim > 0:
                logits_step = self.fc_bottle(F.tanh(logits_step))

            if self.rnnlm_fusion == 'cold_fusion':
                logits_step = F.relu(logits_step)

            logits += [logits_step]
            if self.coverage_weight > 0:
                aw += [aw_step]

        logits = torch.cat(logits, dim=1)
        if self.coverage_weight > 0:
            aw = torch.stack(aw, dim=1)
        if self.rnnlm_fusion:
            logits_lm = torch.cat(logits_lm, dim=1)

        return logits, aw, logits_lm

    def recurrency(self, dec_in, dec_state):
        """Recurrency.

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
        for i_l in range(self.n_layers):
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
            if i_l > 0 and self.res or self.dense_res:
                if self.res:
                    hx_list[i_l] += sum(hx_list[i_l - 1])
                elif self.dense_res:
                    hx_list[i_l] += sum(hx_list[:i_l])

        dec_out = hx_list[-1].unsqueeze(1)
        return dec_out, (hx_list, cx_list)

    def greedy(self, enc_out, x_lens, max_decode_len, task, dir, exclude_eos):
        """Greedy decoding in the inference stage.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, encoder_n_units]`
            x_lens (list): A list of length `[B]`
            max_decode_len (int): the maximum sequence length of tokens
            task (int): the index of a task
            dir (str): fwd or bwd
            exclude_eos (bool):
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`

        """
        if dir == 'bwd':
            assert getattr(self, 'bwd_weight_' + str(task)) > 0

        batch, enc_time, enc_n_units = enc_out.size()

        # Initialization
        dec_out, dec_state = self._init_dec_state(enc_out, x_lens)
        self.score.reset()
        aw_step = None
        lm_state = None

        # Start from <SOS>
        y = Variable(enc_out.data.new(
            batch, 1).fill_(sos).long(), volatile=True)

        _best_hyps, _aw = [], []
        y_lens = np.zeros((batch,), dtype=np.int32)
        eos_flags = [False] * batch
        for t in range(max_decode_len + 1):
            # Update RNNLM states
            if self.rnnlm_fusion:
                y_lm = self.rnnlm.emb(y)
                logits_step_rnnlm, lm_out, lm_state = self.rnnlm.predict(y_lm, lm_state)

            y = self.emb(y)

            if t > 0:
                # Recurrency
                dec_in = torch.cat([y, cv], dim=-1)
                if self.rnnlm_fusion in ['embedding_fusion', 'state_embedding_fusion']:
                    dec_in = torch.cat([dec_in, y_lm], dim=-1)
                if self.rnnlm_fusion in ['state_fusion', 'state_embedding_fusion']:
                    dec_in = torch.cat([dec_in, lm_out], dim=-1)
                dec_out, dec_state = getattr(self, 'decoder' + taskdir)(
                    dec_in, dec_state)

            # Score
            cv, aw_step = self.score(enc_out, x_lens, dec_out, aw_step)

            # Generate
            logits_step = self.fc_dec(dec_out)
            if 'c' in self.generate_feature:
                logits_step += self.fc_cv(cv)
            if 'y' in self.generate_feature:
                logits_step += self.fc_emb(y)

            # RNNLM fusion
            if self.rnnlm_fusion == 'cold_fusion_simple':
                # Fine-grained gating
                gate = F.sigmoid(self.fc_lm_gate(torch.cat([dec_out, lm_out], dim=-1)))
                logits_step += self.fc_lm(torch.mul(gate, lm_out))
            elif self.rnnlm_fusion == 'cold_fusion':
                # Prob injection
                rnnlm_feat = self.fc_lm_logits(logits_step_rnnlm)
                # Fine-grained gating
                gate = F.sigmoid(self.fc_lm_gate(
                    torch.cat([dec_out, rnnlm_feat], dim=-1)))
                logits_step += self.fc_lm(torch.mul(gate, lm_out))
                if getattr(self, 'bottleneck_dim_' + str(task)) == 0:
                    logits_step = F.relu(logits_step)
            elif self.rnnlm_fusion == 'logits_fusion':
                logits_step += logits_step_rnnlm

            if getattr(self, 'bottleneck_dim_' + str(task)) > 0:
                if self.rnnlm_fusion == 'cold_fusion':
                    logits_step = F.relu(logits_step)
                else:
                    logits_step = F.tanh(logits_step)
                logits_step = getattr(self, 'fc' + taskdir)(logits_step)

            # Pick up 1-best
            y = torch.max(logits_step.squeeze(1), dim=1)[1].unsqueeze(1)
            _best_hyps += [y]
            _aw += [aw_step]

            # Count lengths of hypotheses
            for b in range(batch):
                if not eos_flags[b]:
                    if y.data.cpu().numpy()[b] == eos:
                        eos_flags[b] = True
                    y_lens[b] += 1
                    # NOTE: include <EOS>

            # Break if <EOS> is outputed in all mini-batch
            if sum(eos_flags) == batch:
                break

        # Concatenate in L dimension
        _best_hyps = torch.cat(_best_hyps, dim=1)
        _aw = torch.stack(_aw, dim=1)

        # Convert to numpy
        _best_hyps = var2np(_best_hyps)
        _aw = var2np(_aw)

        if getattr(self, 'n_heads_' + str(task)) > 1:
            _aw = _aw[:, :, :, 0]
            # TODO(hirofumi): fix for MHA

        # Truncate by the first <EOS>
        if dir == 'bwd':
            # Reverse the order
            best_hyps = [_best_hyps[b, :y_lens[b]][::-1] for b in range(batch)]
            aw = [_aw[b, :y_lens[b]][::-1] for b in range(batch)]
        else:
            best_hyps = [_best_hyps[b, :y_lens[b]] for b in range(batch)]
            aw = [_aw[b, :y_lens[b]] for b in range(batch)]

        # Exclude <EOS>
        if exclude_eos:
            best_hyps = [best_hyps[b][:-1] if eos_flags[b]
                         else best_hyps[b] for b in range(batch)]

        return best_hyps, aw

    def beam_search(self, enc_out, x_lens, beam_width,
                    max_decode_len, min_decode_len, min_decode_len_ratio,
                    length_penalty, coverage_penalty, coverage_threshold, rnnlm_weight,
                    task, dir, exclude_eos):
        """Beam search decoding in the inference stage.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, enc_n_units]`
            x_lens (list): A list of length `[B]`
            beam_width (int): the size of beam
            max_decode_len (int): the maximum sequence length of tokens
            min_decode_len (int): the minimum sequence length of tokens
            min_decode_len_ratio (float):
            length_penalty (float): length penalty
            coverage_penalty (float): coverage penalty
            coverage_threshold (float): threshold for coverage penalty
            rnnlm_weight (float): the weight of RNNLM score
            task (int): the index of a task
            dir (str): fwd or bwd
            exclude_eos (bool):
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`

        """
        if dir == 'bwd':
            assert getattr(self, 'bwd_weight_' + str(task)) > 0

        batch_size = enc_out.size(0)

        # For shallow fusion
        if rnnlm_weight > 0 and not getattr(self, 'rnnlm_fusion_type_' + str(task)):
            assert getattr(self, 'rnnlm' + taskdir) is not None
            assert not getattr(self, 'rnnlm' + taskdir).training

        # Cold fusion
        if getattr(self, 'rnnlm_fusion_type_' + str(task)):
            assert getattr(self, 'rnnlm' + taskdir) is not None
            assert not getattr(self, 'rnnlm' + taskdir).training

        best_hyps, aw = [], []
        y_lens = np.zeros((batch_size,), dtype=np.int32)
        eos_flags = [False] * batch_size
        for b in range(batch_size):
            # Initialization per utterance
            dec_out, hx_list, cx_list = self._init_dec_state(
                enc_out[b:b + 1], x_lens[b], task, dir)
            context_vec = Variable(enc_out.data.new(
                1, 1, enc_out.size(-1)).fill_(0.), volatile=True)
            self.score.reset()

            complete = []
            beam = [{'hyp': [sos],
                     'score': 0,  # log 1
                     'dec_out': dec_out,
                     'hx_list': hx_list,
                     'cx_list': cx_list,
                     'context_vec': context_vec,
                     'aw_steps': [None],
                     'rnnlm_state': None,
                     'previous_coverage': 0}]
            for t in range(max_decode_len + 1):
                new_beam = []
                for i_beam in range(len(beam)):
                    # Update RNNLM states
                    if rnnlm_weight > 0 or getattr(self, 'rnnlm_fusion_type_' + str(task)):
                        y_rnnlm = Variable(enc_out.data.new(
                            1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                        y_rnnlm = getattr(self, 'rnnlm' + taskdir).embed(
                            y_rnnlm)
                        logits_step_rnnlm, rnnlm_out, rnnlm_state = getattr(self, 'rnnlm' + taskdir).predict(
                            y_rnnlm, beam[i_beam]['rnnlm_state'])

                    y = Variable(enc_out.data.new(
                        1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                    y = self.emb(y)

                    if t == 0:
                        dec_out = beam[i_beam]['dec_out']
                    else:
                        # Recurrency
                        dec_in = torch.cat(
                            [y, beam[i_beam]['context_vec']], dim=-1)
                        if getattr(self, 'rnnlm_fusion_type_' + str(task)) in ['embedding_fusion', 'state_embedding_fusion']:
                            dec_in = torch.cat([dec_in, y_rnnlm], dim=-1)
                        if getattr(self, 'rnnlm_fusion_type_' + str(task)) in ['state_fusion', 'state_embedding_fusion']:
                            dec_in = torch.cat([dec_in, rnnlm_out], dim=-1)

                        dec_out, hx_list, cx_list = getattr(self, 'decoder' + taskdir)(
                            dec_in, beam[i_beam]['hx_list'], beam[i_beam]['cx_list'])

                    # Score
                    context_vec, aw_step = self.score(
                        enc_out[b:b + 1, :x_lens[b]], x_lens[b:b + 1],
                        dec_out, beam[i_beam]['aw_steps'][-1])

                    # Generate
                    logits_step = getattr(self, 'W_d' + taskdir)(dec_out)
                    if 'c' in self.generate_feat:
                        logits_step += getattr(self, 'W_c' + taskdir)(context_vec)

                    # RNNLM fusion
                    if getattr(self, 'rnnlm_fusion_type_' + str(task)) == 'cold_fusion_simple':
                        # Fine-grained gating
                        gate = F.sigmoid(getattr(self, 'W_rnnlm_gate' + taskdir)(
                            torch.cat([dec_out, rnnlm_out], dim=-1)))
                        logits_step += getattr(self, 'W_rnnlm' + taskdir)(
                            torch.mul(gate, rnnlm_out))
                    elif getattr(self, 'rnnlm_fusion_type_' + str(task)) == 'cold_fusion':
                        # Prob injection
                        rnnlm_feat = getattr(self, 'W_rnnlm_logits' + taskdir)(
                            logits_step_rnnlm)
                        # Fine-grained gating
                        gate = F.sigmoid(getattr(self, 'W_rnnlm_gate' + taskdir)(
                            torch.cat([dec_out, rnnlm_feat], dim=-1)))
                        logits_step += getattr(self, 'W_rnnlm' + taskdir)(
                            torch.mul(gate, rnnlm_out))
                        if getattr(self, 'bottleneck_dim_' + str(task)) == 0:
                            logits_step = F.relu(logits_step)
                    elif getattr(self, 'rnnlm_fusion_type_' + str(task)) == 'logits_fusion':
                        logits_step += logits_step_rnnlm

                    if getattr(self, 'bottleneck_dim_' + str(task)) > 0:
                        if getattr(self, 'rnnlm_fusion_type_' + str(task)) == 'cold_fusion':
                            logits_step = F.relu(logits_step)
                        else:
                            logits_step = F.tanh(logits_step)
                        logits_step = getattr(self, 'fc' + taskdir)(
                            logits_step)

                    # Path through the softmax layer & convert to log-scale
                    log_probs = F.log_softmax(logits_step.squeeze(1), dim=1)
                    # log_probs = logits_step.squeeze(1)
                    # NOTE: `[1 (B), 1, n_classes]` -> `[1 (B), n_classes]`

                    # Pick up the top-k scores
                    log_probs_topk, indices_topk = torch.topk(
                        log_probs, k=beam_width, dim=1, largest=True, sorted=True)

                    for k in range(beam_width):
                        # Exclude short hypotheses
                        if indices_topk[0, k].data[0] == eos and len(beam[i_beam]['hyp']) < min_decode_len:
                            continue
                        if indices_topk[0, k].data[0] == eos and len(beam[i_beam]['hyp']) < x_lens[b] * min_decode_len_ratio:
                            continue

                        # Add length penalty
                        score = beam[i_beam]['score'] + \
                            log_probs_topk.data[0, k] + length_penalty

                        # Add coverage penalty
                        if coverage_penalty > 0:
                            # Recompute converage penalty in each step
                            score -= beam[i_beam]['previous_coverage'] * \
                                coverage_penalty

                            aw_steps = torch.stack(
                                beam[i_beam]['aw_steps'][1:] + [aw_step], dim=1)

                            if getattr(self, 'n_heads_' + str(task)) > 1:
                                cov_sum = aw_steps.data[0,
                                                        :, :, 0].cpu().numpy()
                                # TODO(hirofumi): fix for MHA
                            else:
                                cov_sum = aw_steps.data[0].cpu().numpy()
                            if coverage_threshold == 0:
                                cov_sum = np.sum(cov_sum)
                            else:
                                cov_sum = np.sum(cov_sum[np.where(
                                    cov_sum > coverage_threshold)[0]])
                            score += cov_sum * coverage_penalty
                        else:
                            cov_sum = 0

                        # Add RNNLM score
                        if rnnlm_weight > 0:
                            rnnlm_log_probs = F.log_softmax(
                                logits_step_rnnlm.squeeze(1), dim=1)
                            assert log_probs.size() == rnnlm_log_probs.size()
                            score += rnnlm_log_probs.data[0,
                                                          indices_topk.data[0, k]] * rnnlm_weight
                        elif not getattr(self, 'rnnlm_fusion_type_' + str(task)):
                            rnnlm_state = None

                        new_beam.append(
                            {'hyp': beam[i_beam]['hyp'] + [indices_topk.data[0, k]],
                             'score': score,
                             'hx_list': copy.deepcopy(hx_list),
                             'cx_list': copy.deepcopy(cx_list),
                             'dec_out': dec_out,
                             'context_vec': context_vec,
                             'aw_steps': beam[i_beam]['aw_steps'] + [aw_step],
                             'rnnlm_state': copy.deepcopy(rnnlm_state),
                             'previous_coverage': cov_sum})

                new_beam = sorted(
                    new_beam, key=lambda x: x['score'], reverse=True)

                # Remove complete hypotheses
                not_complete = []
                for cand in new_beam[:beam_width]:
                    if cand['hyp'][-1] == eos:
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
            aw += [complete[0]['aw_steps'][1:]]
            y_lens[b] = len(complete[0]['hyp'][1:])
            if complete[0]['hyp'][-1] == eos:
                eos_flags[b] = True

        # Concatenate in L dimension
        for b in range(len(aw)):
            aw[b] = var2np(torch.stack(aw[b], dim=1).squeeze(0))
            if getattr(self, 'n_heads_' + str(task)) > 1:
                aw[b] = aw[b][:, :, 0]
                # TODO(hirofumi): fix for MHA

        # Reverse the order
        if dir == 'bwd':
            best_hyps = [best_hyps[b][::-1] for b in range(batch_size)]
            aw = [aw[b][::-1] for b in range(batch_size)]

        # Exclude <EOS>
        if exclude_eos:
            best_hyps = [best_hyps[b][:-1] if eos_flags[b]
                         else best_hyps[b] for b in range(batch_size)]

        return best_hyps, aw
