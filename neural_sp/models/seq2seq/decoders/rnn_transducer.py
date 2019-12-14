#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN Transducer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging
import numpy as np
import random
import torch
import torch.nn as nn
import warp_rnnt

from neural_sp.models.criterion import kldiv_lsm_ctc
from neural_sp.models.seq2seq.decoders.ctc import CTC
from neural_sp.models.seq2seq.decoders.decoder_base import DecoderBase
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list
from neural_sp.models.torch_utils import repeat

random.seed(1)

LOG_0 = float(np.finfo(np.float32).min)
LOG_1 = 0

logger = logging.getLogger(__name__)


class RNNTransducer(DecoderBase):
    """RNN Transducer.

    Args:
        special_symbols (dict):
            eos (int): index for <eos> (shared with <sos>)
            unk (int): index for <unk>
            pad (int): index for <pad>
            blank (int): index for <blank>
        enc_n_units (int):
        rnn_type (str): lstm_transducer or gru_transducer
        n_units (int): number of units in each RNN layer
        n_projs (int): number of units in each projection layer
        n_layers (int): number of RNN layers
        bottleneck_dim (int): dimension of the bottleneck layer before the softmax layer for label generation
        emb_dim (int): dimension of the embedding in target spaces.
        vocab (int): number of nodes in softmax layer
        dropout (float): dropout probability for the RNN layer
        dropout_emb (float): dropout probability for the embedding layer
        lsm_prob (float): label smoothing probability
        ctc_weight (float):
        ctc_lsm_prob (float): label smoothing probability for CTC
        ctc_fc_list (list):
        lm_init (RNNLM):
        share_lm_softmax (bool):
        global_weight (float):
        mtl_per_batch (bool):
        param_init (float):
        end_pointing (bool):

    """

    def __init__(self,
                 special_symbols,
                 enc_n_units,
                 rnn_type,
                 n_units,
                 n_projs,
                 n_layers,
                 bottleneck_dim,
                 emb_dim,
                 vocab,
                 dropout=0.,
                 dropout_emb=0.,
                 lsm_prob=0.,
                 ctc_weight=0.,
                 ctc_lsm_prob=0.,
                 ctc_fc_list=[],
                 lm_init=None,
                 global_weight=1.,
                 mtl_per_batch=False,
                 param_init=0.1,
                 end_pointing=True):

        super(RNNTransducer, self).__init__()

        self.eos = special_symbols['eos']
        self.unk = special_symbols['unk']
        self.pad = special_symbols['pad']
        self.blank = special_symbols['blank']
        self.vocab = vocab
        self.rnn_type = rnn_type
        assert rnn_type in ['lstm_transducer', 'gru_transducer']
        self.enc_n_units = enc_n_units
        self.dec_n_units = n_units
        self.n_projs = n_projs
        self.n_layers = n_layers
        self.lsm_prob = lsm_prob
        self.ctc_weight = ctc_weight
        self.global_weight = global_weight
        self.mtl_per_batch = mtl_per_batch

        # VAD
        self.end_pointing = end_pointing

        # for cache
        self.prev_spk = ''
        self.lmstate_final = None
        self.state_cache = OrderedDict()

        if ctc_weight > 0:
            self.ctc = CTC(eos=self.eos,
                           blank=self.blank,
                           enc_n_units=enc_n_units,
                           vocab=vocab,
                           dropout=dropout,
                           lsm_prob=ctc_lsm_prob,
                           fc_list=ctc_fc_list,
                           param_init=param_init)

        if ctc_weight < global_weight:
            # import warprnnt_pytorch
            # self.warprnnt_loss = warprnnt_pytorch.RNNTLoss()

            # Prediction network
            rnn_l = nn.LSTM if rnn_type == 'lstm_transducer' else nn.GRU
            self.rnn = nn.ModuleList()
            self.dropout = nn.Dropout(p=dropout)
            if n_projs > 0:
                self.proj = repeat(nn.Linear(n_units, n_projs), n_layers)
            dec_idim = emb_dim
            for l in range(n_layers):
                self.rnn += [rnn_l(dec_idim, n_units, 1, batch_first=True)]
                dec_idim = n_projs if n_projs > 0 else n_units

            self.embed = nn.Embedding(vocab, emb_dim, padding_idx=self.pad)
            self.dropout_emb = nn.Dropout(p=dropout_emb)

            # Joint network
            self.w_enc = nn.Linear(enc_n_units, bottleneck_dim)
            self.w_dec = nn.Linear(dec_idim, bottleneck_dim, bias=False)
            self.output = nn.Linear(bottleneck_dim, vocab)

        self.reset_parameters(param_init)

        # prediction network initialization with pre-trained LM
        if lm_init is not None:
            assert lm_init.vocab == vocab
            assert lm_init.n_units == n_units
            assert lm_init.n_projs == n_projs
            assert lm_init.n_layers == n_layers

            param_dict = dict(lm_init.named_parameters())
            for n, p in self.named_parameters():
                if n in param_dict.keys() and p.size() == param_dict[n].size():
                    if 'output' in n:
                        continue
                    p.data = param_dict[n].data
                    logger.info('Overwrite %s' % n)

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0.)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
            elif p.dim() in [2, 4]:
                nn.init.uniform_(p, a=-param_init, b=param_init)
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', param_init))
            else:
                raise ValueError(n)

    def start_scheduled_sampling(self):
        self._ss_prob = 0.

    def forward(self, eouts, elens, ys, task='all', ys_hist=[], teacher_probs=None):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (IntTensor): `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            task (str): all or ys or ys_sub*
            ys_hist (list):
            teacher_probs (FloatTensor): `[B, L, vocab]`
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None, 'loss_transducer': None, 'loss_ctc': None}
        loss = eouts.new_zeros((1,))

        # CTC loss
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task):
            loss_ctc = self.ctc(eouts, elens, ys)
            observation['loss_ctc'] = loss_ctc.item()
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight

        # XE loss
        if self.global_weight - self.ctc_weight > 0 and (task == 'all' or ('ctc' not in task)):
            loss_transducer = self.forward_rnnt(eouts, elens, ys)
            observation['loss_transducer'] = loss_transducer.item()
            if self.mtl_per_batch:
                loss += loss_transducer
            else:
                loss += loss_transducer * (self.global_weight - self.ctc_weight)

        observation['loss'] = loss.item()
        return loss, observation

    def forward_rnnt(self, eouts, elens, ys):
        """Compute XE loss for the attention-based sequence-to-sequence model.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (IntTensor): `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[1]`

        """
        # Append <sos> and <eos>
        eos = eouts.new_zeros(1).fill_(self.eos).long()
        if self.end_pointing:
            _ys = [np2tensor(np.fromiter(y + [self.eos], dtype=np.int64), self.device_id) for y in ys]
        else:
            _ys = [np2tensor(np.fromiter(y, dtype=np.int64), self.device_id) for y in ys]
        ylens = np2tensor(np.fromiter([y.size(0) for y in _ys], dtype=np.int32))
        ys_in_pad = pad_list([torch.cat([eos, y], dim=0) for y in _ys], self.pad)
        ys_out_pad = pad_list(_ys, self.blank)

        # Update prediction network
        ys_emb = self.dropout_emb(self.embed(ys_in_pad))
        dout, _ = self.recurrency(ys_emb, None)

        # Compute output distribution
        logits = self.joint(eouts, dout)

        # Compute Transducer loss
        log_probs = torch.log_softmax(logits, dim=-1)
        if self.device_id >= 0:
            ys_out_pad = ys_out_pad.cuda(self.device_id)
            elens = elens.cuda(self.device_id)
            ylens = ylens.cuda(self.device_id)

        assert log_probs.size(2) == ys_out_pad.size(1) + 1
        # loss = self.warprnnt_loss(log_probs, ys_out_pad.int(), elens, ylens)
        # NOTE: Transducer loss has already been normalized by bs
        # NOTE: index 0 is reserved for blank in warprnnt_pytorch
        loss = warp_rnnt.rnnt_loss(log_probs, ys_out_pad.int(), elens, ylens,
                                   average_frames=False,
                                   reduction='mean',
                                   gather=False)

        # Label smoothing for Transducer
        # if self.lsm_prob > 0:
        #     loss = loss * (1 - self.lsm_prob) + kldiv_lsm_ctc(logits,
        #                                                       ylens=elens,
        #                                                       size_average=True) * self.lsm_prob
        # TODO(hirofumi): this leads to out of memory

        return loss

    def joint(self, eouts, douts, non_linear=torch.tanh):
        """Combine encoder outputs and prediction network outputs.

        Args:
            eouts (FloatTensor): `[B, T, n_units]`
            douts (FloatTensor): `[B, L, n_units]`
        Returns:
            out (FloatTensor): `[B, T, L, vocab]`

        """
        # broadcast
        eouts = eouts.unsqueeze(2)  # `[B, T, 1, n_units]`
        douts = douts.unsqueeze(1)  # `[B, 1, L, n_units]`
        out = non_linear(self.w_enc(eouts) + self.w_dec(douts))
        out = self.output(out)
        return out

    def recurrency(self, ys_emb, dstate):
        """Update prediction network.

        Args:
            ys_emb (FloatTensor): `[B, L, emb_dim]`
            dstate (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`
        Returns:
            dout (FloatTensor): `[B, L, emb_dim]`
            new_dstate (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`

        """
        if dstate is None:
            dstate = self.zero_state(ys_emb.size(0))
        new_dstate = {'hxs': None, 'cxs': None}

        new_hxs, new_cxs = [], []
        for l in range(self.n_layers):
            if self.rnn_type == 'lstm_transducer':
                ys_emb, (h_l, c_l) = self.rnn[l](ys_emb, hx=(dstate['hxs'][l:l + 1],
                                                             dstate['cxs'][l:l + 1]))
                new_cxs.append(c_l)
            elif self.rnn_type == 'gru_transducer':
                ys_emb, h_l = self.rnn[l](ys_emb, hx=dstate['hxs'][l:l + 1])
            new_hxs.append(h_l)
            ys_emb = self.dropout(ys_emb)
            if self.n_projs > 0:
                ys_emb = torch.tanh(self.proj[l](ys_emb))

        # Repackage
        new_dstate['hxs'] = torch.cat(new_hxs, dim=0)
        if self.rnn_type == 'lstm_transducer':
            new_dstate['cxs'] = torch.cat(new_cxs, dim=0)

        return ys_emb, new_dstate

    def zero_state(self, batch_size):
        """Initialize hidden states.

        Args:
            batch_size (int): batch size
        Returns:
            zero_state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`

        """
        w = next(self.parameters())
        zero_state = {'hxs': None, 'cxs': None}
        hxs, cxs = [], []
        for l in range(self.n_layers):
            if self.rnn_type == 'lstm_transducer':
                cxs.append(w.new_zeros(1, batch_size, self.dec_n_units))
            hxs.append(w.new_zeros(1, batch_size, self.dec_n_units))
        zero_state['hxs'] = torch.cat(hxs, dim=0)  # `[n_layers, B, dec_n_units]`
        if self.rnn_type == 'lstm_transducer':
            zero_state['cxs'] = torch.cat(cxs, dim=0)  # `[n_layers, B, dec_n_units]`
        return zero_state

    def greedy(self, eouts, elens, max_len_ratio, idx2token,
               exclude_eos=False, oracle=False,
               refs_id=None, utt_ids=None, speakers=None):
        """Greedy decoding in the inference stage.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from hypothesis
            oracle (bool): teacher-forcing mode
            refs_id (list): reference list
            utt_ids (list): utterance id list
            speakers (list): speaker list
        Returns:
            hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw: dummy

        """
        bs = eouts.size(0)

        hyps = []
        for b in range(bs):
            hyp_b = []
            # Initialization
            y = eouts.new_zeros(bs, 1).fill_(self.eos).long()
            y_emb = self.dropout_emb(self.embed(y))
            dout, dstate = self.recurrency(y_emb, None)

            for t in range(elens[b]):
                # Pick up 1-best per frame
                out = self.joint(eouts[b:b + 1, t:t + 1], dout.squeeze(1))
                y = out.squeeze(2).argmax(-1)
                idx = y[0].item()

                # Update prediction network only when predicting non-blank labels
                if idx != self.blank:
                    # early stop
                    if self.end_pointing and idx == self.eos:
                        if not exclude_eos:
                            hyp_b += [idx]
                        break

                    hyp_b += [idx]
                    if oracle:
                        y = eouts.new_zeros(1, 1).fill_(refs_id[b, len(hyp_b) - 1]).long()
                    y_emb = self.dropout_emb(self.embed(y))
                    dout, dstate = self.recurrency(y_emb, dstate)

            hyps += [hyp_b]

        for b in range(bs):
            if utt_ids is not None:
                logger.info('Utt-id: %s' % utt_ids[b])
            if refs_id is not None and self.vocab == idx2token.vocab:
                logger.info('Ref: %s' % idx2token(refs_id[b]))
            logger.info('Hyp: %s' % idx2token(hyps[b]))

        return hyps, None

    def beam_search(self, eouts, elens, params, idx2token,
                    lm=None, lm_rev=None, ctc_log_probs=None,
                    nbest=1, exclude_eos=False,
                    refs_id=None, utt_ids=None, speakers=None,
                    ensmbl_eouts=None, ensmbl_elens=None, ensmbl_decs=[]):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (IntTensor): `[B]`
            params (dict):
                recog_beam_width (int): size of hyp
                recog_max_len_ratio (int): maximum sequence length of tokens
                recog_min_len_ratio (float): minimum sequence length of tokens
                recog_length_penalty (float): length penalty
                recog_coverage_penalty (float): coverage penalty
                recog_coverage_threshold (float): threshold for coverage penalty
                recog_lm_weight (float): weight of LM score
            idx2token (): converter from index to token
            lm (RNNLM or GatedConvLM or TransformerLM):
            lm_rev (RNNLM or GatedConvLM or TransformerLM):
            ctc_log_probs (FloatTensor):
            nbest (int):
            exclude_eos (bool):
            refs_id (list):
            utt_ids (list):
            speakers (list):
            ensmbl_eouts (list): list of FloatTensor
            ensmbl_elens (list) list of list
            ensmbl_decs (list): list of torch.nn.Module
        Returns:
            nbest_hyps_idx (list): A list of length `[B]`, which contains list of N hypotheses
            aws: dummy
            scores: dummy

        """
        bs = eouts.size(0)
        best_hyps = []

        oracle = params['recog_oracle']
        beam_width = params['recog_beam_width']
        ctc_weight = params['recog_ctc_weight']
        lm_weight = params['recog_lm_weight']
        asr_state_carry_over = params['recog_asr_state_carry_over']
        lm_state_carry_over = params['recog_lm_state_carry_over']
        lm_usage = params['recog_lm_usage']

        if lm is not None:
            lm.eval()

        for b in range(bs):
            # Initialization
            y = eouts.new_zeros(bs, 1).fill_(self.eos).long()
            y_emb = self.dropout_emb(self.embed(y))
            dout, dstate = self.recurrency(y_emb, None)
            lmstate = None

            if lm_state_carry_over:
                lmstate = self.lmstate_final
            self.prev_spk = speakers[b]

            end_hyps = []
            hyps = [{'hyp': [self.eos],
                     'lattice': [],
                     'ref_id': [self.eos],
                     'score': 0.,
                     'score_lm': 0.,
                     'score_ctc': 0.,
                     'dout': dout,
                     'dstate': dstate,
                     'lmstate': lmstate,
                     }]
            for t in range(elens[b]):
                new_hyps = []
                for hyp in hyps:
                    prev_idx = ([self.eos] + refs_id[b])[t] if oracle else hyp['hyp'][-1]
                    score = hyp['score']
                    score_lm = hyp['score_lm']
                    dout = hyp['dout']
                    dstate = hyp['dstate']
                    lmstate = hyp['lmstate']

                    # Pick up the top-k scores
                    out = self.joint(eouts[b:b + 1, t:t + 1], dout.squeeze(1))
                    log_probs = torch.log_softmax(out.squeeze(2), dim=-1)
                    log_probs_topk, topk_ids = torch.topk(
                        log_probs[0, 0], k=min(beam_width, self.vocab), dim=-1, largest=True, sorted=True)

                    for k in range(beam_width):
                        idx = topk_ids[k].item()
                        score += log_probs_topk[k].item()

                        # Update prediction network only when predicting non-blank labels
                        lattice = hyp['lattice'] + [idx]
                        if idx == self.blank:
                            hyp_id = hyp['hyp']
                        else:
                            hyp_id = hyp['hyp'] + [idx]
                            hyp_str = ' '.join(list(map(str, hyp_id[1:])))
                            if hyp_str in self.state_cache.keys():
                                # from cache
                                dout = self.state_cache[hyp_str]['dout']
                                new_dstate = self.state_cache[hyp_str]['dstate']
                            else:
                                if oracle:
                                    y = eouts.new_zeros(1, 1).fill_(refs_id[b, len(hyp_id) - 1]).long()
                                else:
                                    y = eouts.new_zeros(1, 1).fill_(idx).long()
                                y_emb = self.dropout_emb(self.embed(y))
                                dout, new_dstate = self.recurrency(y_emb, dstate)

                                # Update LM states for shallow fusion
                                if lm_weight > 0 and lm is not None:
                                    _, lmstate, lm_log_probs = lm.predict(
                                        eouts.new_zeros(1, 1).fill_(prev_idx), hyp['lmstate'])
                                    local_score_lm = lm_log_probs[0, idx].item()
                                    score_lm += local_score_lm * lm_weight
                                    score += local_score_lm * lm_weight

                                # to cache
                                self.state_cache[hyp_str] = {
                                    'lattice': lattice,
                                    'dout': dout,
                                    'dstate': new_dstate,
                                    'lmstate': lmstate,
                                }

                        new_hyps.append({'hyp': hyp_id,
                                         'lattice': lattice,
                                         'score': score,
                                         'score_lm': score_lm,
                                         'score_ctc': 0,  # TODO(hirofumi):
                                         'dout': dout,
                                         'dstate': dstate if idx == self.blank else new_dstate,
                                         'lmstate': lmstate,
                                         })

                # Local pruning
                new_hyps_tmp = sorted(new_hyps, key=lambda x: x['score'], reverse=True)[:beam_width]

                # Remove complete hypotheses
                new_hyps = []
                for hyp in new_hyps_tmp:
                    if oracle:
                        if t == len(refs_id[b]):
                            end_hyps += [hyp]
                        else:
                            new_hyps += [hyp]
                    else:
                        if self.end_pointing and hyp['hyp'][-1] == self.eos:
                            end_hyps += [hyp]
                        else:
                            new_hyps += [hyp]
                if len(end_hyps) >= beam_width:
                    end_hyps = end_hyps[:beam_width]
                    logger.info('End-pointed at %d / %d frames' % (t, elens[b]))
                    break
                hyps = new_hyps[:]

            # Rescoing lattice
            if lm_weight > 0 and lm is not None and lm_usage == 'rescoring':
                new_hyps = []
                for hyp in hyps:
                    ys = [np2tensor(np.fromiter(hyp['hyp'], dtype=np.int64), self.device_id)]
                    ys_pad = pad_list(ys, lm.pad)
                    _, _, lm_log_probs = lm.predict(ys_pad, None)
                    score_ctc = 0  # TODO(hirofumi):
                    score_lm = lm_log_probs.sum() * lm_weight
                    new_hyps.append({'hyp': hyp['hyp'],
                                     'score': hyp['score'] + score_lm,
                                     'score_ctc': score_ctc,
                                     'score_lm': score_lm
                                     })
                hyps = sorted(new_hyps, key=lambda x: x['score'], reverse=True)

            # Exclude <eos>
            if False and exclude_eos and self.end_pointing and hyps[0]['hyp'][-1] == self.eos:
                best_hyps.append([hyps[0]['hyp'][1:-1]])
            else:
                best_hyps.append([hyps[0]['hyp'][1:]])

            # Reset state cache
            self.state_cache = OrderedDict()

            if utt_ids is not None:
                logger.info('Utt-id: %s' % utt_ids[b])
            if refs_id is not None and self.vocab == idx2token.vocab:
                logger.info('Ref: %s' % idx2token(refs_id[b]))
            logger.info('Hyp: %s' % idx2token(hyps[0]['hyp'][1:]))
            logger.info('log prob (hyp): %.7f' % hyps[0]['score'])
            if ctc_weight > 0 and ctc_log_probs is not None:
                logger.info('log prob (hyp, ctc): %.7f' % (hyps[0]['score_ctc']))
            # logger.info('log prob (lp): %.7f' % hyps[0]['score_lp'])
            if lm_weight > 0 and lm is not None:
                logger.info('log prob (hyp, lm): %.7f' % (hyps[0]['score_lm']))

        return np.array(best_hyps), None, None
