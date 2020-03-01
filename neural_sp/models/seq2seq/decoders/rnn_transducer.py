#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN transducer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging
import numpy as np
import random
import torch
import torch.nn as nn

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
    """RNN transducer.

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
        global_weight (float):
        mtl_per_batch (bool):
        param_init (str): parameter initialization method

    """

    def __init__(self, special_symbols,
                 enc_n_units, rnn_type, n_units, n_projs, n_layers,
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
                 param_init=0.1):

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
                           param_init=0.1)

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
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
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

    def forward(self, eouts, elens, ys, task='all', ys_hist=[],
                teacher_logits=None, recog_params={}):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            task (str): all/ys*/ys_sub*
            ys_hist (list): dummy (not used)
            teacher_logits (FloatTensor): `[B, L, vocab]`
            recog_params (dict): parameters for MBR training
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None, 'loss_transducer': None, 'loss_ctc': None, 'loss_mbr': None}
        loss = eouts.new_zeros((1,))

        # CTC loss
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task):
            loss_ctc, _ = self.ctc(eouts, elens, ys)
            observation['loss_ctc'] = loss_ctc.item()
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight

        # XE loss
        if self.global_weight - self.ctc_weight > 0 and (task == 'all' or 'ctc' not in task):
            loss_transducer = self.forward_transducer(eouts, elens, ys)
            observation['loss_transducer'] = loss_transducer.item()
            if self.mtl_per_batch:
                loss += loss_transducer
            else:
                loss += loss_transducer * (self.global_weight - self.ctc_weight)

        observation['loss'] = loss.item()
        return loss, observation

    def forward_transducer(self, eouts, elens, ys):
        """Compute RNN-T loss.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[1]`

        """
        # Append <sos> and <eos>
        eos = eouts.new_zeros(1).fill_(self.eos).long()
        _ys = [np2tensor(np.fromiter(y, dtype=np.int64), self.device_id) for y in ys]
        ylens = np2tensor(np.fromiter([y.size(0) for y in _ys], dtype=np.int32))
        ys_in = pad_list([torch.cat([eos, y], dim=0) for y in _ys], self.pad)
        ys_out = pad_list(_ys, self.blank)

        # Update prediction network
        ys_emb = self.dropout_emb(self.embed(ys_in))
        dout, _ = self.recurrency(ys_emb, None)

        # Compute output distribution
        logits = self.joint(eouts, dout)

        # Compute Transducer loss
        log_probs = torch.log_softmax(logits, dim=-1)
        ys_out = ys_out.cuda(self.device_id)
        elens = elens.cuda(self.device_id)
        ylens = ylens.cuda(self.device_id)

        assert log_probs.size(2) == ys_out.size(1) + 1
        # loss = self.warprnnt_loss(log_probs, ys_out.int(), elens, ylens)
        # NOTE: Transducer loss has already been normalized by bs
        # NOTE: index 0 is reserved for blank in warprnnt_pytorch
        import warp_rnnt
        loss = warp_rnnt.rnnt_loss(log_probs, ys_out.int(), elens, ylens,
                                   average_frames=False,
                                   reduction='mean',
                                   gather=False)
        return loss

    def joint(self, eouts, douts, activation=torch.tanh):
        """Combine encoder outputs and prediction network outputs.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            douts (FloatTensor): `[B, L, dec_n_units]`
        Returns:
            out (FloatTensor): `[B, T, L, vocab]`

        """
        # broadcast
        eouts = eouts.unsqueeze(2)  # `[B, T, 1, enc_n_units]`
        douts = douts.unsqueeze(1)  # `[B, 1, L, dec_n_units]`
        out = activation(self.w_enc(eouts) + self.w_dec(douts))
        out = self.output(out)
        return out

    def recurrency(self, ys_emb, dstate):
        """Update prediction network.

        Args:
            ys_emb (FloatTensor): `[B, L, emb_dim]`
            dstate (dict):
                hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                cxs (FloatTensor): `[n_layers, B, dec_n_units]`
        Returns:
            dout (FloatTensor): `[B, L, emb_dim]`
            new_dstate (dict):
                hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                cxs (FloatTensor): `[n_layers, B, dec_n_units]`

        """
        if dstate is None:
            dstate = self.zero_state(ys_emb.size(0))
        new_dstate = {'hxs': None, 'cxs': None}

        new_hxs, new_cxs = [], []
        for l in range(self.n_layers):
            if self.rnn_type == 'lstm_transducer':
                ys_emb, (h, c) = self.rnn[l](ys_emb, hx=(dstate['hxs'][l:l + 1],
                                                         dstate['cxs'][l:l + 1]))
                new_cxs.append(c)
            elif self.rnn_type == 'gru_transducer':
                ys_emb, h = self.rnn[l](ys_emb, hx=dstate['hxs'][l:l + 1])
            new_hxs.append(h)
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
                hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                cxs (FloatTensor): `[n_layers, B, dec_n_units]`

        """
        w = next(self.parameters())
        zero_state = {'hxs': None, 'cxs': None}
        zero_state['hxs'] = w.new_zeros(self.n_layers, batch_size, self.dec_n_units)
        if self.rnn_type == 'lstm_transducer':
            zero_state['cxs'] = w.new_zeros(self.n_layers, batch_size, self.dec_n_units)
        return zero_state

    def greedy(self, eouts, elens, max_len_ratio, idx2token,
               exclude_eos=False, refs_id=None, utt_ids=None, speakers=None):
        """Greedy decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from hypothesis
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
            y = eouts.new_zeros(1, 1).fill_(self.eos).long()
            y_emb = self.dropout_emb(self.embed(y))
            dout, dstate = self.recurrency(y_emb, None)

            for t in range(elens[b]):
                # Pick up 1-best per frame
                out = self.joint(eouts[b:b + 1, t:t + 1], dout)
                y = out.squeeze(2).argmax(-1)
                idx = y[0].item()

                # Update prediction network only when predicting non-blank labels
                if idx != self.blank:
                    hyp_b += [idx]
                    y_emb = self.dropout_emb(self.embed(y))
                    dout, dstate = self.recurrency(y_emb, dstate)

            hyps += [hyp_b]

        for b in range(bs):
            if utt_ids is not None:
                logger.debug('Utt-id: %s' % utt_ids[b])
            if refs_id is not None and self.vocab == idx2token.vocab:
                logger.debug('Ref: %s' % idx2token(refs_id[b]))
            logger.debug('Hyp: %s' % idx2token(hyps[b]))

        return hyps, None

    def beam_search(self, eouts, elens, params, idx2token,
                    lm=None, lm_second=None, lm_second_bwd=None, ctc_log_probs=None,
                    nbest=1, exclude_eos=False,
                    refs_id=None, utt_ids=None, speakers=None,
                    ensmbl_eouts=None, ensmbl_elens=None, ensmbl_decs=[]):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            params (dict):
                recog_beam_width (int): size of beam
                recog_max_len_ratio (int): maximum sequence length of tokens
                recog_min_len_ratio (float): minimum sequence length of tokens
                recog_length_penalty (float): length penalty
                recog_coverage_penalty (float): coverage penalty
                recog_coverage_threshold (float): threshold for coverage penalty
                recog_lm_weight (float): weight of LM score
            idx2token (): converter from index to token
            lm: firsh path LM
            lm_second: second path LM
            lm_second_bwd: secoding path backward LM
            ctc_log_probs (FloatTensor):
            nbest (int):
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (list): reference list
            utt_ids (list): utterance id list
            speakers (list): speaker list
            ensmbl_eouts (list): list of FloatTensor
            ensmbl_elens (list) list of list
            ensmbl_decs (list): list of torch.nn.Module
        Returns:
            nbest_hyps_idx (list): A list of length `[B]`, which contains list of N hypotheses
            aws: dummy
            scores: dummy

        """
        bs = eouts.size(0)

        beam_width = params['recog_beam_width']
        ctc_weight = params['recog_ctc_weight']
        lm_weight = params['recog_lm_weight']
        lm_weight_second = params['recog_lm_second_weight']
        lm_weight_second_bwd = params['recog_lm_bwd_weight']
        asr_state_carry_over = params['recog_asr_state_carry_over']
        lm_state_carry_over = params['recog_lm_state_carry_over']

        if lm is not None:
            assert lm_weight > 0
            lm.eval()
        if lm_second is not None:
            assert lm_weight_second > 0
            lm_second.eval()
        if lm_second_bwd is not None:
            assert lm_weight_second_bwd > 0
            lm_second_bwd.eval()

        nbest_hyps_idx = []
        eos_flags = []
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
                     'ref_id': [self.eos],
                     'score': 0.,
                     'score_rnnt': 0.,
                     'score_lm': 0.,
                     'score_ctc': 0.,
                     'dout': dout,
                     'dstate': dstate,
                     'lmstate': lmstate}]
            for t in range(elens[b]):
                # preprocess for batch decoding
                douts = torch.cat([beam['dout'] for beam in hyps], dim=0)
                outs = self.joint(eouts[b:b + 1, t:t + 1].repeat([douts.size(0), 1, 1]), douts)
                log_probs = torch.log_softmax(outs.squeeze(2).squeeze(1), dim=-1)

                new_hyps = []
                for j, beam in enumerate(hyps):
                    prev_idx = beam['hyp'][-1]
                    dout = douts[j:j + 1]
                    dstate = beam['dstate']
                    lmstate = beam['lmstate']

                    # Pick up the top-k scores
                    log_probs_topk, topk_ids = torch.topk(log_probs[j], k=beam_width, dim=-1, largest=True, sorted=True)

                    for k in range(beam_width):
                        idx = topk_ids[k].item()

                        if idx == self.blank:
                            beam['score'] += log_probs_topk[k].item()
                            beam['score_rnnt'] += log_probs_topk[k].item()
                            new_hyps.append(beam.copy())
                            continue

                        # skip blank-dominant frames
                        # if log_probs_topk[self.blank].item() > 0.7:
                        #     continue

                        beam['score_rnnt'] += log_probs_topk[k].item()
                        score = beam['score_rnnt']

                        # Update prediction network only when predicting non-blank labels
                        hyp_id = beam['hyp'] + [idx]
                        hyp_str = ' '.join(list(map(str, hyp_id)))
                        # if hyp_str in self.state_cache.keys():
                        #     # from cache
                        #     dout = self.state_cache[hyp_str]['dout']
                        #     new_dstate = self.state_cache[hyp_str]['dstate']
                        #     lmstate = self.state_cache[hyp_str]['lmstate']
                        # else:
                        y = eouts.new_zeros(1, 1).fill_(idx).long()
                        y_emb = self.dropout_emb(self.embed(y))
                        dout, new_dstate = self.recurrency(y_emb, dstate)

                        # Update LM states for shallow fusion
                        if lm is not None:
                            _, lmstate, scores_lm = lm.predict(
                                eouts.new_zeros(1, 1).fill_(prev_idx), lmstate)
                            beam['score_lm'] += scores_lm[0, -1, idx].item()
                            score += scores_lm[0, -1, idx].item() * lm_weight

                        # TODO: add CTC score

                        # store in cache
                        self.state_cache[hyp_str] = {
                            'dout': dout,
                            'dstate': new_dstate,
                            'lmstate': lmstate,
                        }

                        new_hyps.append({'hyp': hyp_id,
                                         'score': score,
                                         'score_rnnt': beam['score_rnnt'],
                                         'score_lm': beam['score_lm'],
                                         'score_ctc': beam['score_ctc'],
                                         'dout': dout,
                                         'dstate': new_dstate,
                                         'lmstate': lmstate})

                # Merge hypotheses having the same token sequences
                new_hyps_merged = {}
                for beam in new_hyps:
                    hyp_str = ' '.join(list(map(str, beam['hyp'])))
                    if hyp_str not in new_hyps_merged.keys():
                        new_hyps_merged[hyp_str] = beam
                    elif hyp_str in new_hyps_merged.keys():
                        if beam['score'] > new_hyps_merged[hyp_str]['score']:
                            new_hyps_merged[hyp_str] = beam
                new_hyps = [v for v in new_hyps_merged.values()]

                # Local pruning
                new_hyps_tmp = sorted(new_hyps, key=lambda x: x['score'], reverse=True)[:beam_width]

                # Remove complete hypotheses
                new_hyps = []
                for hyp in new_hyps_tmp:
                    new_hyps += [hyp]
                if len(end_hyps) >= beam_width:
                    end_hyps = end_hyps[:beam_width]
                    break
                hyps = new_hyps[:]

            # Global pruning
            if len(end_hyps) == 0:
                end_hyps = hyps[:]
            elif len(end_hyps) < nbest and nbest > 1:
                end_hyps.extend(hyps[:nbest - len(end_hyps)])

            # forward second path LM rescoring
            if lm_second is not None:
                self.lm_rescoring(end_hyps, lm_second, lm_weight_second, tag='second')

            # backward secodn path LM rescoring
            if lm_second_bwd is not None:
                self.lm_rescoring(end_hyps, lm_second_bwd, lm_weight_second_bwd, tag='second_rev')

            end_hyps = sorted(end_hyps, key=lambda x: x['score'], reverse=True)

            # Reset state cache
            self.state_cache = OrderedDict()

            if utt_ids is not None:
                logger.info('Utt-id: %s' % utt_ids[b])
            if idx2token is not None:
                logger.info('=' * 200)
                for k in range(len(end_hyps)):
                    if refs_id is not None and self.vocab == idx2token.vocab:
                        logger.info('Ref: %s' % idx2token(refs_id[b]))
                    logger.info('Hyp: %s' % idx2token(end_hyps[k]['hyp'][1:]))
                    logger.info('log prob (hyp): %.7f' % end_hyps[k]['score'])
                    if ctc_log_probs is not None:
                        logger.info('log prob (hyp, ctc): %.7f' % (end_hyps[k]['score_ctc']))
                    if lm is not None:
                        logger.info('log prob (hyp, lm): %.7f' % (end_hyps[k]['score_lm']))
                    logger.info('-' * 50)

            # N-best list
            nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:]) for n in range(nbest)]]

            # Check <eos>
            eos_flags.append([(end_hyps[n]['hyp'][-1] == self.eos) for n in range(nbest)])

        return nbest_hyps_idx, None, None
