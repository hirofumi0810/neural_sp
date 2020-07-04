#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN transducer."""

from collections import OrderedDict
import logging
import numpy as np
import random
import torch
import torch.nn as nn

from neural_sp.models.lm.rnnlm import RNNLM
from neural_sp.models.seq2seq.decoders.beam_search import BeamSearch
from neural_sp.models.seq2seq.decoders.ctc import CTC
from neural_sp.models.seq2seq.decoders.ctc import CTCPrefixScore
from neural_sp.models.seq2seq.decoders.decoder_base import DecoderBase
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list
from neural_sp.models.torch_utils import repeat
from neural_sp.models.torch_utils import tensor2np

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
        ctc_weight (float):
        ctc_lsm_prob (float): label smoothing probability for CTC
        ctc_fc_list (list):
        global_weight (float):
        mtl_per_batch (bool):
        param_init (str): parameter initialization method
        external_lm (RNNLM): external RNNLM for prediction network initialization

    """

    def __init__(self, special_symbols,
                 enc_n_units, rnn_type, n_units, n_projs, n_layers,
                 bottleneck_dim, emb_dim, vocab,
                 dropout, dropout_emb,
                 ctc_weight, ctc_lsm_prob, ctc_fc_list,
                 global_weight, mtl_per_batch, param_init, external_lm):

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
        self.rnnt_weight = global_weight - ctc_weight
        self.ctc_weight = ctc_weight
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

        if self.rnnt_weight > 0:
            # import warprnnt_pytorch
            # self.warprnnt_loss = warprnnt_pytorch.RNNTLoss()

            # Prediction network
            self.rnn = nn.ModuleList()
            rnn = nn.LSTM if rnn_type == 'lstm_transducer' else nn.GRU
            dec_odim = emb_dim
            self.proj = repeat(nn.Linear(n_units, n_projs), n_layers) if n_projs > 0 else None
            self.dropout = nn.Dropout(p=dropout)
            for _ in range(n_layers):
                self.rnn += [rnn(dec_odim, n_units, 1, batch_first=True)]
                dec_odim = n_units
                if n_projs > 0:
                    dec_odim = n_projs

            self.embed = nn.Embedding(vocab, emb_dim, padding_idx=self.pad)
            self.dropout_emb = nn.Dropout(p=dropout_emb)

            # Joint network
            self.w_enc = nn.Linear(enc_n_units, bottleneck_dim)
            self.w_dec = nn.Linear(dec_odim, bottleneck_dim, bias=False)
            self.output = nn.Linear(bottleneck_dim, vocab)

        self.reset_parameters(param_init)

        # prediction network initialization with pre-trained LM
        if external_lm is not None:
            assert external_lm.vocab == vocab
            assert external_lm.n_units == n_units
            assert external_lm.n_projs == n_projs
            assert external_lm.n_layers == n_layers
            param_dict = dict(external_lm.named_parameters())
            for n, p in self.named_parameters():
                if n in param_dict.keys() and p.size() == param_dict[n].size():
                    if 'output' in n:
                        continue
                    p.data = param_dict[n].data
                    logger.info('Overwrite %s' % n)

    @staticmethod
    def add_args(parser, args):
        """Add arguments."""
        group = parser.add_argument_group("RNN-T decoder")
        # common (LAS/RNN-T)
        if not hasattr(args, 'dec_n_units'):
            group.add_argument('--dec_n_units', type=int, default=512,
                               help='number of units in each decoder RNN layer')
            group.add_argument('--dec_n_projs', type=int, default=0,
                               help='number of units in the projection layer after each decoder RNN layer')
            group.add_argument('--dec_bottleneck_dim', type=int, default=1024,
                               help='number of dimensions of the bottleneck layer before the softmax layer')
            group.add_argument('--emb_dim', type=int, default=512,
                               help='number of dimensions in the embedding layer')
        return parser

    @staticmethod
    def define_name(dir_name, args):
        dir_name += '_' + args.dec_type

        dir_name += str(args.dec_n_units) + 'H'
        if args.dec_n_projs > 0:
            dir_name += str(args.dec_n_projs) + 'P'
        dir_name += str(args.dec_n_layers) + 'L'

        return dir_name

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

    def forward(self, eouts, elens, ys, task='all',
                teacher_logits=None, recog_params={}, idx2token=None):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys (list): length `B`, each of which contains a list of size `[L]`
            task (str): all/ys*/ys_sub*
            teacher_logits (FloatTensor): `[B, L, vocab]`
            recog_params (dict): parameters for MBR training
            idx2token ():
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
        if self.rnnt_weight > 0 and (task == 'all' or 'ctc' not in task):
            loss_transducer = self.forward_transducer(eouts, elens, ys)
            observation['loss_transducer'] = loss_transducer.item()
            if self.mtl_per_batch:
                loss += loss_transducer
            else:
                loss += loss_transducer * self.rnnt_weight

        observation['loss'] = loss.item()
        return loss, observation

    def forward_transducer(self, eouts, elens, ys):
        """Compute RNN-T loss.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys (list): length `B`, each of which contains a list of size `[L]`
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
        assert log_probs.size(2) == ys_out.size(1) + 1
        if self.device_id >= 0:
            ys_out = ys_out.cuda(self.device_id)
            elens = elens.cuda(self.device_id)
            ylens = ylens.cuda(self.device_id)
            import warp_rnnt
            loss = warp_rnnt.rnnt_loss(log_probs, ys_out.int(), elens, ylens,
                                       average_frames=False,
                                       reduction='mean',
                                       gather=False)
        else:
            import warprnnt_pytorch
            self.warprnnt_loss = warprnnt_pytorch.RNNTLoss()
            loss = self.warprnnt_loss(log_probs, ys_out.int(), elens, ylens)
            # NOTE: Transducer loss has already been normalized by bs
            # NOTE: index 0 is reserved for blank in warprnnt_pytorch

        return loss

    def joint(self, eouts, douts):
        """Combine encoder outputs and prediction network outputs.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            douts (FloatTensor): `[B, L, dec_n_units]`
        Returns:
            out (FloatTensor): `[B, T, L, vocab]`

        """
        eouts = eouts.unsqueeze(2)  # `[B, T, 1, enc_n_units]`
        douts = douts.unsqueeze(1)  # `[B, 1, L, dec_n_units]`
        out = torch.tanh(self.w_enc(eouts) + self.w_dec(douts))
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
        for lth in range(self.n_layers):
            if self.rnn_type == 'lstm_transducer':
                ys_emb, (h, c) = self.rnn[lth](ys_emb, hx=(dstate['hxs'][lth:lth + 1],
                                                           dstate['cxs'][lth:lth + 1]))
                new_cxs.append(c)
            elif self.rnn_type == 'gru_transducer':
                ys_emb, h = self.rnn[lth](ys_emb, hx=dstate['hxs'][lth:lth + 1])
            new_hxs.append(h)
            ys_emb = self.dropout(ys_emb)
            if self.proj is not None:
                ys_emb = torch.tanh(self.proj[lth](ys_emb))

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
            hyps (list): length `B`, each of which contains arrays of size `[L]`
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

        if idx2token is not None:
            for b in range(bs):
                if utt_ids is not None:
                    logger.debug('Utt-id: %s' % utt_ids[b])
                if refs_id is not None and self.vocab == idx2token.vocab:
                    logger.debug('Ref: %s' % idx2token(refs_id[b]))
                logger.debug('Hyp: %s' % idx2token(hyps[b]))

        return hyps, None

    def beam_search(self, eouts, elens, params, idx2token=None,
                    lm=None, lm_second=None, lm_second_bwd=None, ctc_log_probs=None,
                    nbest=1, exclude_eos=False,
                    refs_id=None, utt_ids=None, speakers=None,
                    ensmbl_eouts=None, ensmbl_elens=None, ensmbl_decs=[]):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            params (dict): hyperparameters for decoding
            idx2token (): converter from index to token
            lm: firsh path LM
            lm_second: second path LM
            lm_second_bwd: secoding path backward LM
            ctc_log_probs (FloatTensor): `[B, T, vocab]`
            nbest (int): number of N-best list
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (list): reference list
            utt_ids (list): utterance id list
            speakers (list): speaker list
            ensmbl_eouts (list): list of FloatTensor
            ensmbl_elens (list) list of list
            ensmbl_decs (list): list of torch.nn.Module
        Returns:
            nbest_hyps_idx (list): length `B`, each of which contains list of N hypotheses
            aws: dummy
            scores: dummy

        """
        bs = eouts.size(0)

        beam_width = params['recog_beam_width']
        assert 1 <= nbest <= beam_width
        ctc_weight = params['recog_ctc_weight']
        lm_weight = params['recog_lm_weight']
        lm_weight_second = params['recog_lm_second_weight']
        lm_weight_second_bwd = params['recog_lm_bwd_weight']
        # asr_state_carry_over = params['recog_asr_state_carry_over']
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

        if ctc_log_probs is not None:
            assert ctc_weight > 0
            ctc_log_probs = tensor2np(ctc_log_probs)

        nbest_hyps_idx = []
        eos_flags = []
        for b in range(bs):
            # Initialization per utterance
            y = eouts.new_zeros(bs, 1).fill_(self.eos).long()
            y_emb = self.dropout_emb(self.embed(y))
            dout, dstate = self.recurrency(y_emb, None)
            lmstate = None

            # For joint CTC-Attention decoding
            ctc_prefix_scorer = None
            if ctc_log_probs is not None:
                ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[b], self.blank, self.eos)

            if speakers is not None:
                if speakers[b] == self.prev_spk:
                    if lm_state_carry_over and isinstance(lm, RNNLM):
                        lmstate = self.lmstate_final
                self.prev_spk = speakers[b]

            helper = BeamSearch(beam_width, self.eos, ctc_weight, self.device_id)

            end_hyps = []
            hyps = [{'hyp': [self.eos],
                     'ys': [self.eos],
                     'score': 0.,
                     'score_rnnt': 0.,
                     'score_lm': 0.,
                     'score_ctc': 0.,
                     'dout': dout,
                     'dstate': dstate,
                     'lmstate': lmstate,
                     'ctc_state': ctc_prefix_scorer.initial_state() if ctc_prefix_scorer is not None else None}]
            for t in range(elens[b]):
                # preprocess for batch decoding
                douts = torch.cat([beam['dout'] for beam in hyps], dim=0)
                outs = self.joint(eouts[b:b + 1, t:t + 1].repeat([douts.size(0), 1, 1]), douts)
                scores_rnnt = torch.log_softmax(outs.squeeze(2).squeeze(1), dim=-1)

                # Update LM states for shallow fusion
                y = eouts.new_zeros(len(hyps), 1).long()
                for j, beam in enumerate(hyps):
                    y[j, 0] = beam['hyp'][-1]
                lmstate, scores_lm = None, None
                if lm is not None:
                    if hyps[0]['lmstate'] is not None:
                        lm_hxs = torch.cat([beam['lmstate']['hxs'] for beam in hyps], dim=1)
                        lm_cxs = torch.cat([beam['lmstate']['cxs'] for beam in hyps], dim=1)
                        lmstate = {'hxs': lm_hxs, 'cxs': lm_cxs}
                    lmout, lmstate, scores_lm = lm.predict(y, lmstate)

                new_hyps = []
                for j, beam in enumerate(hyps):
                    dout = douts[j:j + 1]
                    dstate = beam['dstate']
                    lmstate = beam['lmstate']

                    # Attention scores
                    total_scores_rnnt = beam['score_rnnt'] + scores_rnnt[j:j + 1]
                    total_scores = total_scores_rnnt * (1 - ctc_weight)

                    # Add LM score <after> top-K selection
                    total_scores_topk, topk_ids = torch.topk(
                        total_scores, k=beam_width, dim=-1, largest=True, sorted=True)
                    if lm is not None:
                        total_scores_lm = beam['score_lm'] + scores_lm[j, -1, topk_ids[0]]
                        total_scores_topk += total_scores_lm * lm_weight
                    else:
                        total_scores_lm = eouts.new_zeros(beam_width)

                    # Add CTC score
                    new_ctc_states, total_scores_ctc, total_scores_topk = helper.add_ctc_score(
                        beam['hyp'], topk_ids, beam['ctc_state'],
                        total_scores_topk, ctc_prefix_scorer)

                    for k in range(beam_width):
                        idx = topk_ids[0, k].item()

                        if idx == self.blank:
                            beam['score'] = total_scores_topk[0, k].item()
                            beam['score_rnnt'] = total_scores_topk[0, k].item()
                            new_hyps.append(beam.copy())
                            continue

                        # skip blank-dominant frames
                        # if total_scores_topk[0, self.blank].item() > 0.7:
                        #     continue

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

                        # store in cache
                        self.state_cache[hyp_str] = {
                            'dout': dout,
                            'dstate': new_dstate,
                            'lmstate': {'hxs': lmstate['hxs'][:, j:j + 1],
                                        'cxs': lmstate['cxs'][:, j:j + 1]} if lmstate is not None else None,
                        }

                        new_hyps.append({'hyp': hyp_id,
                                         'score': total_scores_topk[0, k].item(),
                                         'score_rnnt': total_scores_rnnt[0, idx].item(),
                                         'score_ctc': total_scores_ctc[k].item(),
                                         'score_lm': total_scores_lm[k].item(),
                                         'dout': dout,
                                         'dstate': new_dstate,
                                         'lmstate': {'hxs': lmstate['hxs'][:, j:j + 1],
                                                     'cxs': lmstate['cxs'][:, j:j + 1]} if lmstate is not None else None,
                                         'ctc_state': new_ctc_states[k] if ctc_prefix_scorer is not None else None})

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
                new_hyps_sorted = sorted(new_hyps, key=lambda x: x['score'], reverse=True)[:beam_width]

                # Remove complete hypotheses
                new_hyps, end_hyps, is_finish = helper.remove_complete_hyp(new_hyps_sorted, end_hyps)
                hyps = new_hyps[:]
                if is_finish:
                    break

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
                self.lm_rescoring(end_hyps, lm_second_bwd, lm_weight_second_bwd, tag='second_bwd')

            # Sort by score
            end_hyps = sorted(end_hyps, key=lambda x: x['score'], reverse=True)

            # Reset state cache
            self.state_cache = OrderedDict()

            if idx2token is not None:
                if utt_ids is not None:
                    logger.info('Utt-id: %s' % utt_ids[b])
                assert self.vocab == idx2token.vocab
                logger.info('=' * 200)
                for k in range(len(end_hyps)):
                    if refs_id is not None:
                        logger.info('Ref: %s' % idx2token(refs_id[b]))
                    logger.info('Hyp: %s' % idx2token(end_hyps[k]['hyp'][1:]))
                    logger.info('log prob (hyp): %.7f' % end_hyps[k]['score'])
                    if ctc_prefix_scorer is not None:
                        logger.info('log prob (hyp, ctc): %.7f' % (end_hyps[k]['score_ctc'] * ctc_weight))
                    if lm is not None:
                        logger.info('log prob (hyp, first-path lm): %.7f' % (end_hyps[k]['score_lm'] * lm_weight))
                    if lm_second is not None:
                        logger.info('log prob (hyp, second-path lm): %.7f' %
                                    (end_hyps[k]['score_lm_second'] * lm_weight_second))
                    if lm_second_bwd is not None:
                        logger.info('log prob (hyp, second-path lm, reverse): %.7f' %
                                    (end_hyps[k]['score_lm_second_rev'] * lm_weight_second_bwd))
                    logger.info('-' * 50)

            # N-best list
            nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:]) for n in range(nbest)]]

            # Check <eos>
            eos_flags.append([(end_hyps[n]['hyp'][-1] == self.eos) for n in range(nbest)])

        return nbest_hyps_idx, None, None
