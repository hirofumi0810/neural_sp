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
from neural_sp.models.seq2seq.decoders.decoder_base import DecoderBase
from neural_sp.models.torch_utils import (
    np2tensor,
    pad_list,
    repeat,
    tensor2scalar
)

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
        enc_n_units (int): number of units of encoder outputs
        rnn_type (str): lstm_transducer/gru_transducer
        n_units (int): number of units in each RNN layer
        n_projs (int): number of units in each projection layer
        n_layers (int): number of RNN layers
        bottleneck_dim (int): dimension of bottleneck layer before softmax layer for label generation
        emb_dim (int): dimension of embedding in target spaces.
        vocab (int): number of nodes in softmax layer
        dropout (float): dropout probability for RNN layer
        dropout_emb (float): dropout probability for embedding layer
        ctc_weight (float): CTC loss weight
        ctc_lsm_prob (float): label smoothing probability for CTC
        ctc_fc_list (List): fully-connected layer configuration before the CTC softmax
        external_lm (RNNLM): external RNNLM for prediction network initialization
        global_weight (float): global loss weight for multi-task learning
        mtl_per_batch (bool): change mini-batch per task for multi-task training
        param_init (float): parameter initialization method

    """

    def __init__(self, special_symbols,
                 enc_n_units, rnn_type, n_units, n_projs, n_layers,
                 bottleneck_dim, emb_dim, vocab,
                 dropout, dropout_emb,
                 ctc_weight, ctc_lsm_prob, ctc_fc_list,
                 external_lm, global_weight, mtl_per_batch, param_init):

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
        self.embed_cache = None

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
                dec_odim = n_projs if n_projs > 0 else n_units

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

    def forward(self, eouts, elens, ys, task='all',
                teacher_logits=None,
                recog_params={}, idx2token=None, trigger_points=None):
        """Forward pass.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys (List): length `[B]`, each of which contains a list of size `[L]`
            task (str): all/ys*/ys_sub*
            teacher_logits (FloatTensor): `[B, L, vocab]`
            recog_params (dict): parameters for MBR training
            idx2token ():
            trigger_points (np.ndarray): `[B, L]`
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None, 'loss_transducer': None, 'loss_ctc': None, 'loss_mbr': None}
        loss = eouts.new_zeros((1,))

        # CTC loss
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task):
            loss_ctc, _ = self.ctc(eouts, elens, ys)
            observation['loss_ctc'] = tensor2scalar(loss_ctc)
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight

        # RNN-T loss
        if self.rnnt_weight > 0 and (task == 'all' or 'ctc' not in task):
            loss_transducer = self.forward_transducer(eouts, elens, ys)
            observation['loss_transducer'] = tensor2scalar(loss_transducer)
            if self.mtl_per_batch:
                loss += loss_transducer
            else:
                loss += loss_transducer * self.rnnt_weight

        observation['loss'] = tensor2scalar(loss)
        return loss, observation

    def forward_transducer(self, eouts, elens, ys):
        """Compute Transducer loss.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys (List): length `[B]`, each of which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[1]`

        """
        # Append <sos> and <eos>
        _ys = [np2tensor(np.fromiter(y, dtype=np.int64), eouts.device) for y in ys]
        ylens = np2tensor(np.fromiter([y.size(0) for y in _ys], dtype=np.int32))
        eos = eouts.new_zeros((1,), dtype=torch.int64).fill_(self.eos)
        ys_in = pad_list([torch.cat([eos, y], dim=0) for y in _ys], self.pad)  # `[B, L+1]`
        ys_out = pad_list(_ys, self.blank)  # `[B, L]`

        # Update prediction network
        ys_emb = self.dropout_emb(self.embed(ys_in))
        dout, _ = self.recurrency(ys_emb, None)

        # Compute output distribution
        logits = self.joint(eouts, dout)  # `[B, T, L+1, vocab]`

        # Compute Transducer loss
        log_probs = torch.log_softmax(logits, dim=-1)
        assert log_probs.size(2) == ys_out.size(1) + 1
        if self.device_id >= 0:
            ys_out = ys_out.to(eouts.device)
            elens = elens.to(eouts.device)
            ylens = ylens.to(eouts.device)
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
                ys_emb = torch.relu(self.proj[lth](ys_emb))

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
               exclude_eos=False, refs_id=None, utt_ids=None, speakers=None,
               trigger_points=None, teacher_force=False):
        """Greedy decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (List): reference list
            utt_ids (List): utterance id list
            speakers (List): speaker list
            trigger_points: dummy
            teacher_force: dummy
        Returns:
            hyps (List): length `[B]`, each of which contains arrays of size `[L]`
            aw: dummy

        """
        bs = eouts.size(0)

        hyps = []
        for b in range(bs):
            hyp_b = []
            # Initialization
            y = eouts.new_zeros((1, 1), dtype=torch.int64).fill_(self.eos)
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
                logger.debug('=' * 200)

        return hyps, None

    def cache_embedding(self, device):
        """Cache token emebdding."""
        if self.embed_cache is None:
            indices = torch.arange(0, self.vocab, 1, dtype=torch.int64).to(device)
            self.embed_cache = self.dropout_emb(self.embed(indices))  # `[1, vocab, emb_dim]`

    def beam_search(self, eouts, elens, params, idx2token=None,
                    lm=None, lm_second=None, lm_second_bwd=None, ctc_log_probs=None,
                    nbest=1, exclude_eos=False,
                    refs_id=None, utt_ids=None, speakers=None,
                    ensmbl_eouts=[], ensmbl_elens=[], ensmbl_decs=[]):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            params (dict): decoding hyperparameters
            idx2token (): converter from index to token
            lm (torch.nn.module): firsh path LM
            lm_second (torch.nn.module): second path LM
            lm_second_bwd (torch.nn.module): secoding path backward LM
            ctc_log_probs (FloatTensor): `[B, T, vocab]`
            nbest (int): number of N-best list
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (List): reference list
            utt_ids (List): utterance id list
            speakers (List): speaker list
            ensmbl_eouts (List[FloatTensor]): encoder outputs for ensemble models
            ensmbl_elens (List[IntTensor]) encoder outputs for ensemble models
            ensmbl_decs (List[torch.nn.Module): decoders for ensemble models
        Returns:
            nbest_hyps_idx (List): length `[B]`, each of which contains list of N hypotheses
            aws: dummy
            scores: dummy

        """
        bs = eouts.size(0)

        beam_width = params.get('recog_beam_width')
        assert 1 <= nbest <= beam_width
        ctc_weight = params.get('recog_ctc_weight')
        assert ctc_weight == 0
        assert ctc_log_probs is None
        cache_emb = params.get('recog_cache_embedding')
        lm_weight = params.get('recog_lm_weight')
        lm_weight_second = params.get('recog_lm_second_weight')
        lm_weight_second_bwd = params.get('recog_lm_bwd_weight')
        lm_state_CO = params.get('recog_lm_state_carry_over')
        softmax_smoothing = params.get('recog_softmax_smoothing')
        beam_search_type = params.get('recog_rnnt_beam_search_type')

        helper = BeamSearch(beam_width, self.eos, ctc_weight, lm_weight, eouts.device)
        lm = helper.verify_lm_eval_mode(lm, lm_weight, cache_emb)
        if lm is not None:
            assert isinstance(lm, RNNLM)
        lm_second = helper.verify_lm_eval_mode(lm_second, lm_weight_second, cache_emb)
        lm_second_bwd = helper.verify_lm_eval_mode(lm_second_bwd, lm_weight_second_bwd, cache_emb)

        # cache token embeddings
        if cache_emb:
            self.cache_embedding(eouts.device)

        nbest_hyps_idx = []
        for b in range(bs):
            # Initialization per utterance
            dstate = {'hxs': eouts.new_zeros(self.n_layers, 1, self.dec_n_units),
                      'cxs': eouts.new_zeros(self.n_layers, 1, self.dec_n_units)}
            lmstate = {'hxs': eouts.new_zeros(lm.n_layers, 1, lm.n_units),
                       'cxs': eouts.new_zeros(lm.n_layers, 1, lm.n_units)} if lm is not None else None

            if speakers is not None:
                if speakers[b] == self.prev_spk:
                    if lm_state_CO:
                        lmstate = self.lmstate_final
                self.prev_spk = speakers[b]

            end_hyps = []
            hyps = [{'hyp': [self.eos],
                     'hyp_ids_str': '',
                     'ys': [self.eos],
                     'score': 0.,
                     'score_rnnt': 0.,
                     'score_lm': 0.,
                     'dout': None,
                     'dstate': dstate,
                     'lmstate': lmstate,
                     'update_pred_net': True}]

            if beam_search_type == 'simple':
                hyps, new_hyps_sorted = self._beam_search_simple(
                    hyps, helper, eouts[b:b + 1], elens[b], softmax_smoothing, lm)
            elif beam_search_type == 'time_sync':
                raise NotImplementedError
            else:
                raise NotImplementedError(beam_search_type)

            # Global pruning
            end_hyps = hyps[:]
            if len(end_hyps) < nbest and nbest > 1:
                end_hyps.extend(new_hyps_sorted[:nbest - len(end_hyps)])

            # forward/backward second path LM rescoring
            helper.lm_rescoring(end_hyps, lm_second, lm_weight_second, tag='second')
            helper.lm_rescoring(end_hyps, lm_second_bwd, lm_weight_second_bwd, tag='second_bwd')

            # Sort by score
            end_hyps = sorted(end_hyps, key=lambda x: x['score'] / max(len(x['hyp'][1:]), 1), reverse=True)

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
                    if len(end_hyps[k]['hyp']) > 1:
                        logger.info('num tokens (hyp): %d' % len(end_hyps[k]['hyp'][1:]))
                    logger.info('log prob (hyp): %.7f' % end_hyps[k]['score'])
                    logger.info('log prob (hyp, rnnt): %.7f' % end_hyps[k]['score_rnnt'])
                    if lm is not None:
                        logger.info('log prob (hyp, first-path lm): %.7f' %
                                    (end_hyps[k]['score_lm'] * lm_weight))
                    if lm_second is not None:
                        logger.info('log prob (hyp, second-path lm): %.7f' %
                                    (end_hyps[k]['score_lm_second'] * lm_weight_second))
                    if lm_second_bwd is not None:
                        logger.info('log prob (hyp, second-path lm, reverse): %.7f' %
                                    (end_hyps[k]['score_lm_second_bwd'] * lm_weight_second_bwd))
                    logger.info('-' * 50)

            # N-best list
            nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:]) for n in range(nbest)]]

        # Store ASR/LM state
        if bs == 1:
            self.dstates_final = end_hyps[0]['dstate']
            self.lmstate_final = end_hyps[0]['lmstate']

        return nbest_hyps_idx, None, None

    def _beam_search_simple(self, hyps, helper, eout, elen, softmax_smoothing, lm):
        beam_width = helper.beam_width
        lm_weight = helper.lm_weight
        merge_prob = True

        for t in range(elen):
            # bachfy all hypotheses (not in the cache, non-blank) for prediction network and LM
            batch_hyps = [beam for beam in hyps if beam['update_pred_net']]
            if len(batch_hyps) > 0:
                ys = eout.new_zeros((len(batch_hyps), 1), dtype=torch.int64)
                for i, beam in enumerate(batch_hyps):
                    ys[i] = beam['hyp'][-1]
                dstates_prev = {'hxs': torch.cat([beam['dstate']['hxs'] for beam in batch_hyps], dim=1),
                                'cxs': torch.cat([beam['dstate']['cxs'] for beam in batch_hyps], dim=1)}
                ys_emb = self.dropout_emb(self.embed(ys)) if self.embed_cache is None else self.embed_cache[ys]
                douts, dstates = self.recurrency(ys_emb, dstates_prev)

                # Update LM states for shallow fusion
                _, lmstates, scores_lm = helper.update_rnnlm_state_batch(lm, batch_hyps, ys)

                hyp_ids_strs = [beam['hyp_ids_str'] for beam in hyps]

                for i, beam in enumerate(batch_hyps):
                    dstate = {'hxs': dstates['hxs'][:, i:i + 1],
                              'cxs': dstates['cxs'][:, i:i + 1]}
                    lmstate = {'hxs': lmstates['hxs'][:, i:i + 1],
                               'cxs': lmstates['cxs'][:, i:i + 1]} if lmstates is not None else None
                    index = hyp_ids_strs.index(beam['hyp_ids_str'])

                    hyps[index]['dout'] = douts[i:i + 1]
                    hyps[index]['dstate'] = dstate
                    hyps[index]['lmstate'] = lmstate
                    hyps[index]['next_scores_lm'] = scores_lm[i:i + 1]
                    assert hyps[index]['update_pred_net']
                    hyps[index]['update_pred_net'] = False

                    # register to cache
                    self.state_cache[beam['hyp_ids_str']] = {
                        'dout': douts[i:i + 1],
                        'dstate': dstate,
                        'next_scores_lm': scores_lm[i:i + 1],
                        'lmstate': lmstate,
                    }

            # batchfy all hypotheses for joint network
            douts = torch.cat([beam['dout'] for beam in hyps], dim=0)
            logits = self.joint(eout[:, t:t + 1].repeat([len(hyps), 1, 1]), douts)
            logits *= softmax_smoothing
            scores_rnnt = torch.log_softmax(logits.squeeze(2).squeeze(1), dim=-1)  # `[B, vocab]`

            new_hyps = []
            for j, beam in enumerate(hyps):
                # Transducer scores
                total_scores_rnnt = beam['score_rnnt'] + scores_rnnt[j:j + 1]
                total_scores_topk, topk_ids = torch.topk(
                    total_scores_rnnt, k=beam_width, dim=-1, largest=True, sorted=True)

                for k in range(beam_width):
                    idx = topk_ids[0, k].item()

                    if idx == self.blank:
                        new_hyps.append(beam.copy())
                        new_hyps[-1]['score'] += scores_rnnt[j, self.blank].item()
                        new_hyps[-1]['score_rnnt'] += scores_rnnt[j, self.blank].item()
                        new_hyps[-1]['update_pred_net'] = False
                        continue

                    total_score = total_scores_topk[0, k].item()
                    total_score_lm = 0
                    if lm is not None:
                        total_score_lm = beam['score_lm'] + beam['next_scores_lm'][0, -1, idx].item()
                        total_score += total_score_lm * lm_weight

                    hyp_ids = beam['hyp'] + [idx]
                    hyp_ids_str = ' '.join(list(map(str, hyp_ids)))
                    exist_cache = hyp_ids_str in self.state_cache.keys()
                    if exist_cache:
                        # from cache
                        dout = self.state_cache[hyp_ids_str]['dout']
                        dstate = self.state_cache[hyp_ids_str]['dstate']
                        scores_lm = self.state_cache[hyp_ids_str]['next_scores_lm']
                        lmstate = self.state_cache[hyp_ids_str]['lmstate']
                    else:
                        # prediction network and LM will be updated later
                        dout = None
                        dstate = beam['dstate']
                        scores_lm = None
                        lmstate = beam['lmstate']

                    new_hyps.append({'hyp': hyp_ids,
                                     'hyp_ids_str': hyp_ids_str,
                                     'score': total_score,
                                     'score_rnnt': total_scores_rnnt[0, idx].item(),
                                     'score_lm': total_score_lm,
                                     'dout': dout,
                                     'dstate': dstate,
                                     'next_scores_lm': scores_lm,
                                     'lmstate': lmstate,
                                     'update_pred_net': not exist_cache})

            # Local pruning
            new_hyps_sorted = sorted(new_hyps, key=lambda x: x['score'], reverse=True)
            new_hyps_sorted = helper.merge_rnnt_path(new_hyps_sorted, merge_prob)
            hyps = new_hyps_sorted[:beam_width]

        return hyps, new_hyps_sorted
