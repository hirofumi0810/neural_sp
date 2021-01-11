# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""CTC decoder."""

from collections import OrderedDict
from distutils.version import LooseVersion
from itertools import groupby
import logging
import numpy as np
import random
import torch
import torch.nn as nn

from neural_sp.models.criterion import kldiv_lsm_ctc
from neural_sp.models.seq2seq.decoders.beam_search import BeamSearch
from neural_sp.models.seq2seq.decoders.decoder_base import DecoderBase
from neural_sp.models.torch_utils import (
    make_pad_mask,
    np2tensor,
    pad_list,
    tensor2np
)

random.seed(1)

# LOG_0 = float(np.finfo(np.float32).min)
LOG_0 = -1e10
LOG_1 = 0

logger = logging.getLogger(__name__)


class CTC(DecoderBase):
    """Connectionist temporal classification (CTC).

    Args:
        eos (int): index for <eos> (shared with <sos>)
        blank (int): index for <blank>
        enc_n_units (int):
        vocab (int): number of nodes in softmax layer
        dropout (float): dropout probability for the RNN layer
        lsm_prob (float): label smoothing probability
        fc_list (List):
        param_init (float): parameter initialization method
        backward (bool): flip the output sequence

    """

    def __init__(self,
                 eos,
                 blank,
                 enc_n_units,
                 vocab,
                 dropout=0.,
                 lsm_prob=0.,
                 fc_list=None,
                 param_init=0.1,
                 backward=False):

        super(CTC, self).__init__()

        self.eos = eos
        self.blank = blank
        self.vocab = vocab
        self.lsm_prob = lsm_prob
        self.bwd = backward

        self.space = -1  # TODO(hirofumi): fix later

        # for posterior plot
        self.prob_dict = {}
        self.data_dict = {}

        # Fully-connected layers before the softmax
        if fc_list is not None and len(fc_list) > 0:
            _fc_list = [int(fc) for fc in fc_list.split('_')]
            fc_layers = OrderedDict()
            for i in range(len(_fc_list)):
                input_dim = enc_n_units if i == 0 else _fc_list[i - 1]
                fc_layers['fc' + str(i)] = nn.Linear(input_dim, _fc_list[i])
                fc_layers['dropout' + str(i)] = nn.Dropout(p=dropout)
            fc_layers['fc' + str(len(_fc_list))] = nn.Linear(_fc_list[-1], vocab)
            self.output = nn.Sequential(fc_layers)
        else:
            self.output = nn.Linear(enc_n_units, vocab)

        self.use_warpctc = LooseVersion(torch.__version__) < LooseVersion("1.4.0")
        if self.use_warpctc:
            import warpctc_pytorch
            self.ctc_loss = warpctc_pytorch.CTCLoss(size_average=True)
        else:
            self.ctc_loss = nn.CTCLoss(reduction="sum")

        self.forced_aligner = CTCForcedAligner()

    def forward(self, eouts, elens, ys, forced_align=False):
        """Compute CTC loss.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (List): length `[B]`
            ys (List): length `[B]`, each of which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[1]`
            trigger_points (IntTensor): `[B, L]`

        """
        # Concatenate all elements in ys for warpctc_pytorch
        ylens = np2tensor(np.fromiter([len(y) for y in ys], dtype=np.int32))
        ys_ctc = torch.cat([np2tensor(np.fromiter(y[::-1] if self.bwd else y, dtype=np.int32))
                            for y in ys], dim=0)
        # NOTE: do not copy to GPUs here

        # Compute CTC loss
        logits = self.output(eouts)
        loss = self.loss_fn(logits.transpose(1, 0), ys_ctc, elens, ylens)

        # Label smoothing for CTC
        if self.lsm_prob > 0:
            loss = loss * (1 - self.lsm_prob) + kldiv_lsm_ctc(logits, elens) * self.lsm_prob

        trigger_points = self.forced_aligner(logits.clone(), elens, ys, ylens) if forced_align else None

        if not self.training:
            self.data_dict['elens'] = tensor2np(elens)
            self.prob_dict['probs'] = tensor2np(torch.softmax(logits, dim=-1))

        return loss, trigger_points

    def loss_fn(self, logits, ys_ctc, elens, ylens):
        if self.use_warpctc:
            loss = self.ctc_loss(logits, ys_ctc, elens.cpu(), ylens).to(logits.device)
            # NOTE: ctc loss has already been normalized by bs
            # NOTE: index 0 is reserved for blank in warpctc_pytorch
        else:
            # Use the deterministic CuDNN implementation of CTC loss to avoid
            #  [issue#17798](https://github.com/pytorch/pytorch/issues/17798)
            with torch.backends.cudnn.flags(deterministic=True):
                loss = self.ctc_loss(logits.log_softmax(2),
                                     ys_ctc, elens, ylens) / logits.size(1)
        return loss

    def trigger_points(self, eouts, elens):
        """Extract trigger points for inference.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
        Returns:
            trigger_points_pred (IntTensor): `[B, L]`

        """
        bs, xmax, _ = eouts.size()
        log_probs = torch.log_softmax(self.output(eouts), dim=-1)
        best_paths = log_probs.argmax(-1)  # `[B, L]`

        hyps = []
        for b in range(bs):
            indices = [best_paths[b, t].item() for t in range(elens[b])]

            # Step 1. Collapse repeated labels
            collapsed_indices = [x[0] for x in groupby(indices)]

            # Step 2. Remove all blank labels
            best_hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
            hyps.append(best_hyp)

        ymax = max([len(h) for h in hyps])

        # pick up trigger points
        trigger_points_pred = log_probs.new_zeros((bs, ymax + 1), dtype=torch.int32)  # +1 for <eos>
        for b in range(bs):
            n_triggers = 0
            for t in range(elens[b]):
                token_idx = best_paths[b, t]

                if token_idx == self.blank:
                    continue
                if not (t == 0 or token_idx != best_paths[b, t - 1]):
                    continue

                # NOTE: select the most left trigger points
                trigger_points_pred[b, n_triggers] = t
                n_triggers += 1

        return trigger_points_pred

    def greedy(self, eouts, elens):
        """Greedy decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (np.ndarray): `[B]`
        Returns:
            hyps (np.ndarray): Best path hypothesis. `[B, L]`

        """
        log_probs = torch.log_softmax(self.output(eouts), dim=-1)
        best_paths = log_probs.argmax(-1)  # `[B, L]`

        hyps = []
        for b in range(eouts.size(0)):
            indices = [best_paths[b, t].item() for t in range(elens[b])]

            # Step 1. Collapse repeated labels
            collapsed_indices = [x[0] for x in groupby(indices)]

            # Step 2. Remove all blank labels
            best_hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
            hyps.append([best_hyp])

        return hyps

    def beam_search(self, eouts, elens, params, idx2token,
                    lm=None, lm_second=None, lm_second_bwd=None,
                    nbest=1, refs_id=None, utt_ids=None, speakers=None):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (List): length `[B]`
            params (dict):
                recog_beam_width (int): size of beam
                recog_length_penalty (float): length penalty
                recog_lm_weight (float): weight of first path LM score
                recog_lm_second_weight (float): weight of second path LM score
                recog_lm_bwd_weight (float): weight of second path backward LM score
            idx2token (): converter from index to token
            lm (torch.nn.module): firsh path LM
            lm_second (torch.nn.module): second path LM
            lm_second_bwd (torch.nn.module): second path backward LM
            nbest (int):
            refs_id (List): reference list
            utt_ids (List): utterance id list
            speakers (List): speaker list
        Returns:
            nbest_hyps_idx (List[List[List]]): Best path hypothesis

        """
        bs = eouts.size(0)

        beam_width = params['recog_beam_width']
        lp_weight = params['recog_length_penalty']
        lm_weight = params['recog_lm_weight']
        lm_weight_second = params['recog_lm_second_weight']
        lm_weight_second_bwd = params['recog_lm_bwd_weight']

        helper = BeamSearch(beam_width, self.eos, 1.0, eouts.device)
        lm = helper.verify_lm_eval_mode(lm, lm_weight)
        lm_second = helper.verify_lm_eval_mode(lm_second, lm_weight_second)
        lm_second_bwd = helper.verify_lm_eval_mode(lm_second_bwd, lm_weight_second_bwd)

        nbest_hyps_idx = []
        log_probs = torch.log_softmax(self.output(eouts), dim=-1)
        for b in range(bs):
            # Elements in the beam are (prefix, (p_b, p_no_blank))
            # Initialize the beam with the empty sequence, a probability of
            # 1 for ending in blank and zero for ending in non-blank (in log space).
            beam = [{'hyp': [self.eos],  # <eos> is used for LM
                     'p_b': LOG_1,
                     'p_nb': LOG_0,
                     'score_lm': LOG_1,
                     'lmstate': None}]

            for t in range(elens[b]):
                new_beam = []

                # Pick up the top-k scores
                log_probs_topk, topk_ids = torch.topk(
                    log_probs[b:b + 1, t], k=min(beam_width, self.vocab), dim=-1, largest=True, sorted=True)

                for i_beam in range(len(beam)):
                    hyp = beam[i_beam]['hyp'][:]
                    p_b = beam[i_beam]['p_b']
                    p_nb = beam[i_beam]['p_nb']
                    score_lm = beam[i_beam]['score_lm']

                    # case 1. hyp is not extended
                    new_p_b = np.logaddexp(p_b + log_probs[b, t, self.blank].item(),
                                           p_nb + log_probs[b, t, self.blank].item())
                    if len(hyp) > 1:
                        new_p_nb = p_nb + log_probs[b, t, hyp[-1]].item()
                    else:
                        new_p_nb = LOG_0
                    score_ctc = np.logaddexp(new_p_b, new_p_nb)
                    score_lp = len(hyp[1:]) * lp_weight
                    new_beam.append({'hyp': hyp,
                                     'score': score_ctc + score_lm + score_lp,
                                     'p_b': new_p_b,
                                     'p_nb': new_p_nb,
                                     'score_ctc': score_ctc,
                                     'score_lm': score_lm,
                                     'score_lp': score_lp,
                                     'lmstate': beam[i_beam]['lmstate']})

                    # Update LM states for shallow fusion
                    if lm is not None:
                        _, lmstate, lm_log_probs = lm.predict(
                            eouts.new_zeros(1, 1).fill_(hyp[-1]), beam[i_beam]['lmstate'])
                    else:
                        lmstate = None

                    # case 2. hyp is extended
                    new_p_b = LOG_0
                    for c in tensor2np(topk_ids)[0]:
                        p_t = log_probs[b, t, c].item()

                        if c == self.blank:
                            continue

                        c_prev = hyp[-1] if len(hyp) > 1 else None
                        if c == c_prev:
                            new_p_nb = p_b + p_t
                            # TODO(hirofumi): apply character LM here
                        else:
                            new_p_nb = np.logaddexp(p_b + p_t, p_nb + p_t)
                            # TODO(hirofumi): apply character LM here
                            if c == self.space:
                                pass
                                # TODO(hirofumi): apply word LM here

                        score_ctc = np.logaddexp(new_p_b, new_p_nb)
                        score_lp = (len(hyp[1:]) + 1) * lp_weight
                        if lm_weight > 0 and lm is not None:
                            local_score_lm = lm_log_probs[0, 0, c].item() * lm_weight
                            score_lm += local_score_lm
                        new_beam.append({'hyp': hyp + [c],
                                         'score': score_ctc + score_lm + score_lp,
                                         'p_b': new_p_b,
                                         'p_nb': new_p_nb,
                                         'score_ctc': score_ctc,
                                         'score_lm': score_lm,
                                         'score_lp': score_lp,
                                         'lmstate': lmstate})

                # Pruning
                beam = sorted(new_beam, key=lambda x: x['score'], reverse=True)[:beam_width]

            # forward second path LM rescoring
            helper.lm_rescoring(beam, lm_second, lm_weight_second, tag='second')

            # backward secodn path LM rescoring
            helper.lm_rescoring(beam, lm_second_bwd, lm_weight_second_bwd, tag='second_bwd')

            # Exclude <eos>
            nbest_hyps_idx.append([hyp['hyp'][1:] for hyp in beam])

            if idx2token is not None:
                if utt_ids is not None:
                    logger.info('Utt-id: %s' % utt_ids[b])
                assert self.vocab == idx2token.vocab
                logger.info('=' * 200)
                for k in range(len(beam)):
                    if refs_id is not None:
                        logger.info('Ref: %s' % idx2token(refs_id[b]))
                    logger.info('Hyp: %s' % idx2token(beam[k]['hyp'][1:]))
                    logger.info('log prob (hyp): %.7f' % beam[k]['score'])
                    logger.info('log prob (hyp, ctc): %.7f' % (beam[k]['score_ctc']))
                    logger.info('log prob (hyp, lp): %.7f' % (beam[k]['score_lp'] * lp_weight))
                    if lm is not None:
                        logger.info('log prob (hyp, first-path lm): %.7f' %
                                    (beam[k]['score_lm'] * lm_weight))
                    if lm_second is not None:
                        logger.info('log prob (hyp, second-path lm): %.7f' %
                                    (beam[k]['score_lm_second'] * lm_weight_second))
                    logger.info('-' * 50)

        return nbest_hyps_idx


def _label_to_path(labels, blank):
    path = labels.new_zeros(labels.size(0), labels.size(1) * 2 + 1).fill_(blank).long()
    path[:, 1::2] = labels
    return path


def _flip_path(path, path_lens):
    """Flips label sequence.
    This function rotates a label sequence and flips it.
    ``path[b, t]`` stores a label at time ``t`` in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[b, t] = path[b, t + path_lens[b]]``
    .. ::
       a b c d .     . a b c d    d c b a .
       e f . . .  -> . . . e f -> f e . . .
       g h i j k     g h i j k    k j i h g

    Args:
        path (FloatTensor): `[B, 2*L+1]`
        path_lens (LongTensor): `[B]`
    Returns:
        FloatTensor: `[B, 2*L+1]`

    """
    bs = path.size(0)
    max_path_len = path.size(1)
    rotate = (torch.arange(max_path_len) + path_lens[:, None]) % max_path_len
    return torch.flip(path[torch.arange(bs, dtype=torch.int64)[:, None], rotate], dims=[1])


def _flip_label_probability(log_probs, xlens):
    """Flips a label probability matrix.
    This function rotates a label probability matrix and flips it.
    ``log_probs[i, b, l]`` stores log probability of label ``l`` at ``i``-th
    input in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[i, b, l] = log_probs[i + xlens[b], b, l]``

    Args:
        cum_log_prob (FloatTensor): `[T, B, vocab]`
        xlens (LongTensor): `[B]`
    Returns:
        FloatTensor: `[T, B, vocab]`

    """
    xmax, bs, vocab = log_probs.size()
    rotate = (torch.arange(xmax, dtype=torch.int64)[:, None] + xlens) % xmax
    return torch.flip(log_probs[rotate[:, :, None],
                                torch.arange(bs, dtype=torch.int64)[None, :, None],
                                torch.arange(vocab, dtype=torch.int64)[None, None, :]], dims=[0])


def _flip_path_probability(cum_log_prob, xlens, path_lens):
    """Flips a path probability matrix.
    This function returns a path probability matrix and flips it.
    ``cum_log_prob[i, b, t]`` stores log probability at ``i``-th input and
    at time ``t`` in a output sequence in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[i, j, k] = cum_log_prob[i + xlens[j], j, k + path_lens[j]]``

    Args:
        cum_log_prob (FloatTensor): `[T, B, 2*L+1]`
        xlens (LongTensor): `[B]`
        path_lens (LongTensor): `[B]`
    Returns:
        FloatTensor: `[T, B, 2*L+1]`

    """
    xmax, bs, max_path_len = cum_log_prob.size()
    rotate_input = ((torch.arange(xmax, dtype=torch.int64)[:, None] + xlens) % xmax)
    rotate_label = ((torch.arange(max_path_len, dtype=torch.int64) + path_lens[:, None]) % max_path_len)
    return torch.flip(cum_log_prob[rotate_input[:, :, None],
                                   torch.arange(bs, dtype=torch.int64)[None, :, None],
                                   rotate_label], dims=[0, 2])


class CTCForcedAligner(object):
    def __init__(self, blank=0):
        self.blank = blank
        self.log0 = LOG_0

    def __call__(self, logits, elens, ys, ylens):
        """Forced alignment with references.

        Args:
            logits (FloatTensor): `[B, T, vocab]`
            elens (List): length `[B]`
            ys (List): length `[B]`, each of which contains a list of size `[L]`
            ylens (List): length `[B]`
        Returns:
            trigger_points (IntTensor): `[B, L]`

        """
        with torch.no_grad():
            ys = [np2tensor(np.fromiter(y, dtype=np.int64), logits.device) for y in ys]
            ys_in_pad = pad_list(ys, 0)

            # zero padding
            mask = make_pad_mask(elens.to(logits.device))
            mask = mask.unsqueeze(2).expand_as(logits)
            logits = logits.masked_fill_(mask == 0, self.log0)
            log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)  # `[T, B, vocab]`

            trigger_points = self.align(log_probs, elens, ys_in_pad, ylens)
        return trigger_points

    def _computes_transition(self, prev_log_prob, path, path_lens, cum_log_prob, y, skip_accum=False):
        bs, max_path_len = path.size()
        mat = prev_log_prob.new_zeros(3, bs, max_path_len).fill_(self.log0)
        mat[0, :, :] = prev_log_prob
        mat[1, :, 1:] = prev_log_prob[:, :-1]
        mat[2, :, 2:] = prev_log_prob[:, :-2]
        # disable transition between the same symbols
        # (including blank-to-blank)
        same_transition = (path[:, :-2] == path[:, 2:])
        mat[2, :, 2:][same_transition] = self.log0
        log_prob = torch.logsumexp(mat, dim=0)
        outside = torch.arange(max_path_len, dtype=torch.int64) >= path_lens.unsqueeze(1)
        log_prob[outside] = self.log0
        if not skip_accum:
            cum_log_prob += log_prob
        batch_index = torch.arange(bs, dtype=torch.int64).unsqueeze(1)
        log_prob += y[batch_index, path]
        return log_prob

    def align(self, log_probs, elens, ys, ylens, add_eos=True):
        """Calculte the best CTC alignment with the forward-backward algorithm.
        Args:
            log_probs (FloatTensor): `[T, B, vocab]`
            elens (FloatTensor): `[B]`
            ys (FloatTensor): `[B, L]`
            ylens (FloatTensor): `[B]`
            add_eos (bool): Use the last time index as a boundary corresponding to <eos>
        Returns:
            trigger_points (IntTensor): `[B, L]`

        """
        xmax, bs, vocab = log_probs.size()

        path = _label_to_path(ys, self.blank)
        path_lens = 2 * ylens.long() + 1

        ymax = ys.size(1)
        max_path_len = path.size(1)
        assert ys.size() == (bs, ymax), ys.size()
        assert path.size() == (bs, ymax * 2 + 1)

        alpha = log_probs.new_zeros(bs, max_path_len).fill_(self.log0)
        alpha[:, 0] = LOG_1
        beta = alpha.clone()
        gamma = alpha.clone()

        batch_index = torch.arange(bs, dtype=torch.int64).unsqueeze(1)
        seq_index = torch.arange(xmax, dtype=torch.int64).unsqueeze(1).unsqueeze(2)
        log_probs_fwd_bwd = log_probs[seq_index, batch_index, path]

        # forward algorithm
        for t in range(xmax):
            alpha = self._computes_transition(alpha, path, path_lens, log_probs_fwd_bwd[t], log_probs[t])

        # backward algorithm
        r_path = _flip_path(path, path_lens)
        log_probs_inv = _flip_label_probability(log_probs, elens.long())  # `[T, B, vocab]`
        log_probs_fwd_bwd = _flip_path_probability(log_probs_fwd_bwd, elens.long(), path_lens)  # `[T, B, 2*L+1]`
        for t in range(xmax):
            beta = self._computes_transition(beta, r_path, path_lens, log_probs_fwd_bwd[t], log_probs_inv[t])

        # pick up the best CTC path
        best_aligns = log_probs.new_zeros((bs, xmax), dtype=torch.int64)

        # forward algorithm
        log_probs_fwd_bwd = _flip_path_probability(log_probs_fwd_bwd, elens.long(), path_lens)
        for t in range(xmax):
            gamma = self._computes_transition(gamma, path, path_lens, log_probs_fwd_bwd[t], log_probs[t],
                                              skip_accum=True)

            # select paths where gamma is valid
            log_probs_fwd_bwd[t] = log_probs_fwd_bwd[t].masked_fill_(gamma == self.log0, self.log0)

            # pick up the best alignment
            offsets = log_probs_fwd_bwd[t].argmax(1)
            for b in range(bs):
                if t <= elens[b] - 1:
                    token_idx = path[b, offsets[b]]
                    best_aligns[b, t] = token_idx

            # remove the rest of paths
            gamma = log_probs.new_zeros(bs, max_path_len).fill_(self.log0)
            for b in range(bs):
                gamma[b, offsets[b]] = LOG_1

        # pick up trigger points
        trigger_aligns = torch.zeros((bs, xmax), dtype=torch.int64)
        trigger_points = log_probs.new_zeros((bs, ymax + 1), dtype=torch.int32)  # +1 for <eos>
        for b in range(bs):
            n_triggers = 0
            if add_eos:
                trigger_points[b, ylens[b]] = elens[b] - 1
                # NOTE: use the last time index as a boundary corresponding to <eos>
                # Otherwise, index: 0 is used for <eos>
            for t in range(elens[b]):
                token_idx = best_aligns[b, t]
                if token_idx == self.blank:
                    continue
                if not (t == 0 or token_idx != best_aligns[b, t - 1]):
                    continue

                # NOTE: select the most left trigger points
                trigger_aligns[b, t] = token_idx
                trigger_points[b, n_triggers] = t
                n_triggers += 1

        assert ylens.sum() == (trigger_aligns != 0).sum()
        return trigger_points


class CTCPrefixScore(object):
    """Compute CTC label sequence scores.

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the probabilities of multiple labels
    simultaneously

    [Reference]:
        https://github.com/espnet/espnet
    """

    def __init__(self, log_probs, blank, eos, truncate=False):
        """
        Args:
            log_probs (np.ndarray):
            blank (int): index of <blank>
            eos (int): index of <eos>
            truncate (bool): restart prefix search from the previous CTC spike

        """
        self.blank = blank
        self.eos = eos
        self.xlen_prev = 0
        self.xlen = len(log_probs)
        self.log_probs = log_probs
        self.log0 = LOG_0

        self.truncate = truncate
        self.offset = 0

    def initial_state(self):
        """Obtain an initial CTC state

        Returns:
            ctc_states (np.ndarray): `[T, 2]`

        """
        # initial CTC state is made of a frame x 2 tensor that corresponds to
        # r_t^n(<sos>) and r_t^b(<sos>), where 0 and 1 of axis=1 represent
        # superscripts n and b (non-blank and blank), respectively.
        r = np.full((self.xlen, 2), self.log0, dtype=np.float32)
        r[0, 1] = self.log_probs[0, self.blank]
        for i in range(1, self.xlen):
            r[i, 1] = r[i - 1, 1] + self.log_probs[i, self.blank]
        return r

    def register_new_chunk(self, log_probs_chunk):
        self.xlen_prev = self.xlen
        self.log_probs = np.concatenate([self.log_probs, log_probs_chunk], axis=0)
        self.xlen = len(self.log_probs)

    def __call__(self, hyp, cs, r_prev, new_chunk=False):
        """Compute CTC prefix scores for next labels.

        Args:
            hyp (List): prefix label sequence
            cs (np.ndarray): array of next labels. A tensor of size `[beam_width]`
            r_prev (np.ndarray): previous CTC state `[T, 2]`
        Returns:
            ctc_scores (np.ndarray): `[beam_width]`
            ctc_states (np.ndarray): `[beam_width, T, 2]`

        """
        beam_width = len(cs)

        # initialize CTC states
        ylen = len(hyp) - 1  # ignore sos
        # new CTC states are prepared as a frame x (n or b) x n_labels tensor
        # that corresponds to r_t^n(h) and r_t^b(h).
        r = np.ndarray((self.xlen, 2, beam_width), dtype=np.float32)
        xs = self.log_probs[:, cs]
        if ylen == 0:
            r[0, 0] = xs[0]
            r[0, 1] = self.log0
        else:
            r[ylen - 1] = self.log0

        # Initialize CTC state for the new chunk
        if new_chunk and self.xlen_prev > 0:
            xlen_prev = r_prev.shape[0]
            r_new = np.full((self.xlen - xlen_prev, 2), self.log0, dtype=np.float32)
            r_new[0, 1] = r_prev[xlen_prev - 1, 1] + self.log_probs[xlen_prev, self.blank]
            for i in range(xlen_prev + 1, self.xlen):
                r_new[i - xlen_prev, 1] = r_new[i - xlen_prev - 1, 1] + self.log_probs[i, self.blank]
            r_prev = np.concatenate([r_prev, r_new], axis=0)

        # prepare forward probabilities for the last label
        r_sum = np.logaddexp(r_prev[:, 0], r_prev[:, 1])  # log(r_t^n(g) + r_t^b(g))
        last = hyp[-1]
        if ylen > 0 and last in cs:
            log_phi = np.ndarray((self.xlen, beam_width), dtype=np.float32)
            for k in range(beam_width):
                log_phi[:, k] = r_sum if cs[k] != last else r_prev[:, 1]
        else:
            log_phi = r_sum  # `[T]`

        # compute forward probabilities log(r_t^n(h)), log(r_t^b(h)),
        # and log prefix probabilities log(psi)
        start = max(ylen, 1)
        log_psi = r[start - 1, 0]
        for t in range(start, self.xlen):
            # non-blank
            r[t, 0] = np.logaddexp(r[t - 1, 0], log_phi[t - 1]) + xs[t]
            # blank
            r[t, 1] = np.logaddexp(r[t - 1, 0], r[t - 1, 1]) + self.log_probs[t, self.blank]
            log_psi = np.logaddexp(log_psi, log_phi[t - 1] + xs[t])

        # get P(...eos|X) that ends with the prefix itself
        eos_pos = np.where(cs == self.eos)[0]
        if len(eos_pos) > 0:
            log_psi[eos_pos] = r_sum[-1]  # log(r_T^n(g) + r_T^b(g))

        # return the log prefix probability and CTC states, where the label axis
        # of the CTC states is moved to the first axis to slice it easily
        return log_psi, np.rollaxis(r, 2)
