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
from neural_sp.models.lm.rnnlm import RNNLM
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

        # for cache
        self.prev_spk = ''
        self.lmstate_final = None

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

    def probs(self, eouts, temperature=1.):
        """Get CTC probabilities.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
        Returns:
            probs (FloatTensor): `[B, T, vocab]`

        """
        return torch.softmax(self.output(eouts) / temperature, dim=-1)

    def scores(self, eouts, temperature=1.):
        """Get log-scale CTC probabilities.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
        Returns:
            log_probs (FloatTensor): `[B, T, vocab]`

        """
        return torch.log_softmax(self.output(eouts) / temperature, dim=-1)

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

    def initialize_beam(self, hyp, lmstate):
        """Initialize beam."""
        hyps = [{'hyp': hyp,
                 'hyp_ids_str': '',
                 'p_b': LOG_1,
                 'p_nb': LOG_0,
                 'score_lm': LOG_1,
                 'lmstate': lmstate,
                 'update_lm': True}]
        return hyps

    def beam_search(self, eouts, elens, params, idx2token,
                    lm=None, lm_second=None, lm_second_bwd=None,
                    nbest=1, refs_id=None, utt_ids=None, speakers=None):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (List): length `[B]`
            params (dict): decoding hyperparameters
            idx2token (): converter from index to token
            lm (torch.nn.module): firsh-pass LM
            lm_second (torch.nn.module): second-pass LM
            lm_second_bwd (torch.nn.module): second-pass backward LM
            nbest (int): number of N-best list
            refs_id (List): reference list
            utt_ids (List): utterance id list
            speakers (List): speaker list
        Returns:
            nbest_hyps_idx (List[List[List]]): Best path hypothesis

        """
        bs = eouts.size(0)

        beam_width = params.get('recog_beam_width')
        lp_weight = params.get('recog_length_penalty')
        cache_emb = params.get('recog_cache_embedding')
        lm_weight = params.get('recog_lm_weight')
        lm_weight_second = params.get('recog_lm_second_weight')
        lm_weight_second_bwd = params.get('recog_lm_bwd_weight')
        lm_state_CO = params.get('recog_lm_state_carry_over')
        softmax_smoothing = params.get('recog_softmax_smoothing')

        helper = BeamSearch(beam_width, self.eos, 1.0, lm_weight, eouts.device)
        lm = helper.verify_lm_eval_mode(lm, lm_weight, cache_emb)
        if lm is not None:
            assert isinstance(lm, RNNLM)
        lm_second = helper.verify_lm_eval_mode(lm_second, lm_weight_second, cache_emb)
        lm_second_bwd = helper.verify_lm_eval_mode(lm_second_bwd, lm_weight_second_bwd, cache_emb)

        log_probs = torch.log_softmax(self.output(eouts) * softmax_smoothing, dim=-1)

        nbest_hyps_idx = []
        for b in range(bs):
            # Initialization per utterance
            lmstate = {'hxs': eouts.new_zeros(lm.n_layers, 1, lm.n_units),
                       'cxs': eouts.new_zeros(lm.n_layers, 1, lm.n_units)} if lm is not None else None

            if speakers is not None:
                if speakers[b] == self.prev_spk:
                    if lm_state_CO:
                        lmstate = self.lmstate_final
                self.prev_spk = speakers[b]

            hyps = self.initialize_beam([self.eos], lmstate)
            self.state_cache = OrderedDict()

            hyps, new_hyps = self._beam_search(hyps, helper, log_probs[b], lm, lp_weight)

            # Global pruning
            end_hyps = hyps[:]
            if len(end_hyps) < nbest and nbest > 1:
                end_hyps.extend(new_hyps[:nbest - len(end_hyps)])

            # forward/backward second-pass LM rescoring
            end_hyps = helper.lm_rescoring(end_hyps, lm_second, lm_weight_second, tag='second')
            end_hyps = helper.lm_rescoring(end_hyps, lm_second_bwd, lm_weight_second_bwd, tag='second_bwd')

            # Normalize by length
            end_hyps = sorted(end_hyps, key=lambda x: x['score'] / max(len(x['hyp'][1:]), 1), reverse=True)

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
                    logger.info('log prob (hyp, ctc): %.7f' % (end_hyps[k]['score_ctc']))
                    logger.info('log prob (hyp, lp): %.7f' % (end_hyps[k]['score_lp'] * lp_weight))
                    if lm is not None:
                        logger.info('log prob (hyp, first-pass lm): %.7f' %
                                    (end_hyps[k]['score_lm'] * lm_weight))
                    if lm_second is not None:
                        logger.info('log prob (hyp, second-pass lm): %.7f' %
                                    (end_hyps[k]['score_lm_second'] * lm_weight_second))
                    if lm_second_bwd is not None:
                        logger.info('log prob (hyp, second-pass lm, reverse): %.7f' %
                                    (end_hyps[k]['score_lm_second_bwd'] * lm_weight_second_bwd))
                    logger.info('-' * 50)

            # N-best list (exclude <eos>)
            nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:]) for n in range(nbest)]]

        # Store LM state
        if bs == 1:
            self.lmstate_final = end_hyps[0]['lmstate']

        return nbest_hyps_idx

    def _beam_search(self, hyps, helper, scores_ctc, lm, lp_weight):
        beam_width = helper.beam_width
        lm_weight = helper.lm_weight
        merge_prob = True

        # scores_ctc: `[T, vocab]`
        for t in range(scores_ctc.size(0)):
            # Pick up the top-k scores
            _, topk_ids = torch.topk(
                scores_ctc[t, 1:],  # exclude blank
                k=min(beam_width, self.vocab), dim=-1, largest=True, sorted=True)
            topk_ids += 1  # index:0 is for blank

            # bachfy all hypotheses (not in the cache, non-blank) for LM
            batch_hyps = [beam for beam in hyps if beam['update_lm']]
            if len(batch_hyps) > 0:
                ys = scores_ctc.new_zeros((len(batch_hyps), 1), dtype=torch.int64)
                for i, beam in enumerate(batch_hyps):
                    ys[i] = beam['hyp'][-1]

                # Update LM states for shallow fusion
                _, lmstates, scores_lm = helper.update_rnnlm_state_batch(lm, batch_hyps, ys)

                hyp_ids_strs = [beam['hyp_ids_str'] for beam in hyps]

                for i, beam in enumerate(batch_hyps):
                    lmstate = {'hxs': lmstates['hxs'][:, i:i + 1],
                               'cxs': lmstates['cxs'][:, i:i + 1]} if lmstates is not None else None
                    index = hyp_ids_strs.index(beam['hyp_ids_str'])

                    hyps[index]['lmstate'] = lmstate
                    if lm is not None:
                        hyps[index]['next_scores_lm'] = scores_lm[i:i + 1]
                    else:
                        hyps[index]['next_scores_lm'] = None
                    assert hyps[index]['update_lm']
                    hyps[index]['update_lm'] = False

                    # register to cache
                    self.state_cache[beam['hyp_ids_str']] = {
                        'next_scores_lm': hyps[index]['next_scores_lm'],
                        'lmstate': lmstate,
                    }

            new_hyps = []
            for j, beam in enumerate(hyps):
                p_b = beam['p_b']
                p_nb = beam['p_nb']
                total_score_lm = beam['score_lm']

                # case 1. hyp is not extended
                new_p_b = np.logaddexp(p_b + scores_ctc[t, self.blank].item(),
                                       p_nb + scores_ctc[t, self.blank].item())
                if len(beam['hyp'][1:]) > 0:
                    new_p_nb = p_nb + scores_ctc[t, beam['hyp'][-1]].item()
                else:
                    new_p_nb = LOG_0
                total_score_ctc = np.logaddexp(new_p_b, new_p_nb)
                total_score_lp = len(beam['hyp'][1:]) * lp_weight
                total_score = total_score_ctc + total_score_lp + total_score_lm * lm_weight
                new_hyps.append({'hyp': beam['hyp'][:],
                                 'hyp_ids_str': beam['hyp_ids_str'],
                                 'score': total_score,
                                 'p_b': new_p_b,
                                 'p_nb': new_p_nb,
                                 'score_ctc': total_score_ctc,
                                 'score_lm': total_score_lm,
                                 'score_lp': total_score_lp,
                                 'next_scores_lm': beam['next_scores_lm'],
                                 'lmstate': beam['lmstate'],
                                 'update_lm': False})

                # case 2. hyp is extended
                new_p_b = LOG_0
                for k in range(beam_width):
                    idx = topk_ids[k].item()
                    p_t = scores_ctc[t, idx].item()

                    c_prev = beam['hyp'][-1] if len(beam['hyp']) > 1 else None
                    if idx == c_prev:
                        new_p_nb = p_b + p_t
                        # TODO(hirofumi): apply character LM here
                    else:
                        new_p_nb = np.logaddexp(p_b + p_t, p_nb + p_t)
                        # TODO(hirofumi): apply character LM here
                        if idx == self.space:
                            pass
                            # TODO(hirofumi): apply word LM here

                    total_score_ctc = np.logaddexp(new_p_b, new_p_nb)
                    total_score_lp = (len(beam['hyp'][1:]) + 1) * lp_weight
                    total_score = total_score_ctc + total_score_lp
                    if lm is not None:
                        total_score_lm += beam['next_scores_lm'][0, 0, idx].item()
                    total_score += total_score_lm * lm_weight

                    hyp_ids = beam['hyp'] + [idx]
                    hyp_ids_str = ' '.join(list(map(str, hyp_ids)))
                    exist_cache = hyp_ids_str in self.state_cache.keys()
                    if exist_cache:
                        # from cache
                        scores_lm = self.state_cache[hyp_ids_str]['next_scores_lm']
                        lmstate = self.state_cache[hyp_ids_str]['lmstate']
                    else:
                        # LM will be updated later
                        scores_lm = None
                        lmstate = beam['lmstate']

                    new_hyps.append({'hyp': hyp_ids,
                                     'hyp_ids_str': hyp_ids_str,
                                     'score': total_score,
                                     'p_b': new_p_b,
                                     'p_nb': new_p_nb,
                                     'score_ctc': total_score_ctc,
                                     'score_lm': total_score_lm,
                                     'score_lp': total_score_lp,
                                     'next_scores_lm': scores_lm,
                                     'lmstate': lmstate,
                                     'update_lm': not exist_cache})

            # Pruning
            new_hyps = sorted(new_hyps, key=lambda x: x['score'], reverse=True)
            new_hyps = helper.merge_ctc_path(new_hyps, merge_prob)
            hyps = new_hyps[:beam_width]

        return hyps, new_hyps

    def beam_search_block_sync(self, eouts, params, helper, idx2token, hyps, lm):
        assert eouts.size(0) == 1

        beam_width = params.get('recog_beam_width')
        lp_weight = params.get('recog_length_penalty')
        lm_weight = params.get('recog_lm_weight')
        lm_state_CO = params.get('recog_lm_state_carry_over')
        softmax_smoothing = params.get('recog_softmax_smoothing')

        end_hyps = []
        if hyps is None:
            # Initialization per utterance
            if lm_state_CO:
                lmstate = self.lmstate_final
            else:
                lmstate = {'hxs': eouts.new_zeros(lm.n_layers, 1, lm.n_units),
                           'cxs': eouts.new_zeros(lm.n_layers, 1, lm.n_units)} if lm is not None else None

            self.n_frames = 0
            hyps = self.initialize_beam([self.eos], lmstate)
            self.state_cache = OrderedDict()

        log_probs = torch.log_softmax(self.output(eouts) * softmax_smoothing, dim=-1)
        hyps, _ = self._beam_search(hyps, helper, log_probs[0], lm, lp_weight)

        # merged_hyps = sorted(end_hyps + hyps, key=lambda x: x['score'] / len(x['hyp']), reverse=True)[:beam_width]
        merged_hyps = sorted(end_hyps + hyps, key=lambda x: x['score'], reverse=True)[:beam_width]
        if idx2token is not None:
            logger.info('=' * 200)
            for k in range(len(merged_hyps)):
                logger.info('Hyp: %s' % idx2token(merged_hyps[k]['hyp'][1:]))
                if len(merged_hyps[k]['hyp']) > 1:
                    logger.info('num tokens (hyp): %d' % len(merged_hyps[k]['hyp'][1:]))
                logger.info('log prob (hyp): %.7f' % merged_hyps[k]['score'])
                logger.info('log prob (hyp, ctc): %.7f' % merged_hyps[k]['score_ctc'])
                if lm is not None:
                    logger.info('log prob (hyp, first-pass lm): %.7f' %
                                (merged_hyps[k]['score_lm'] * lm_weight))
                logger.info('-' * 50)

        # Store LM state
        if len(merged_hyps) > 0:
            self.lmstate_final = merged_hyps[0]['lmstate']

        self.n_frames += eouts.size(1)

        return end_hyps, hyps


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


def _computes_transition(seq_log_prob, same_transition, outside,
                         cum_log_prob, log_prob_yt, skip_accum=False):
    bs, max_path_len = seq_log_prob.size()
    mat = seq_log_prob.new_zeros(3, bs, max_path_len).fill_(LOG_0)
    mat[0, :, :] = seq_log_prob
    mat[1, :, 1:] = seq_log_prob[:, :-1]
    mat[2, :, 2:] = seq_log_prob[:, :-2]
    # disable transition between the same symbols
    # (including blank-to-blank)
    mat[2, :, 2:][same_transition] = LOG_0
    seq_log_prob = torch.logsumexp(mat, dim=0)  # overwrite
    seq_log_prob[outside] = LOG_0
    if not skip_accum:
        cum_log_prob += seq_log_prob
    seq_log_prob += log_prob_yt
    return seq_log_prob


class CTCForcedAligner(object):
    def __init__(self, blank=0):
        self.blank = blank

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
            logits = logits.masked_fill_(mask == 0, LOG_0)
            log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)  # `[T, B, vocab]`

            trigger_points = self.align(log_probs, elens, ys_in_pad, ylens)
        return trigger_points

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

        alpha = log_probs.new_zeros(bs, max_path_len).fill_(LOG_0)
        alpha[:, 0] = LOG_1
        beta = alpha.clone()
        gamma = alpha.clone()

        batch_index = torch.arange(bs, dtype=torch.int64).unsqueeze(1)
        frame_index = torch.arange(xmax, dtype=torch.int64).unsqueeze(1).unsqueeze(2)
        log_probs_fwd_bwd = log_probs[frame_index, batch_index, path]
        same_transition = (path[:, :-2] == path[:, 2:])
        outside = torch.arange(max_path_len, dtype=torch.int64) >= path_lens.unsqueeze(1)
        log_probs_gold = log_probs[:, batch_index, path]

        # forward algorithm
        for t in range(xmax):
            alpha = _computes_transition(alpha, same_transition, outside,
                                         log_probs_fwd_bwd[t], log_probs_gold[t])

        # backward algorithm
        r_path = _flip_path(path, path_lens)
        log_probs_inv = _flip_label_probability(log_probs, elens.long())  # `[T, B, vocab]`
        log_probs_fwd_bwd = _flip_path_probability(log_probs_fwd_bwd, elens.long(), path_lens)  # `[T, B, 2*L+1]`
        r_same_transition = (r_path[:, :-2] == r_path[:, 2:])
        log_probs_inv_gold = log_probs_inv[:, batch_index, r_path]
        for t in range(xmax):
            beta = _computes_transition(beta, r_same_transition, outside,
                                        log_probs_fwd_bwd[t], log_probs_inv_gold[t])

        # pick up the best CTC path
        best_aligns = log_probs.new_zeros((bs, xmax), dtype=torch.int64)

        # forward algorithm
        log_probs_fwd_bwd = _flip_path_probability(log_probs_fwd_bwd, elens.long(), path_lens)
        for t in range(xmax):
            gamma = _computes_transition(gamma, same_transition, outside,
                                         log_probs_fwd_bwd[t], log_probs_gold[t],
                                         skip_accum=True)

            # select paths where gamma is valid
            log_probs_fwd_bwd[t] = log_probs_fwd_bwd[t].masked_fill_(gamma == LOG_0, LOG_0)

            # pick up the best alignment
            offsets = log_probs_fwd_bwd[t].argmax(1)
            for b in range(bs):
                if t <= elens[b] - 1:
                    token_idx = path[b, offsets[b]]
                    best_aligns[b, t] = token_idx

            # remove the rest of paths
            gamma = log_probs.new_zeros(bs, max_path_len).fill_(LOG_0)
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
