#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Forward-backward attention decoding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

logger = logging.getLogger("decoding")


def fwd_bwd_attention(nbest_hyps_fwd, aws_fwd, scores_fwd,
                      nbest_hyps_bwd, aws_bwd, scores_bwd,
                      flip, eos, gnmt_decoding, lp_weight, idx2token, refs_id=None):
    """Decoding with the forward and backward attention-based decoders.

    Args:
        nbest_hyps_fwd (list): A list of length `[B]`, which contains list of n hypotheses
        aws_fwd (list): A list of length `[B]`, which contains arrays of size `[L, T]`
        scores_fwd (list):
        nbest_hyps_bwd (list):
        aws_bwd (list):
        scores_bwd (list):
        flip (bool):
        eos (int):
        gnmt_decoding ():
        lp_weight ():
        idx2token (): converter from index to token
        refs_id ():
    Returns:

    """
    bs = len(nbest_hyps_fwd)
    nbest = len(nbest_hyps_fwd[0])

    best_hyps = []
    for b in range(bs):
        max_time = len(aws_fwd[b][0])

        merged = []
        for n in range(nbest):
            # forward
            if len(nbest_hyps_fwd[b][n]) > 1:
                if nbest_hyps_fwd[b][n][-1] == eos:
                    merged.append({'hyp': nbest_hyps_fwd[b][n][:-1],
                                   'score': scores_fwd[b][n][-2]})
                    # NOTE: remove eos probability
                else:
                    merged.append({'hyp': nbest_hyps_fwd[b][n],
                                   'score': scores_fwd[b][n][-1]})
            else:
                # <eos> only
                logger.info(nbest_hyps_fwd[b][n])

            # backward
            if len(nbest_hyps_bwd[b][n]) > 1:
                if nbest_hyps_bwd[b][n][0] == eos:
                    merged.append({'hyp': nbest_hyps_bwd[b][n][1:],
                                   'score': scores_bwd[b][n][1]})
                    # NOTE: remove eos probability
                else:
                    merged.append({'hyp': nbest_hyps_bwd[b][n],
                                   'score': scores_bwd[b][n][0]})
            else:
                # <eos> only
                logger.info(nbest_hyps_bwd[b][n])

        for n_f in range(nbest):
            for n_b in range(nbest):
                for i_f in range(len(aws_fwd[b][n_f]) - 1):
                    for i_b in range(len(aws_bwd[b][n_b]) - 1):
                        if flip:
                            t_prev = max_time - aws_bwd[b][n_b][i_b + 1].argmax(-2)
                            t_curr = aws_fwd[b][n_f][i_f].argmax(-2)
                            t_next = max_time - aws_bwd[b][n_b][i_b - 1].argmax(-2)
                        else:
                            t_prev = aws_bwd[b][n_b][i_b + 1].argmax(-2)
                            t_curr = aws_fwd[b][n_f][i_f].argmax(-2)
                            t_next = aws_bwd[b][n_b][i_b - 1].argmax(-2)

                        # the same token at the same time
                        if t_curr >= t_prev and t_curr <= t_next and nbest_hyps_fwd[b][n_f][i_f] == nbest_hyps_bwd[b][n_b][i_b]:
                            new_hyp = nbest_hyps_fwd[b][n_f][:i_f + 1].tolist() + \
                                nbest_hyps_bwd[b][n_b][i_b + 1:].tolist()
                            score_curr_fwd = scores_fwd[b][n_f][i_f] - scores_fwd[b][n_f][i_f - 1]
                            score_curr_bwd = scores_bwd[b][n_b][i_b] - scores_bwd[b][n_b][i_b + 1]
                            score_curr = max(score_curr_fwd, score_curr_bwd)
                            new_score = scores_fwd[b][n_f][i_f - 1] + scores_bwd[b][n_b][i_b + 1] + score_curr
                            merged.append({'hyp': new_hyp, 'score': new_score})

                            logger.info('time matching')
                            if refs_id is not None:
                                logger.info('Ref: %s' % idx2token(refs_id[b]))
                            logger.info('hyp (fwd): %s' % idx2token(nbest_hyps_fwd[b][n_f]))
                            logger.info('hyp (bwd): %s' % idx2token(nbest_hyps_bwd[b][n_b]))
                            logger.info('hyp (fwd-bwd): %s' % idx2token(new_hyp))
                            logger.info('log prob (fwd): %.3f' % scores_fwd[b][n_f][-1])
                            logger.info('log prob (bwd): %.3f' % scores_bwd[b][n_b][0])
                            logger.info('log prob (fwd-bwd): %.3f' % new_score)

        merged = sorted(merged, key=lambda x: x['score'], reverse=True)
        best_hyps.append(merged[0]['hyp'])

    return best_hyps
