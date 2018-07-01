#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method of A2P + P2W models on the modular training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import pandas as pd

from src.utils.io.labels.word import Word2char
from src.utils.evaluation.edit_distance import compute_wer
from src.utils.evaluation.resolving_unk import resolve_unk
from src.utils.evaluation.normalization import normalize, normalize_swbd, GLM


def eval_word(models_a2p, models_p2w, dataset_a2p, dataset_p2w, eval_batch_size,
              beam_width_a2p, max_decode_len_a2p, min_decode_len_a2p, min_decode_len_ratio_a2p,
              beam_width_p2w, max_decode_len_p2w, min_decode_len_p2w, min_decode_len_ratio_p2w,
              beam_width_p2w_sub=1, max_decode_len_p2w_sub=300, min_decode_len_p2w_sub=1, min_decode_len_ratio_p2w_sub=0,
              length_penalty_a2p=0, coverage_penalty_a2p=0,
              length_penalty_p2w=0, coverage_penalty_p2w=0,
              length_penalty_p2w_sub=0, coverage_penalty_p2w_sub=0,
              rnnlm_weight=0, rnnlm_weight_sub=0,
              progressbar=False, resolving_unk=False, a2c_oracle=False,
              joint_decoding=None, score_sub_weight=0):
    """Evaluate a model by WER.
    Args:
        models_a2p (list): the A2P models to evaluate
        models_p2w (list): the P2W models to evaluate
        dataset_a2p: An instance of a `Dataset' class
        dataset_p2w: An instance of a `Dataset' class
        eval_batch_size (int): the batch size when evaluating the model
        beam_width_a2p (int): the size of beam of A2P models
        max_decode_len_a2p (int): the maximum sequence length of tokens of A2P models
        min_decode_len_a2p (int): the minimum sequence length of tokens of A2P models
        min_decode_len_ratio_a2p (float):
        beam_width_p2w (int): the size of beam in the main task of P2W models
        max_decode_len_p2w (int): the maximum sequence length of tokens in the main task of P2W models
        min_decode_len_p2w (int): the minimum sequence length of tokens in the main task of P2W models
        min_decode_len_ratio_p2w (float):
        beam_width_p2w_sub (int): the size of beam in the sub task of P2W models
            This is used for the nested attention
        max_decode_len_p2w_sub (int): the maximum sequence length of tokens in the sub task of P2W models
        min_decode_len_p2w_sub (int): the minimum sequence length of tokens in the sub task of P2W models
        min_decode_len_ratio_p2w_sub (float):
        length_penalty_a2p (float): length penalty
        coverage_penalty_a2p (float): coverage penalty
        length_penalty_p2w (float): length penalty
        coverage_penalty_p2w (float): coverage penalty
        length_penalty_p2w_sub (float): length penalty
        coverage_penalty_p2w_sub (float): coverage penalty
        rnnlm_weight (float): the weight of RNNLM score of the main task
        rnnlm_weight_sub (float): the weight of RNNLM score of the sub task
        progressbar (bool): if True, visualize the progressbar
        resolving_unk (bool):
        a2c_oracle (bool):
        joint_decoding (bool):
        score_sub_weight (float):
    Returns:
        wer (float): Word error rate
        df (pd.DataFrame): dataframe of substitution, insertion, and deletion
    """
    # Reset data counter
    dataset_a2p.reset()
    dataset_p2w.reset()

    model_a2p = models_a2p[0]
    model_p2w = models_p2w[0]
    # TODO: fix this

    if model_p2w.model_type == 'hierarchical_attention' and joint_decoding:
        word2char = Word2char(dataset_p2w.vocab_file_path,
                              dataset_p2w.vocab_file_path_sub)

    wer = 0
    sub, ins, dele, = 0, 0, 0
    num_words = 0
    num_oov = 0
    if progressbar:
        pbar = tqdm(total=len(dataset_p2w))  # TODO: fix this
    while True:
        batch_a2p, is_new_epoch = dataset_a2p.next(batch_size=eval_batch_size)
        batch_p2w, is_new_epoch = dataset_p2w.next(batch_size=eval_batch_size)

        # Decode (A2P)
        best_hyps_a2p, _, perm_idx = model_a2p.decode(
            batch_a2p['xs'],
            beam_width=beam_width_a2p,
            max_decode_len=max_decode_len_a2p,
            min_decode_len=min_decode_len_a2p,
            min_decode_len_ratio=min_decode_len_a2p,
            length_penalty=length_penalty_a2p,
            coverage_penalty=coverage_penalty_a2p)

        ys = [batch_p2w['ys'][i] for i in perm_idx]

        # Decode (P2W)~
        if model_p2w.model_type == 'nested_attention':
            raise NotImplementedError
        elif model_p2w.model_type == 'hierarchical_attention' and joint_decoding:
            best_hyps, aw, best_hyps_sub, aw_sub, perm_idx = model_p2w.decode(
                best_hyps_a2p,
                beam_width=beam_width_p2w,
                max_decode_len=max_decode_len_p2w,
                min_decode_len=min_decode_len_p2w,
                min_decode_len_ratio=min_decode_len_ratio_p2w,
                length_penalty=length_penalty_p2w,
                coverage_penalty=coverage_penalty_p2w,
                rnnlm_weight=rnnlm_weight,
                joint_decoding=joint_decoding,
                space_index=dataset_p2w.char2idx('_')[0],
                oov_index=dataset_p2w.word2idx('OOV')[0],
                word2char=word2char,
                idx2word=dataset_p2w.idx2word,
                idx2char=dataset_p2w.idx2char,
                score_sub_weight=score_sub_weight)
        else:
            best_hyps, aw, perm_idx = model_p2w.decode(
                best_hyps_a2p,
                beam_width=beam_width_p2w,
                max_decode_len=max_decode_len_p2w,
                min_decode_len=min_decode_len_p2w,
                min_decode_len_ratio=min_decode_len_ratio_p2w,
                length_penalty=length_penalty_p2w,
                coverage_penalty=coverage_penalty_p2w,
                rnnlm_weight=rnnlm_weight)

        ys = [batch_p2w['ys'][i] for i in perm_idx]

        for b in range(len(batch_p2w['xs'])):
            # Reference
            if dataset_p2w.is_test:
                str_ref = ys[b]
                # NOTE: transcript is seperated by space('_')
            else:
                str_ref = dataset_p2w.idx2word(ys[b])

            # Hypothesis
            str_hyp = dataset_p2w.idx2word(best_hyps[b])
            num_oov += str_hyp.count('OOV')

            # Resolving UNK
            if resolving_unk and 'OOV' in str_hyp:
                if not (model_p2w.model_type == 'hierarchical_attention' and joint_decoding) and model_p2w.model_type != 'nested_attention':
                    best_hyps_sub, aw_sub, _ = model_p2w.decode(
                        batch_p2w['xs'][b:b + 1],
                        beam_width=beam_width_p2w_sub,
                        max_decode_len=max_decode_len_p2w_sub,
                        min_decode_len=min_decode_len_p2w_sub,
                        min_decode_len_ratio=min_decode_len_ratio_p2w_sub,
                        length_penalty=length_penalty_p2w_sub,
                        coverage_penalty=coverage_penalty_p2w_sub,
                        rnnlm_weight_sub=rnnlm_weight_sub,
                        task_index=1)

                str_hyp = resolve_unk(
                    str_hyp, best_hyps_sub[0], aw[b], aw_sub[0], dataset_p2w.idx2char,
                    diff_time_resolution=2 ** sum(model_p2w.subsample_list) // 2 ** sum(model_p2w.subsample_list[:model_p2w.encoder_num_layers_sub - 1]))
                str_hyp = str_hyp.replace('*', '')

            # Post-proccessing
            if dataset_p2w.corpus in ['csj', 'wsj']:
                str_ref = normalize(str_ref, remove_tokens=['@'])
                str_hyp = normalize(str_hyp, remove_tokens=['@', '>'])
                # NOTE: @ means <sp> (CSJ), noise (WSJ)
            elif dataset_p2w.corpus == 'swbd':
                if 'eval2000' in dataset_p2w.data_type:
                    glm = GLM(dataset_p2w.glm_path)
                    str_ref = normalize_swbd(str_ref, glm)
                    str_hyp = normalize_swbd(str_hyp, glm)
                else:
                    str_hyp = normalize(str_hyp, remove_tokens=['>'])
            elif dataset_p2w.corpus == 'librispeech':
                str_hyp = normalize(str_hyp, remove_tokens=['>'])
            else:
                raise ValueError(dataset_p2w.corpus)

            if len(str_ref) == 0:
                continue

            # print(str_ref)
            # print(str_hyp)

            # Compute WER
            wer_b, sub_b, ins_b, del_b = compute_wer(ref=str_ref.split('_'),
                                                     hyp=str_hyp.split('_'),
                                                     normalize=False)
            wer += wer_b
            sub += sub_b
            ins += ins_b
            dele += del_b
            num_words += len(str_ref.split('_'))

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset_a2p.reset()
    dataset_p2w.reset()

    wer /= num_words
    sub /= num_words
    ins /= num_words
    dele /= num_words

    df = pd.DataFrame({'SUB': [sub], 'INS': [ins], 'DEL': [dele], 'OOV': [num_oov]},
                      columns=['SUB', 'INS', 'DEL', 'OOV'],
                      index=['WER'])

    return wer, df
