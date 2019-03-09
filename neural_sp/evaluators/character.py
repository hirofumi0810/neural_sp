#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate the character-level model by WER & CER."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from tqdm import tqdm

from neural_sp.evaluators.edit_distance import compute_wer
from neural_sp.utils.general import mkdir_join

logger = logging.getLogger("decoding").getChild('character')


def eval_char(models, dataset, decode_params, epoch,
              decode_dir=None, progressbar=False, task_id=0):
    """Evaluate the character-level model by WER & CER.

    Args:
        models (list): the models to evaluate
        dataset: An instance of a `Dataset' class
        decode_params (dict):
        epoch (int):
        decode_dir (str):
        progressbar (bool): if True, visualize the progressbar
        task_id (int): the index of the target task in interest
            0: main task
            1: sub task
            2: sub sub task
    Returns:
        wer (float): Word error rate
        n_sub_w (int): the number of substitution errors for WER
        n_ins_w (int): the number of insertion errors for WER
        n_del_w (int): the number of deletion errors for WER
        cer (float): Character error rate
        n_sub_w (int): the number of substitution errors for CER
        n_ins_c (int): the number of insertion errors for CER
        n_del_c (int): the number of deletion errors for CER

    """
    # Reset data counter
    dataset.reset()

    if decode_dir is None:
        decode_dir = 'decode_' + dataset.set + '_ep' + str(epoch) + '_beam' + str(decode_params['recog_beam_width'])
        decode_dir += '_lp' + str(decode_params['recog_length_penalty'])
        decode_dir += '_cp' + str(decode_params['recog_coverage_penalty'])
        decode_dir += '_' + str(decode_params['recog_min_len_ratio']) + '_' + str(decode_params['recog_max_len_ratio'])
        decode_dir += '_rnnlm' + str(decode_params['recog_rnnlm_weight'])

        ref_trn_save_path = mkdir_join(models[0].save_path, decode_dir, 'ref.trn')
        hyp_trn_save_path = mkdir_join(models[0].save_path, decode_dir, 'hyp.trn')
    else:
        ref_trn_save_path = mkdir_join(decode_dir, 'ref.trn')
        hyp_trn_save_path = mkdir_join(decode_dir, 'hyp.trn')

    wer, cer = 0, 0
    n_sub_w, n_ins_w, n_del_w = 0, 0, 0
    n_sub_c, n_ins_c, n_del_c = 0, 0, 0
    n_word, n_char = 0, 0
    if progressbar:
        pbar = tqdm(total=len(dataset))

    if task_id == 0:
        task = 'ys'
    elif task_id == 1:
        task = 'ys_sub1'
    elif task_id == 2:
        task = 'ys_sub2'
    elif task_id == 3:
        task = 'ys_sub3'

    with open(hyp_trn_save_path, 'w') as f_hyp, open(ref_trn_save_path, 'w') as f_ref:
        while True:
            batch, is_new_ep = dataset.next(decode_params['recog_batch_size'])
            best_hyps, _, perm_ids, _ = models[0].decode(
                batch['xs'], decode_params,
                exclude_eos=True,
                task=task,
                ensemble_models=models[1:] if len(models) > 1 else [],
                speakers=batch['speakers'])
            ys = [batch['text'][i] for i in perm_ids]

            for b in range(len(batch['xs'])):
                ref = ys[b]
                hyp = dataset.id2char(best_hyps[b])

                # Write to trn
                speaker = '_'.join(batch['utt_ids'][b].replace('-', '_').split('_')[:-2])
                start = batch['utt_ids'][b].replace('-', '_').split('_')[-2]
                end = batch['utt_ids'][b].replace('-', '_').split('_')[-1]
                f_ref.write(ref + ' (' + speaker + '-' + start + '-' + end + ')\n')
                f_hyp.write(hyp + ' (' + speaker + '-' + start + '-' + end + ')\n')
                logger.info('utt-id: %s' % batch['utt_ids'][b])
                # logger.info('Ref: %s' % ref.lower())
                logger.info('Ref: %s' % ref)
                logger.info('Hyp: %s' % hyp)
                logger.info('-' * 150)

                if ('char' in dataset.unit and 'nowb' not in dataset.unit) or (task_id > 0 and dataset.unit_sub1 == 'char'):
                    # Compute WER
                    wer_b, sub_b, ins_b, del_b = compute_wer(ref=ref.split(' '),
                                                             hyp=hyp.split(' '),
                                                             normalize=False)
                    wer += wer_b
                    n_sub_w += sub_b
                    n_ins_w += ins_b
                    n_del_w += del_b
                    n_word += len(ref.split(' '))
                    # logger.info('WER: %d%%' % (wer_b / len(ref.split(' '))))

                # Compute CER
                cer_b, sub_b, ins_b, del_b = compute_wer(ref=list(ref),
                                                         hyp=list(hyp),
                                                         normalize=False)
                cer += cer_b
                n_sub_c += sub_b
                n_ins_c += ins_b
                n_del_c += del_b
                n_char += len(ref)
                # logger.info('CER: %d%%' % (cer_b / len(ref)))

                if progressbar:
                    pbar.update(1)

            if is_new_ep:
                break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    if ('char' in dataset.unit and 'nowb' not in dataset.unit) or (task_id > 0 and dataset.unit_sub1 == 'char'):
        wer /= n_word
        n_sub_w /= n_word
        n_ins_w /= n_word
        n_del_w /= n_word
    else:
        wer = n_sub_w = n_ins_w = n_del_w = 0

    cer /= n_char
    n_sub_c /= n_char
    n_ins_c /= n_char
    n_del_c /= n_char

    return (wer, n_sub_w, n_ins_w, n_del_w), (cer, n_sub_c, n_ins_c, n_del_c)
