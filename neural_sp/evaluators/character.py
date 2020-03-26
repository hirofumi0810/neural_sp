#! /usr/bin/env python3
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
from neural_sp.utils import mkdir_join

logger = logging.getLogger(__name__)


def eval_char(models, dataset, recog_params, epoch,
              recog_dir=None, streaming=False, progressbar=False, task_idx=0):
    """Evaluate the character-level model by WER & CER.

    Args:
        models (list): models to evaluate
        dataset (Dataset): evaluation dataset
        recog_params (dict):
        epoch (int):
        recog_dir (str):
        streaming (bool): streaming decoding for the session-level evaluation
        progressbar (bool): visualize the progressbar
        task_idx (int): the index of the target task in interest
            0: main task
            1: sub task
            2: sub sub task
    Returns:
        wer (float): Word error rate
        cer (float): Character error rate

    """
    # Reset data counter
    dataset.reset()

    if recog_dir is None:
        recog_dir = 'decode_' + dataset.set + '_ep' + str(epoch) + '_beam' + str(recog_params['recog_beam_width'])
        recog_dir += '_lp' + str(recog_params['recog_length_penalty'])
        recog_dir += '_cp' + str(recog_params['recog_coverage_penalty'])
        recog_dir += '_' + str(recog_params['recog_min_len_ratio']) + '_' + str(recog_params['recog_max_len_ratio'])
        recog_dir += '_lm' + str(recog_params['recog_lm_weight'])

        ref_trn_save_path = mkdir_join(models[0].save_path, recog_dir, 'ref.trn')
        hyp_trn_save_path = mkdir_join(models[0].save_path, recog_dir, 'hyp.trn')
    else:
        ref_trn_save_path = mkdir_join(recog_dir, 'ref.trn')
        hyp_trn_save_path = mkdir_join(recog_dir, 'hyp.trn')

    wer, cer = 0, 0
    n_sub_w, n_ins_w, n_del_w = 0, 0, 0
    n_sub_c, n_ins_c, n_del_c = 0, 0, 0
    n_word, n_char = 0, 0
    if progressbar:
        pbar = tqdm(total=len(dataset))

    if task_idx == 0:
        task = 'ys'
    elif task_idx == 1:
        task = 'ys_sub1'
    elif task_idx == 2:
        task = 'ys_sub2'
    elif task_idx == 3:
        task = 'ys_sub3'

    with open(hyp_trn_save_path, 'w') as f_hyp, open(ref_trn_save_path, 'w') as f_ref:
        while True:
            batch, is_new_epoch = dataset.next(recog_params['recog_batch_size'])
            if streaming or recog_params['recog_chunk_sync']:
                best_hyps_id, _ = models[0].decode_streaming(
                    batch['xs'], recog_params, dataset.idx2token[0],
                    exclude_eos=True)
            else:
                best_hyps_id, _ = models[0].decode(
                    batch['xs'], recog_params, dataset.idx2token[task_idx],
                    exclude_eos=True,
                    refs_id=batch['ys'] if task_idx == 0 else batch['ys_sub' + str(task_idx)],
                    utt_ids=batch['utt_ids'],
                    speakers=batch['sessions' if dataset.corpus == 'swbd' else 'speakers'],
                    task=task,
                    ensemble_models=models[1:] if len(models) > 1 else [])

            for b in range(len(batch['xs'])):
                ref = batch['text'][b]
                hyp = dataset.idx2token[task_idx](best_hyps_id[b])

                # Write to trn
                speaker = str(batch['speakers'][b]).replace('-', '_')
                if streaming:
                    utt_id = str(batch['utt_ids'][b]) + '_0000000_0000001'
                else:
                    utt_id = str(batch['utt_ids'][b])
                f_ref.write(ref + ' (' + speaker + '-' + utt_id + ')\n')
                f_hyp.write(hyp + ' (' + speaker + '-' + utt_id + ')\n')
                logger.debug('utt-id: %s' % utt_id)
                logger.debug('Ref: %s' % ref)
                logger.debug('Hyp: %s' % hyp)
                logger.debug('-' * 150)

                if not streaming:
                    if ('char' in dataset.unit and 'nowb' not in dataset.unit) or (task_idx > 0 and dataset.unit_sub1 == 'char'):
                        # Compute WER
                        wer_b, sub_b, ins_b, del_b = compute_wer(ref=ref.split(' '),
                                                                 hyp=hyp.split(' '),
                                                                 normalize=False)
                        wer += wer_b
                        n_sub_w += sub_b
                        n_ins_w += ins_b
                        n_del_w += del_b
                        n_word += len(ref.split(' '))

                    # Compute CER
                    if dataset.corpus == 'csj':
                        ref = ref.replace(' ', '')
                        hyp = hyp.replace(' ', '')
                    cer_b, sub_b, ins_b, del_b = compute_wer(ref=list(ref),
                                                             hyp=list(hyp),
                                                             normalize=False)
                    cer += cer_b
                    n_sub_c += sub_b
                    n_ins_c += ins_b
                    n_del_c += del_b
                    n_char += len(ref)

                if progressbar:
                    pbar.update(1)

            if is_new_epoch:
                break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    if not streaming:
        if ('char' in dataset.unit and 'nowb' not in dataset.unit) or (task_idx > 0 and dataset.unit_sub1 == 'char'):
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

    logger.debug('WER (%s): %.2f %%' % (dataset.set, wer))
    logger.debug('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub_w, n_ins_w, n_del_w))
    logger.debug('CER (%s): %.2f %%' % (dataset.set, cer))
    logger.debug('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub_c, n_ins_c, n_del_c))

    return wer, cer
