#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate the wordpiece-level model by WER."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from tqdm import tqdm

from neural_sp.evaluators.edit_distance import compute_wer
from neural_sp.utils.general import mkdir_join

logger = logging.getLogger("decoding").getChild('wordpiece')


def eval_wordpiece(models, dataset, recog_params, epoch,
                   recog_dir=None, progressbar=False):
    """Evaluate the wordpiece-level model by WER.

    Args:
        models (list): models to evaluate
        dataset: An instance of a `Dataset' class
        recog_params (recog_dict):
        epoch (int):
        recog_dir (str):
        progressbar (bool): visualize the progressbar
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

    with open(hyp_trn_save_path, 'w') as f_hyp, open(ref_trn_save_path, 'w') as f_ref:
        while True:
            batch, is_new_epoch = dataset.next(recog_params['recog_batch_size'])
            best_hyps_id, _, perm_id, _ = models[0].decode(
                batch['xs'], recog_params, dataset.idx2token[0],
                exclude_eos=True,
                refs_id=batch['ys'],
                utt_ids=batch['utt_ids'],
                speakers=batch['sessions'] if dataset.corpus == 'swbd' else batch['speakers'],
                ensemble_models=models[1:] if len(models) > 1 else [])
            ys = [batch['text'][i] for i in perm_id]

            for b in range(len(batch['xs'])):
                ref = ys[b]
                hyp = dataset.idx2token[0](best_hyps_id[b])

                # Write to trn
                utt_id = str(batch['utt_ids'][b])
                speaker = str(batch['speakers'][b]).replace('-', '_')
                f_ref.write(ref + ' (' + speaker + '-' + utt_id + ')\n')
                f_hyp.write(hyp + ' (' + speaker + '-' + utt_id + ')\n')
                logger.info('utt-id: %s' % batch['utt_ids'][b])
                logger.info('Ref: %s' % ref)
                logger.info('Hyp: %s' % hyp)
                logger.info('-' * 150)

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

    wer /= n_word
    n_sub_w /= n_word
    n_ins_w /= n_word
    n_del_w /= n_word

    cer /= n_char
    n_sub_c /= n_char
    n_ins_c /= n_char
    n_del_c /= n_char

    logger.info('WER (%s): %.2f %%' % (dataset.set, wer))
    logger.info('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub_w, n_ins_w, n_del_w))
    logger.info('CER (%s): %.2f %%' % (dataset.set, cer))
    logger.info('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub_c, n_ins_c, n_del_c))

    return wer, cer
