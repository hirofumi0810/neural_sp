#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate the word-level model by WER."""

import codecs
import copy
import logging
import numpy as np
from tqdm import tqdm

from neural_sp.evaluators.edit_distance import compute_wer
from neural_sp.evaluators.resolving_unk import resolve_unk
from neural_sp.utils import mkdir_join

logger = logging.getLogger(__name__)


def eval_word(models, dataloader, recog_params, epoch,
              recog_dir=None, streaming=False, progressbar=False):
    """Evaluate the word-level model by WER.

    Args:
        models (list): models to evaluate
        dataloader (torch.utils.data.DataLoader): evaluation dataloader
        recog_params (dict):
        epoch (int):
        recog_dir (str):
        streaming (bool): streaming decoding for the session-level evaluation
        progressbar (bool): visualize the progressbar
    Returns:
        wer (float): Word error rate
        cer (float): Character error rate
        n_oov_total (int): totol number of OOV

    """
    if recog_dir is None:
        recog_dir = 'decode_' + dataloader.set + '_ep' + str(epoch) + '_beam' + str(recog_params['recog_beam_width'])
        recog_dir += '_lp' + str(recog_params['recog_length_penalty'])
        recog_dir += '_cp' + str(recog_params['recog_coverage_penalty'])
        recog_dir += '_' + str(recog_params['recog_min_len_ratio']) + '_' + str(recog_params['recog_max_len_ratio'])
        recog_dir += '_lm' + str(recog_params['recog_lm_weight'])

        ref_trn_path = mkdir_join(models[0].save_path, recog_dir, 'ref.trn')
        hyp_trn_path = mkdir_join(models[0].save_path, recog_dir, 'hyp.trn')
    else:
        ref_trn_path = mkdir_join(recog_dir, 'ref.trn')
        hyp_trn_path = mkdir_join(recog_dir, 'hyp.trn')

    wer, cer = 0, 0
    n_sub_w, n_ins_w, n_del_w = 0, 0, 0
    n_sub_c, n_ins_c, n_del_c = 0, 0, 0
    n_word, n_char = 0, 0
    n_oov_total = 0

    # Reset data counter
    dataloader.reset(recog_params['recog_batch_size'])

    if progressbar:
        pbar = tqdm(total=len(dataloader))

    with codecs.open(hyp_trn_path, 'w', encoding='utf-8') as f_hyp, \
            codecs.open(ref_trn_path, 'w', encoding='utf-8') as f_ref:
        while True:
            batch, is_new_epoch = dataloader.next(recog_params['recog_batch_size'])
            if streaming or recog_params['recog_chunk_sync']:
                best_hyps_id, _ = models[0].decode_streaming(
                    batch['xs'], recog_params, dataloader.idx2token[0],
                    exclude_eos=True)
            else:
                best_hyps_id, aws = models[0].decode(
                    batch['xs'], recog_params,
                    idx2token=dataloader.idx2token[0] if progressbar else None,
                    exclude_eos=True,
                    refs_id=batch['ys'],
                    utt_ids=batch['utt_ids'],
                    speakers=batch['sessions' if dataloader.corpus == 'swbd' else 'speakers'],
                    ensemble_models=models[1:] if len(models) > 1 else [])

            for b in range(len(batch['xs'])):
                ref = batch['text'][b]
                hyp = dataloader.idx2token[0](best_hyps_id[b])

                n_oov_total += hyp.count('<unk>')

                # Resolving UNK
                if recog_params['recog_resolving_unk'] and '<unk>' in hyp:
                    recog_params_char = copy.deepcopy(recog_params)
                    recog_params_char['recog_lm_weight'] = 0
                    recog_params_char['recog_beam_width'] = 1
                    best_hyps_id_char, aw_char = models[0].decode(
                        batch['xs'][b:b + 1], recog_params_char,
                        idx2token=dataloader.idx2token[1] if progressbar else None,
                        exclude_eos=True,
                        refs_id=batch['ys_sub1'],
                        utt_ids=batch['utt_ids'],
                        speakers=batch['sessions'] if dataloader.corpus == 'swbd' else batch['speakers'],
                        task='ys_sub1')
                    # TODO(hirofumi): support ys_sub2 and ys_sub3

                    assert not streaming

                    hyp = resolve_unk(
                        hyp, best_hyps_id_char[0], aws[b], aw_char[0], dataloader.idx2token[1],
                        subsample_factor_word=np.prod(models[0].subsample),
                        subsample_factor_char=np.prod(models[0].subsample[:models[0].enc_n_layers_sub1 - 1]))
                    logger.debug('Hyp (after OOV resolution): %s' % hyp)
                    hyp = hyp.replace('*', '')

                    # Compute CER
                    ref_char = ref
                    hyp_char = hyp
                    if dataloader.corpus == 'csj':
                        ref_char = ref.replace(' ', '')
                        hyp_char = hyp.replace(' ', '')
                    cer_b, sub_b, ins_b, del_b = compute_wer(ref=list(ref_char),
                                                             hyp=list(hyp_char),
                                                             normalize=False)
                    cer += cer_b
                    n_sub_c += sub_b
                    n_ins_c += ins_b
                    n_del_c += del_b
                    n_char += len(ref_char)

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
                    # Compute WER
                    wer_b, sub_b, ins_b, del_b = compute_wer(ref=ref.split(' '),
                                                             hyp=hyp.split(' '),
                                                             normalize=False)
                    wer += wer_b
                    n_sub_w += sub_b
                    n_ins_w += ins_b
                    n_del_w += del_b
                    n_word += len(ref.split(' '))

                if progressbar:
                    pbar.update(1)

            if is_new_epoch:
                break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataloader.reset()

    if not streaming:
        wer /= n_word
        n_sub_w /= n_word
        n_ins_w /= n_word
        n_del_w /= n_word

        if n_char > 0:
            cer /= n_char
            n_sub_c /= n_char
            n_ins_c /= n_char
            n_del_c /= n_char

    logger.debug('WER (%s): %.2f %%' % (dataloader.set, wer))
    logger.debug('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub_w, n_ins_w, n_del_w))
    logger.debug('CER (%s): %.2f %%' % (dataloader.set, cer))
    logger.debug('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub_c, n_ins_c, n_del_c))
    logger.debug('OOV (total): %d' % (n_oov_total))

    return wer, cer, n_oov_total
