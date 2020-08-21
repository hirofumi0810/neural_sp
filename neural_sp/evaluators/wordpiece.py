#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate the wordpiece-level model by WER."""

import logging
from tqdm import tqdm

from neural_sp.evaluators.edit_distance import compute_wer
from neural_sp.utils import mkdir_join

logger = logging.getLogger(__name__)


def eval_wordpiece(models, dataset, recog_params, epoch,
                   recog_dir=None, streaming=False, progressbar=False,
                   fine_grained=False):
    """Evaluate the wordpiece-level model by WER.

    Args:
        models (list): models to evaluate
        dataset (Dataset): evaluation dataset
        recog_params (dict):
        epoch (int):
        recog_dir (str):
        streaming (bool): streaming decoding for the session-level evaluation
        progressbar (bool): visualize the progressbar
        fine_grained (bool): calculate fine-grained WER distributions based on input lengths
    Returns:
        wer (float): Word error rate
        cer (float): Character error rate

    """
    if recog_dir is None:
        recog_dir = 'decode_' + dataset.set + '_ep' + str(epoch) + '_beam' + str(recog_params['recog_beam_width'])
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
    wer_dist = {}  # calculate WER distribution based on input lengths
    n_streamable, quantity_rate, n_utt = 0, 0, 0
    last_success_frame_ratio = 0

    # Reset data counter
    dataset.reset(recog_params['recog_batch_size'])

    if progressbar:
        pbar = tqdm(total=len(dataset))

    with open(hyp_trn_path, 'w') as f_hyp, open(ref_trn_path, 'w') as f_ref:
        while True:
            batch, is_new_epoch = dataset.next(recog_params['recog_batch_size'])
            if streaming or recog_params['recog_chunk_sync']:
                best_hyps_id, _ = models[0].decode_streaming(
                    batch['xs'], recog_params, dataset.idx2token[0],
                    exclude_eos=True)
            else:
                best_hyps_id, _ = models[0].decode(
                    batch['xs'], recog_params,
                    idx2token=dataset.idx2token[0] if progressbar else None,
                    exclude_eos=True,
                    refs_id=batch['ys'],
                    utt_ids=batch['utt_ids'],
                    speakers=batch['sessions' if dataset.corpus == 'swbd' else 'speakers'],
                    ensemble_models=models[1:] if len(models) > 1 else [])

            for b in range(len(batch['xs'])):
                ref = batch['text'][b]
                if ref[0] == '<':
                    ref = ref.split('>')[1]
                hyp = dataset.idx2token[0](best_hyps_id[b])

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

                    if fine_grained:
                        xlen_bin = (batch['xlens'][b] // 200 + 1) * 200
                        if xlen_bin in wer_dist.keys():
                            wer_dist[xlen_bin] += [wer_b / 100]
                        else:
                            wer_dist[xlen_bin] = [wer_b / 100]

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
                    if models[0].streamable():
                        n_streamable += 1
                    else:
                        last_success_frame_ratio += models[0].last_success_frame_ratio()
                    quantity_rate += models[0].quantity_rate()
                    n_utt += 1

                if progressbar:
                    pbar.update(1)

            if is_new_epoch:
                break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    if not streaming:
        wer /= n_word
        n_sub_w /= n_word
        n_ins_w /= n_word
        n_del_w /= n_word

        cer /= n_char
        n_sub_c /= n_char
        n_ins_c /= n_char
        n_del_c /= n_char

        if n_utt - n_streamable > 0:
            last_success_frame_ratio /= (n_utt - n_streamable)
        n_streamable /= n_utt
        quantity_rate /= n_utt

        if fine_grained:
            for len_bin, wers in sorted(wer_dist.items(), key=lambda x: x[0]):
                logger.info('  WER (%s): %.2f %% (%d)' % (dataset.set, sum(wers) / len(wers), len_bin))

    logger.debug('WER (%s): %.2f %%' % (dataset.set, wer))
    logger.debug('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub_w, n_ins_w, n_del_w))
    logger.debug('CER (%s): %.2f %%' % (dataset.set, cer))
    logger.debug('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub_c, n_ins_c, n_del_c))

    logger.info('Streamablility (%s): %.2f %%' % (dataset.set, n_streamable * 100))
    logger.info('Quantity rate (%s): %.2f %%' % (dataset.set, quantity_rate * 100))
    logger.info('Last success frame ratio (%s): %.2f %%' % (dataset.set, last_success_frame_ratio))

    return wer, cer
