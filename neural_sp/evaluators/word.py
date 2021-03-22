# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate a word-level model by WER."""

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
              recog_dir=None, streaming=False, progressbar=False,
              edit_distance=True, fine_grained=False, oracle=False, teacher_force=False):
    """Evaluate a word-level model by WER.

    Args:
        models (List): models to evaluate
        dataloader (torch.utils.data.DataLoader): evaluation dataloader
        recog_params (omegaconf.dictconfig.DictConfig): decoding hyperparameters
        epoch (int): current epoch
        recog_dir (str): directory path to save hypotheses
        streaming (bool): streaming decoding for session-level evaluation
        progressbar (bool): visualize progressbar
        edit_distance (bool): calculate edit-distance (can be skipped for RTF calculation)
        fine_grained (bool): calculate fine-grained WER distributions based on input lengths
        oracle (bool): calculate oracle WER
        teacher_force (bool): conduct decoding in teacher-forcing mode
    Returns:
        wer (float): Word error rate
        cer (float): Character error rate
        n_oov_total (int): total number of OOV

    """
    if recog_dir is None:
        recog_dir = 'decode_' + dataloader.set + '_ep' + \
            str(epoch) + '_beam' + str(recog_params.get('recog_beam_width'))
        recog_dir += '_lp' + str(recog_params.get('recog_length_penalty'))
        recog_dir += '_cp' + str(recog_params.get('recog_coverage_penalty'))
        recog_dir += '_' + str(recog_params.get('recog_min_len_ratio')) + '_' + \
            str(recog_params.get('recog_max_len_ratio'))
        recog_dir += '_lm' + str(recog_params.get('recog_lm_weight'))

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
    n_oov_total = 0

    wer_oracle = 0
    n_oracle_hit = 0
    n_utt = 0

    # Reset data counter
    dataloader.reset(recog_params.get('recog_batch_size'))

    if progressbar:
        pbar = tqdm(total=len(dataloader))

    with codecs.open(hyp_trn_path, 'w', encoding='utf-8') as f_hyp, \
            codecs.open(ref_trn_path, 'w', encoding='utf-8') as f_ref:
        for batch in dataloader:
            speakers = batch['sessions' if dataloader.corpus == 'swbd' else 'speakers']
            if streaming or recog_params.get('recog_block_sync'):
                nbest_hyps_id = models[0].decode_streaming(
                    batch['xs'], recog_params, dataloader.idx2token[0],
                    exclude_eos=True,
                    speaker=speakers[0])[0]
            else:
                nbest_hyps_id, aws = models[0].decode(
                    batch['xs'], recog_params,
                    idx2token=dataloader.idx2token[0],
                    exclude_eos=True,
                    refs_id=batch['ys'],
                    utt_ids=batch['utt_ids'],
                    speakers=speakers,
                    ensemble_models=models[1:] if len(models) > 1 else [])

            for b in range(len(batch['xs'])):
                ref = batch['text'][b]
                nbest_hyps = [dataloader.idx2token[0](hyp_id) for hyp_id in nbest_hyps_id[b]]
                n_oov_total += nbest_hyps[0].count('<unk>')

                # Resolving UNK
                if recog_params.get('recog_resolving_unk') and '<unk>' in nbest_hyps[0]:
                    recog_params_char = copy.deepcopy(recog_params)
                    recog_params_char['recog_lm_weight'] = 0
                    recog_params_char['recog_beam_width'] = 1
                    best_hyps_id_char, aw_char = models[0].decode(
                        batch['xs'][b:b + 1], recog_params_char,
                        idx2token=dataloader.idx2token[1],
                        exclude_eos=True,
                        refs_id=batch['ys_sub1'],
                        utt_ids=batch['utt_ids'],
                        speakers=speakers,
                        task='ys_sub1')
                    # TODO(hirofumi): support ys_sub2

                    assert not streaming

                    nbest_hyps[0] = resolve_unk(
                        nbest_hyps[0], best_hyps_id_char[0], aws[b], aw_char[0], dataloader.idx2token[1],
                        subsample_factor_word=np.prod(models[0].subsample),
                        subsample_factor_char=np.prod(models[0].subsample[:models[0].enc_n_layers_sub1 - 1]))
                    logger.debug('Hyp (after OOV resolution): %s' % nbest_hyps[0])
                    nbest_hyps[0] = nbest_hyps[0].replace('*', '')

                    # Compute CER
                    ref_char = ref
                    hyp_char = nbest_hyps[0]
                    if dataloader.corpus == 'csj':
                        ref_char = ref_char.replace(' ', '')
                        hyp_char = hyp_char.replace(' ', '')
                    err_b, sub_b, ins_b, del_b = compute_wer(ref=list(ref_char),
                                                             hyp=list(hyp_char))
                    cer += err_b
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
                f_hyp.write(nbest_hyps[0] + ' (' + speaker + '-' + utt_id + ')\n')
                logger.debug('utt-id (%d/%d): %s' % (n_utt + 1, len(dataloader), utt_id))
                logger.debug('Ref: %s' % ref)
                logger.debug('Hyp: %s' % nbest_hyps[0])
                logger.debug('-' * 150)

                if edit_distance and not streaming:
                    # Compute WER
                    err_b, sub_b, ins_b, del_b = compute_wer(ref=ref.split(' '),
                                                             hyp=nbest_hyps[0].split(' '))
                    wer += err_b
                    n_sub_w += sub_b
                    n_ins_w += ins_b
                    n_del_w += del_b
                    n_word += len(ref.split(' '))

                    # Compute oracle WER
                    if oracle and len(nbest_hyps) > 1:
                        wers_b = [err_b] + [compute_wer(ref=ref.split(' '),
                                                        hyp=hyp_n.split(' '))[0]
                                            for hyp_n in nbest_hyps[1:]]
                        oracle_idx = np.argmin(np.array(wers_b))
                        if oracle_idx == 0:
                            n_oracle_hit += len(batch['utt_ids'])
                        wer_oracle += wers_b[oracle_idx]
                        # NOTE: OOV resolution is not considered

                    if fine_grained:
                        xlen_bin = (batch['xlens'][b] // 200 + 1) * 200
                        if xlen_bin in wer_dist.keys():
                            wer_dist[xlen_bin] += [err_b / 100]
                        else:
                            wer_dist[xlen_bin] = [err_b / 100]

                n_utt += len(batch['utt_ids'])
                if progressbar:
                    pbar.update(len(batch['utt_ids']))

    if progressbar:
        pbar.close()

    # Reset data counters
    dataloader.reset(is_new_epoch=True)

    if edit_distance and not streaming:
        wer /= n_word
        n_sub_w /= n_word
        n_ins_w /= n_word
        n_del_w /= n_word

        if n_char > 0:
            cer /= n_char
            n_sub_c /= n_char
            n_ins_c /= n_char
            n_del_c /= n_char

        if recog_params.get('recog_beam_width') > 1:
            logger.info('WER (%s): %.2f %%' % (dataloader.set, wer))
            logger.info('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub_w, n_ins_w, n_del_w))
            logger.info('CER (%s): %.2f %%' % (dataloader.set, cer))
            logger.info('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub_c, n_ins_c, n_del_c))
            logger.info('OOV (total): %d' % (n_oov_total))

        if oracle:
            wer_oracle /= n_word
            oracle_hit_rate = n_oracle_hit * 100 / n_utt
            logger.info('Oracle WER (%s): %.2f %%' % (dataloader.set, wer_oracle))
            logger.info('Oracle hit rate (%s): %.2f %%' % (dataloader.set, oracle_hit_rate))

        if fine_grained:
            for len_bin, wers in sorted(wer_dist.items(), key=lambda x: x[0]):
                logger.info('  WER (%s): %.2f %% (%d)' % (dataloader.set, sum(wers) / len(wers), len_bin))

    return wer, cer, n_oov_total
