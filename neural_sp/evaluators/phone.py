# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate a phone-level model by PER."""

import codecs
import logging
import numpy as np
from tqdm import tqdm

from neural_sp.evaluators.edit_distance import compute_wer
from neural_sp.utils import mkdir_join

logger = logging.getLogger(__name__)


def eval_phone(models, dataloader, recog_params, epoch,
               recog_dir=None, streaming=False, progressbar=False,
               edit_distance=True, fine_grained=False, oracle=False, teacher_force=False):
    """Evaluate a phone-level model by PER.

    Args:
        models (List): models to evaluate
        dataloader (torch.utils.data.DataLoader): evaluation dataloader
        recog_params (omegaconf.dictconfig.DictConfig): decoding hyperparameters
        epoch (int): current epoch
        recog_dir (str): directory path to save hypotheses
        streaming (bool): streaming decoding for the session-level evaluation
        progressbar (bool): visualize the progressbar
        edit_distance (bool): calculate edit-distance (can be skipped for RTF calculation)
        fine_grained (bool): calculate fine-grained PER distributions based on input lengths
        oracle (bool): calculate oracle PER
        teacher_force (bool): conduct decoding in teacher-forcing mode
    Returns:
        per (float): Phone error rate

    """
    if recog_dir is None:
        recog_dir = 'decode_' + dataloader.set + '_ep' + \
            str(epoch) + '_beam' + str(recog_params.get('recog_beam_width'))
        recog_dir += '_lp' + str(recog_params.get('recog_length_penalty'))
        recog_dir += '_cp' + str(recog_params.get('recog_coverage_penalty'))
        recog_dir += '_' + str(recog_params.get('recog_min_len_ratio')) + '_' + \
            str(recog_params.get('recog_max_len_ratio'))

        ref_trn_path = mkdir_join(models[0].save_path, recog_dir, 'ref.trn')
        hyp_trn_path = mkdir_join(models[0].save_path, recog_dir, 'hyp.trn')
    else:
        ref_trn_path = mkdir_join(recog_dir, 'ref.trn')
        hyp_trn_path = mkdir_join(recog_dir, 'hyp.trn')

    per = 0
    n_sub, n_ins, n_del = 0, 0, 0
    n_phone = 0
    per_dist = {}  # calculate PER distribution based on input lengths

    per_oracle = 0
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
                nbest_hyps_id = models[0].decode(
                    batch['xs'], recog_params,
                    idx2token=dataloader.idx2token[0],
                    exclude_eos=True,
                    refs_id=batch['ys'],
                    utt_ids=batch['utt_ids'],
                    speakers=speakers,
                    ensemble_models=models[1:] if len(models) > 1 else [])[0]

            for b in range(len(batch['xs'])):
                ref = batch['text'][b]
                nbest_hyps = [dataloader.idx2token[0](hyp_id) for hyp_id in nbest_hyps_id[b]]

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
                    # Compute PER
                    err_b, sub_b, ins_b, del_b = compute_wer(ref=ref.split(' '),
                                                             hyp=nbest_hyps[0].split(' '))
                    per += err_b
                    n_sub += sub_b
                    n_ins += ins_b
                    n_del += del_b
                    n_phone += len(ref.split(' '))

                    # Compute oracle PER
                    if oracle and len(nbest_hyps) > 1:
                        pers_b = [err_b] + [compute_wer(ref=ref.split(' '),
                                                        hyp=hyp_n.split(' '))[0]
                                            for hyp_n in nbest_hyps[1:]]
                        oracle_idx = np.argmin(np.array(pers_b))
                        if oracle_idx == 0:
                            n_oracle_hit += len(batch['utt_ids'])
                        per_oracle += pers_b[oracle_idx]

                n_utt += len(batch['utt_ids'])
                if progressbar:
                    pbar.update(len(batch['utt_ids']))

    if progressbar:
        pbar.close()

    # Reset data counters
    dataloader.reset(is_new_epoch=True)

    if edit_distance and not streaming:
        per /= n_phone
        n_sub /= n_phone
        n_ins /= n_phone
        n_del /= n_phone

        if recog_params.get('recog_beam_width') > 1:
            logger.info('PER (%s): %.2f %%' % (dataloader.set, per))
            logger.info('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub, n_ins, n_del))

        if oracle:
            per_oracle /= n_phone
            oracle_hit_rate = n_oracle_hit * 100 / n_utt
            logger.info('Oracle PER (%s): %.2f %%' % (dataloader.set, per_oracle))
            logger.info('Oracle hit rate (%s): %.2f %%' % (dataloader.set, oracle_hit_rate))

        if fine_grained:
            for len_bin, pers in sorted(per_dist.items(), key=lambda x: x[0]):
                logger.info('  PER (%s): %.2f %% (%d)' % (dataloader.set, sum(pers) / len(pers), len_bin))

    return per
