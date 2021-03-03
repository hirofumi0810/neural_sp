# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate a character-level model by WER & CER."""

import codecs
import logging
import numpy as np
from tqdm import tqdm

from neural_sp.evaluators.edit_distance import compute_wer
from neural_sp.utils import mkdir_join

logger = logging.getLogger(__name__)


def eval_char(models, dataloader, recog_params, epoch,
              recog_dir=None, streaming=False, progressbar=False, task_idx=0,
              fine_grained=False, oracle=False, teacher_force=False):
    """Evaluate a character-level model by WER & CER.

    Args:
        models (List): models to evaluate
        dataloader (torch.utils.data.DataLoader): evaluation dataloader
        recog_params (omegaconf.dictconfig.DictConfig): decoding hyperparameters
        epoch (int):
        recog_dir (str):
        streaming (bool): streaming decoding for session-level evaluation
        progressbar (bool): visualize progressbar
        task_idx (int): index of target task in interest
            0: main task
            1: sub task
            2: sub sub task
        fine_grained (bool): calculate fine-grained WER distributions based on input lengths
        oracle (bool): calculate oracle WER
        teacher_force (bool): conduct decoding in teacher-forcing mode
    Returns:
        wer (float): Word error rate
        cer (float): Character error rate

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
    cer_dist = {}  # calculate CER distribution based on input lengths

    cer_oracle = 0
    n_oracle_hit = 0

    n_streamable, quantity_rate, n_utt = 0, 0, 0
    last_success_frame_ratio = 0

    # Reset data counter
    dataloader.reset(recog_params.get('recog_batch_size'))

    if progressbar:
        pbar = tqdm(total=len(dataloader))

    if task_idx == 0:
        task = 'ys'
    elif task_idx == 1:
        task = 'ys_sub1'
    elif task_idx == 2:
        task = 'ys_sub2'
    elif task_idx == 3:
        task = 'ys_sub3'

    with codecs.open(hyp_trn_path, 'w', encoding='utf-8') as f_hyp, \
            codecs.open(ref_trn_path, 'w', encoding='utf-8') as f_ref:
        while True:
            batch, is_new_epoch = dataloader.next(recog_params.get('recog_batch_size'))
            if streaming or recog_params.get('recog_block_sync'):
                nbest_hyps_id = models[0].decode_streaming(
                    batch['xs'], recog_params, dataloader.idx2token[0],
                    exclude_eos=True)[0]
            else:
                nbest_hyps_id = models[0].decode(
                    batch['xs'], recog_params,
                    idx2token=dataloader.idx2token[0],
                    exclude_eos=True,
                    refs_id=batch['ys'] if task_idx == 0 else batch['ys_sub' + str(task_idx)],
                    utt_ids=batch['utt_ids'],
                    speakers=batch['sessions' if dataloader.corpus == 'swbd' else 'speakers'],
                    task=task,
                    ensemble_models=models[1:] if len(models) > 1 else [])[0]

            for b in range(len(batch['xs'])):
                ref = batch['text'][b]
                nbest_hyps_tmp = [dataloader.idx2token[0](hyp_id) for hyp_id in nbest_hyps_id[b]]

                # Truncate the first and last spaces for the char_space unit
                nbest_hyps = []
                for hyp in nbest_hyps_tmp:
                    if len(hyp) > 0 and hyp[0] == ' ':
                        hyp = hyp[1:]
                    if len(hyp) > 0 and hyp[-1] == ' ':
                        hyp = hyp[:-1]
                    nbest_hyps.append(hyp)

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

                if not streaming:
                    if ('char' in dataloader.unit and 'nowb' not in dataloader.unit) or (task_idx > 0 and dataloader.unit_sub1 == 'char'):
                        # Compute WER
                        err_b, sub_b, ins_b, del_b = compute_wer(ref=ref.split(' '),
                                                                 hyp=nbest_hyps[0].split(' '))
                        wer += err_b
                        n_sub_w += sub_b
                        n_ins_w += ins_b
                        n_del_w += del_b
                        n_word += len(ref.split(' '))
                        # NOTE: sentence error rate for Chinese

                    # Compute CER
                    if dataloader.corpus == 'csj':
                        ref = ref.replace(' ', '')
                        nbest_hyps[0] = nbest_hyps[0].replace(' ', '')
                    err_b, sub_b, ins_b, del_b = compute_wer(ref=list(ref),
                                                             hyp=list(nbest_hyps[0]))
                    cer += err_b
                    n_sub_c += sub_b
                    n_ins_c += ins_b
                    n_del_c += del_b
                    n_char += len(ref)

                    # Compute oracle CER
                    if oracle and len(nbest_hyps) > 1:
                        cers_b = [err_b] + [compute_wer(ref=list(ref),
                                                        hyp=list(hyp_n))[0]
                                            for hyp_n in nbest_hyps[1:]]
                        oracle_idx = np.argmin(np.array(cers_b))
                        if oracle_idx == 0:
                            n_oracle_hit += len(batch['utt_ids'])
                        cer_oracle += cers_b[oracle_idx]

                    if fine_grained:
                        xlen_bin = (batch['xlens'][b] // 200 + 1) * 200
                        if xlen_bin in cer_dist.keys():
                            cer_dist[xlen_bin] += [err_b / 100]
                        else:
                            cer_dist[xlen_bin] = [err_b / 100]

                    if models[0].streamable():
                        n_streamable += len(batch['utt_ids'])
                    else:
                        last_success_frame_ratio += models[0].last_success_frame_ratio()
                    quantity_rate += models[0].quantity_rate()

                n_utt += len(batch['utt_ids'])
                if progressbar:
                    pbar.update(len(batch['utt_ids']))

            if is_new_epoch:
                break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataloader.reset()

    if not streaming:
        if ('char' in dataloader.unit and 'nowb' not in dataloader.unit) or (task_idx > 0 and dataloader.unit_sub1 == 'char'):
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

        if n_utt - n_streamable > 0:
            last_success_frame_ratio /= (n_utt - n_streamable)
        n_streamable /= n_utt
        quantity_rate /= n_utt

        if recog_params.get('recog_beam_width') > 1:
            logger.info('WER (%s): %.2f %%' % (dataloader.set, wer))
            logger.info('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub_w, n_ins_w, n_del_w))
            logger.info('CER (%s): %.2f %%' % (dataloader.set, cer))
            logger.info('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub_c, n_ins_c, n_del_c))

        if oracle:
            cer_oracle /= n_char
            oracle_hit_rate = n_oracle_hit * 100 / n_utt
            logger.info('Oracle CER (%s): %.2f %%' % (dataloader.set, cer_oracle))
            logger.info('Oracle hit rate (%s): %.2f %%' % (dataloader.set, oracle_hit_rate))

        if fine_grained:
            for len_bin, cers in sorted(cer_dist.items(), key=lambda x: x[0]):
                logger.info('  CER (%s): %.2f %% (%d)' % (dataloader.set, sum(cers) / len(cers), len_bin))

        logger.info('Streamability (%s): %.2f %%' % (dataloader.set, n_streamable * 100))
        logger.info('Quantity rate (%s): %.2f %%' % (dataloader.set, quantity_rate * 100))
        logger.info('Last success frame ratio (%s): %.2f %%' % (dataloader.set, last_success_frame_ratio))

    return wer, cer
