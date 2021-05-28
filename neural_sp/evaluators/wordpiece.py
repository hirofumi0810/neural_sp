# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate wordpiece-level model by WER."""

import codecs
import logging
import numpy as np
from tqdm import tqdm

from neural_sp.evaluators.edit_distance import compute_wer
from neural_sp.utils import mkdir_join

logger = logging.getLogger(__name__)


def eval_wordpiece(models, dataloader, params, epoch=-1, rank=0,
                   save_dir=None, streaming=False, progressbar=False,
                   edit_distance=True, fine_grained=False, oracle=False,
                   teacher_force=False):
    """Evaluate a wordpiece-level model by WER.

    Args:
        models (List): models to evaluate
        dataloader (torch.utils.data.DataLoader): evaluation dataloader
        params (omegaconf.dictconfig.DictConfig): decoding hyperparameters
        epoch (int): current epoch
        rank (int): rank of current process group
        save_dir (str): directory path to save hypotheses
        streaming (bool): streaming decoding for session-level evaluation
        progressbar (bool): visualize progressbar
        edit_distance (bool): calculate edit-distance (can be skipped for RTF calculation)
        fine_grained (bool): calculate fine-grained WER distributions based on input lengths
        oracle (bool): calculate oracle WER
        teacher_force (bool): conduct decoding in teacher-forcing mode
    Returns:
        wer (float): Word error rate
        cer (float): Character error rate

    """
    if save_dir is None:
        save_dir = 'decode_' + dataloader.set + '_ep' + \
            str(epoch) + '_beam' + str(params.get('recog_beam_width'))
        save_dir += '_lp' + str(params.get('recog_length_penalty'))
        save_dir += '_cp' + str(params.get('recog_coverage_penalty'))
        save_dir += '_' + str(params.get('recog_min_len_ratio')) + '_' + \
            str(params.get('recog_max_len_ratio'))
        save_dir += '_lm' + str(params.get('recog_lm_weight'))

        ref_trn_path = mkdir_join(models[0].save_path, save_dir, 'ref.trn', rank=rank)
        hyp_trn_path = mkdir_join(models[0].save_path, save_dir, 'hyp.trn', rank=rank)
    else:
        ref_trn_path = mkdir_join(save_dir, 'ref.trn', rank=rank)
        hyp_trn_path = mkdir_join(save_dir, 'hyp.trn', rank=rank)

    wer, cer = 0, 0
    n_sub_w, n_ins_w, n_del_w = 0, 0, 0
    n_sub_c, n_ins_c, n_del_c = 0, 0, 0
    n_word, n_char = 0, 0
    wer_dist = {}  # calculate WER distribution based on input lengths

    wer_oracle = 0
    n_oracle_hit = 0

    n_streamable, quantity_rate, n_utt = 0, 0, 0
    last_success_frame_ratio = 0

    # Reset data counter
    dataloader.reset(params.get('recog_batch_size'), 'seq')

    if progressbar:
        pbar = tqdm(total=len(dataloader))

    if rank == 0:
        f_hyp = codecs.open(hyp_trn_path, 'w', encoding='utf-8')
        f_ref = codecs.open(ref_trn_path, 'w', encoding='utf-8')

    for batch in dataloader:
        speakers = batch['sessions' if dataloader.corpus == 'swbd' else 'speakers']
        if streaming or params.get('recog_block_sync'):
            nbest_hyps_id = models[0].decode_streaming(
                batch['xs'], params, dataloader.idx2token[0],
                exclude_eos=True,
                speaker=speakers[0])[0]
        else:
            nbest_hyps_id = models[0].decode(
                batch['xs'], params,
                idx2token=dataloader.idx2token[0],
                exclude_eos=True,
                refs_id=batch['ys'],
                utt_ids=batch['utt_ids'],
                speakers=speakers,
                ensemble_models=models[1:] if len(models) > 1 else [],
                teacher_force=teacher_force)[0]

        for b in range(len(batch['xs'])):
            ref = batch['text'][b]
            if ref[0] == '<':
                ref = ref.split('>')[1]
            nbest_hyps = [dataloader.idx2token[0](hyp_id) for hyp_id in nbest_hyps_id[b]]

            # Write to trn
            speaker = str(batch['speakers'][b]).replace('-', '_')
            if streaming:
                utt_id = str(batch['utt_ids'][b]) + '_0000000_0000001'
            else:
                utt_id = str(batch['utt_ids'][b])
            if rank == 0:
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

                if fine_grained:
                    xlen_bin = (batch['xlens'][b] // 200 + 1) * 200
                    if xlen_bin in wer_dist.keys():
                        wer_dist[xlen_bin] += [err_b / 100]
                    else:
                        wer_dist[xlen_bin] = [err_b / 100]

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

                if models[0].streamable():
                    n_streamable += len(batch['utt_ids'])
                else:
                    last_success_frame_ratio += models[0].last_success_frame_ratio()
                quantity_rate += models[0].quantity_rate()

        n_utt += len(batch['utt_ids'])
        if progressbar:
            pbar.update(len(batch['utt_ids']))

    if rank == 0:
        f_hyp.close()
        f_ref.close()
    if progressbar:
        pbar.close()

    # Reset data counters
    dataloader.reset(is_new_epoch=True)

    if edit_distance and not streaming:
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

        if params.get('recog_beam_width') > 1:
            logger.info('WER (%s): %.2f %%' % (dataloader.set, wer))
            logger.info('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub_w, n_ins_w, n_del_w))
            logger.info('CER (%s): %.2f %%' % (dataloader.set, cer))
            logger.info('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub_c, n_ins_c, n_del_c))

        if oracle:
            wer_oracle /= n_word
            oracle_hit_rate = n_oracle_hit * 100 / n_utt
            logger.info('Oracle WER (%s): %.2f %%' % (dataloader.set, wer_oracle))
            logger.info('Oracle hit rate (%s): %.2f %%' % (dataloader.set, oracle_hit_rate))

        if fine_grained:
            for len_bin, wers in sorted(wer_dist.items(), key=lambda x: x[0]):
                logger.info('  WER (%s): %.2f %% (%d)' % (dataloader.set, sum(wers) / len(wers), len_bin))

        logger.info('Streamability (%s): %.2f %%' % (dataloader.set, n_streamable * 100))
        logger.info('Quantity rate (%s): %.2f %%' % (dataloader.set, quantity_rate * 100))
        logger.info('Last success frame ratio (%s): %.2f %%' % (dataloader.set, last_success_frame_ratio))

    return wer, cer
