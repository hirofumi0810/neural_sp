# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate wordpiece-level model by corpus-level BLEU."""

import codecs
import logging
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from neural_sp.utils import mkdir_join

logger = logging.getLogger(__name__)


def eval_wordpiece_bleu(models, dataloader, params, epoch, rank=0,
                        save_dir=None, streaming=False, progressbar=False,
                        edit_distance=True, fine_grained=False, oracle=False,
                        teacher_force=False):
    """Evaluate a wordpiece-level model by corpus-level BLEU.

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
        fine_grained (bool): calculate fine-grained corpus-level BLEU distributions based on input lengths
        oracle (bool): calculate oracle corpsu-level BLEU
        teacher_force (bool): conduct decoding in teacher-forcing mode
    Returns:
        c_bleu (float): corpus-level 4-gram BLEU

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

    list_of_references_dist = {}  # calculate corpus-level BLEU distribution bucketed by input lengths
    hypotheses_dist = {}

    hypotheses_oracle = []
    n_oracle_hit = 0
    n_utt = 0

    # Reset data counter
    dataloader.reset(params.get('recog_batch_size'), 'seq')

    if progressbar:
        pbar = tqdm(total=len(dataloader))

    list_of_references = []
    hypotheses = []

    if rank == 0:
        f_hyp = codecs.open(hyp_trn_path, 'w', encoding='utf-8')
        f_ref = codecs.open(ref_trn_path, 'w', encoding='utf-8')

    for batch in dataloader:
        if streaming or params.get('recog_block_sync'):
            nbest_hyps_id = models[0].decode_streaming(
                batch['xs'], params, dataloader.idx2token[0],
                exclude_eos=True,
                speaker=batch['speakers'][0])[0]
        else:
            nbest_hyps_id = models[0].decode(
                batch['xs'], params,
                idx2token=dataloader.idx2token[0],
                exclude_eos=True,
                refs_id=batch['ys'],
                utt_ids=batch['utt_ids'],
                speakers=batch['speakers'],
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
                list_of_references += [[ref.split(' ')]]
                hypotheses += [nbest_hyps[0].split(' ')]

                if fine_grained:
                    xlen_bin = (batch['xlens'][b] // 200 + 1) * 200
                    if xlen_bin in hypotheses_dist.keys():
                        list_of_references_dist[xlen_bin] += [[ref.split(' ')]]
                        hypotheses_dist[xlen_bin] += [hypotheses[-1]]
                    else:
                        list_of_references_dist[xlen_bin] = [[ref.split(' ')]]
                        hypotheses_dist[xlen_bin] = [hypotheses[-1]]

                # Compute oracle corpus-level BLEU (selected by sentence-level BLEU)
                if oracle and len(nbest_hyps) > 1:
                    s_blues_b = [sentence_bleu(ref.split(' '), hyp_n.split(' '))
                                 for hyp_n in nbest_hyps]
                    oracle_idx = np.argmax(np.array(s_blues_b))
                    if oracle_idx == 0:
                        n_oracle_hit += len(batch['utt_ids'])
                    hypotheses_oracle += [nbest_hyps[oracle_idx].split(' ')]

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

    c_bleu = corpus_bleu(list_of_references, hypotheses) * 100

    if edit_distance and not streaming:
        if oracle:
            c_bleu_oracle = corpus_bleu(list_of_references, hypotheses_oracle) * 100
            oracle_hit_rate = n_oracle_hit * 100 / n_utt
            logger.info('Oracle corpus-level BLEU (%s): %.2f %%' % (dataloader.set, c_bleu_oracle))
            logger.info('Oracle hit rate (%s): %.2f %%' % (dataloader.set, oracle_hit_rate))

        if fine_grained:
            for len_bin, hypotheses_bin in sorted(hypotheses_dist.items(), key=lambda x: x[0]):
                c_bleu_bin = corpus_bleu(list_of_references_dist[len_bin], hypotheses_bin) * 100
                logger.info('  corpus-level BLEU (%s): %.2f %% (%d)' %
                            (dataloader.set, c_bleu_bin, len_bin))

    logger.info('Corpus-level BLEU (%s): %.2f %%' % (dataloader.set, c_bleu))

    return c_bleu
