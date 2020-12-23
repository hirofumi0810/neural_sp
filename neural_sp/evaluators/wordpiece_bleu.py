# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate a wordpiece-level model by corpus-level BLEU."""

import codecs
import logging
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from neural_sp.utils import mkdir_join

logger = logging.getLogger(__name__)


def eval_wordpiece_bleu(models, dataloader, recog_params, epoch,
                        recog_dir=None, streaming=False, progressbar=False,
                        fine_grained=False, oracle=False, teacher_force=False):
    """Evaluate a wordpiece-level model by corpus-level BLEU.

    Args:
        models (List): models to evaluate
        dataloader (torch.utils.data.DataLoader): evaluation dataloader
        recog_params (dict):
        epoch (int):
        recog_dir (str):
        streaming (bool): streaming decoding for session-level evaluation
        progressbar (bool): visualize progressbar
        oracle (bool): calculate oracle corpsu-level BLEU
        fine_grained (bool): calculate fine-grained corpus-level BLEU distributions based on input lengths
        teacher_force (bool): conduct decoding in teacher-forcing mode
    Returns:
        c_bleu (float): corpus-level 4-gram BLEU

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

    list_of_references_dist = {}  # calculate corpus-level BLEU distribution bucketed by input lengths
    hypotheses_dist = {}

    hypotheses_oracle = []
    n_oracle_hit = 0
    n_utt = 0

    # Reset data counter
    dataloader.reset(recog_params['recog_batch_size'])

    if progressbar:
        pbar = tqdm(total=len(dataloader))

    list_of_references = []
    hypotheses = []

    with codecs.open(hyp_trn_path, 'w', encoding='utf-8') as f_hyp, \
            codecs.open(ref_trn_path, 'w', encoding='utf-8') as f_ref:
        while True:
            batch, is_new_epoch = dataloader.next(recog_params['recog_batch_size'])
            if streaming or recog_params['recog_chunk_sync']:
                nbest_hyps_id = models[0].decode_streaming(
                    batch['xs'], recog_params, dataloader.idx2token[0],
                    exclude_eos=True)[0]
            else:
                nbest_hyps_id = models[0].decode(
                    batch['xs'], recog_params,
                    idx2token=dataloader.idx2token[0] if progressbar else None,
                    exclude_eos=True,
                    refs_id=batch['ys'],
                    utt_ids=batch['utt_ids'],
                    speakers=batch['sessions' if dataloader.corpus == 'swbd' else 'speakers'],
                    ensemble_models=models[1:] if len(models) > 1 else [])[0]

            for b in range(len(batch['xs'])):
                ref = batch['text'][b]
                if ref[0] == '<':
                    ref = ref.split('>')[1]
                nbest_hyps = [dataloader.idx2token[0](hyp_id) for hyp_id in nbest_hyps_id[b]]

                # Write to trn
                # speaker = str(batch['speakers'][b]).replace('-', '_')
                if streaming:
                    utt_id = str(batch['utt_ids'][b]) + '_0000000_0000001'
                else:
                    utt_id = str(batch['utt_ids'][b])
                f_ref.write(ref + '\n')
                f_hyp.write(nbest_hyps[0] + '\n')
                logger.debug('utt-id: %s' % utt_id)
                logger.debug('Ref: %s' % ref)
                logger.debug('Hyp: %s' % nbest_hyps[0])
                logger.debug('-' * 150)

                if not streaming:
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
                            n_oracle_hit += 1
                        hypotheses_oracle += [nbest_hyps[oracle_idx].split(' ')]

                n_utt += 1
                if progressbar:
                    pbar.update(1)

            if is_new_epoch:
                break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataloader.reset()

    c_bleu = corpus_bleu(list_of_references, hypotheses) * 100

    if not streaming:
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

    logger.debug('Corpus-level BLEU (%s): %.2f %%' % (dataloader.set, c_bleu))

    return c_bleu
