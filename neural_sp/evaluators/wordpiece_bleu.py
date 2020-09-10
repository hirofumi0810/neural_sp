#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate the wordpiece-level model by BLEU."""

import codecs
import logging
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from neural_sp.utils import mkdir_join

logger = logging.getLogger(__name__)


def eval_wordpiece_bleu(models, dataloader, recog_params, epoch,
                        recog_dir=None, streaming=False, progressbar=False,
                        fine_grained=False):
    """Evaluate the wordpiece-level model by BLEU.

    Args:
        models (list): models to evaluate
        dataloader (torch.utils.data.DataLoader): evaluation dataloader
        recog_params (dict):
        epoch (int):
        recog_dir (str):
        streaming (bool): streaming decoding for the session-level evaluation
        progressbar (bool): visualize the progressbar
        fine_grained (bool): calculate fine-grained BLEU distributions based on input lengths
    Returns:
        bleu (float): 4-gram BLEU

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

    s_bleu = 0
    n_sentence = 0
    s_bleu_dist = {}  # calculate sentence-level BLEU distribution based on input lengths

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
                best_hyps_id, _ = models[0].decode_streaming(
                    batch['xs'], recog_params, dataloader.idx2token[0],
                    exclude_eos=True)
            else:
                best_hyps_id, _ = models[0].decode(
                    batch['xs'], recog_params,
                    idx2token=dataloader.idx2token[0] if progressbar else None,
                    exclude_eos=True,
                    refs_id=batch['ys'],
                    utt_ids=batch['utt_ids'],
                    speakers=batch['sessions' if dataloader.corpus == 'swbd' else 'speakers'],
                    ensemble_models=models[1:] if len(models) > 1 else [])

            for b in range(len(batch['xs'])):
                ref = batch['text'][b]
                if ref[0] == '<':
                    ref = ref.split('>')[1]
                hyp = dataloader.idx2token[0](best_hyps_id[b])

                # Write to trn
                # speaker = str(batch['speakers'][b]).replace('-', '_')
                if streaming:
                    utt_id = str(batch['utt_ids'][b]) + '_0000000_0000001'
                else:
                    utt_id = str(batch['utt_ids'][b])
                f_ref.write(ref + '\n')
                f_hyp.write(hyp + '\n')
                logger.debug('utt-id: %s' % utt_id)
                logger.debug('Ref: %s' % ref)
                logger.debug('Hyp: %s' % hyp)
                logger.debug('-' * 150)

                if not streaming:
                    list_of_references += [[ref.split(' ')]]
                    hypotheses += [hyp.split(' ')]
                    n_sentence += 1

                    # Compute sentence-level BLEU
                    if fine_grained:
                        s_bleu_b = sentence_bleu([ref.split(' ')], hyp.split(' '))
                        s_bleu += s_bleu_b * 100

                        xlen_bin = (batch['xlens'][b] // 200 + 1) * 200
                        if xlen_bin in s_bleu_dist.keys():
                            s_bleu_dist[xlen_bin] += [s_bleu_b / 100]
                        else:
                            s_bleu_dist[xlen_bin] = [s_bleu_b / 100]

                if progressbar:
                    pbar.update(1)

            if is_new_epoch:
                break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataloader.reset()

    c_bleu = corpus_bleu(list_of_references, hypotheses) * 100
    if not streaming and fine_grained:
        s_bleu /= n_sentence
        for len_bin, s_bleus in sorted(s_bleu_dist.items(), key=lambda x: x[0]):
            logger.info('  sentence-level BLEU (%s): %.2f %% (%d)' %
                        (dataloader.set, sum(s_bleus) / len(s_bleus), len_bin))

    logger.debug('Corpus-level BLEU (%s): %.2f %%' % (dataloader.set, c_bleu))

    return c_bleu
