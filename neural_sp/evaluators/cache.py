#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Store words for OOV caching in advance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from tqdm import tqdm


def store_cache(models, dataset, recog_params, progressbar=False):
    """Store words for OOV caching in advance.

    Args:
        models (list): models to evaluate
        dataset: An instance of a `Dataset' class
        recog_params (dict):
        progressbar (bool): visualize the progressbar

    """
    # Reset data counter
    dataset.reset()

    if progressbar:
        pbar = tqdm(total=len(dataset))  # TODO(hirofumi): fix this

    recog_params_store = copy.deepcopy(recog_params)
    recog_params_store['recog_rnnlm_weight'] = 0
    recog_params_store['recog_beam_width'] = 1
    recog_params_store['recog_n_caches'] = 0
    recog_params_store['recog_oracle'] = True

    while True:
        batch, is_new_epoch = dataset.next(recog_params_store['recog_batch_size'])

        best_hyps_id, best_hyps_str, aws, perm_ids, _ = models[0].decode(
            batch['xs'], recog_params_store, dataset.idx2token[0],
            exclude_eos=True,
            refs_id=batch['ys'],
            refs_text=batch['text'],
            ensemble_models=models[1:] if len(models) > 1 else [],
            speakers=batch['sessions'] if dataset.corpus == 'swbd' else batch['speakers'],
            store_cache=True,
            word_list=[])
        # word_list: index

        if progressbar:
            pbar.update(len(batch['xs']))

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()
