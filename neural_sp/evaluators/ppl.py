#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate a RNNLM by perplexity."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tqdm import tqdm


def eval_ppl(models, dataset, batch_size=1, bptt=-1, n_caches=0, progressbar=False):
    """Evaluate a RNNLM by perprexity.

    Args:
        models (list): the models to evaluate
        dataset: An instance of a `Dataset' class
        batch_size (int):
        bptt (int):
        n_caches (int):
        progressbar (bool): if True, visualize the progressbar
    Returns:
        ppl (float): Perplexity

    """
    # Reset data counter
    dataset.reset()

    model = models[0]

    # Change to the evaluation mode
    model.eval()

    total_loss = 0
    ntokens = 0
    hidden = None
    if progressbar:
        pbar = tqdm(total=len(dataset))
    while True:
        ys, is_new_epoch = dataset.next(batch_size)
        bs = len(ys)

        for t in range(ys.shape[1] - 1):
            loss, hidden = model(ys[:, t:t + 2], hidden, is_eval=True, n_caches=n_caches)[:2]
            total_loss += loss.item() * bs
            ntokens += bs

            if progressbar:
                pbar.update(sum([len(y) for y in ys[:, t:t + 1]]))

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    ppl = np.exp(total_loss / ntokens)

    return ppl
