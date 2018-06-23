#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method of RNNLMs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from tqdm import tqdm


def eval_ppl(models, dataset, progressbar=False):
    """Evaluate a RNNLM by perprexity.
    Args:
        models (list): the models to evaluate
        dataset: An instance of a `Dataset' class
        progressbar (bool): if True, visualize the progressbar
    Returns:
        ppl (float): Perplexity
    """
    # Reset data counter
    dataset.reset()

    model = models[0]
    # TODO: fix this

    loss = 0
    num_utt = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))  # TODO: fix this
    while True:
        batch, is_new_epoch = dataset.next(batch_size=1)

        if dataset.is_test:
            ys = []
            for b in range(len(batch['ys'])):
                if dataset.label_type == 'word':
                    indices = dataset.word2idx(batch['ys'][b])
                elif 'character' in dataset.label_type:
                    indices = dataset.char2idx(batch['ys'][b])
                else:
                    raise ValueError(dataset.label_type)
                ys += [indices]
                # NOTE: transcript is seperated by space('_')
        else:
            ys = batch['ys']

        loss += model(ys, is_eval=True)[0].data[0]
        num_utt += 1

        if progressbar:
            pbar.update(len(batch['ys']))

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    ppl = math.exp(loss / num_utt)

    return ppl
